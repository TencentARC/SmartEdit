import pdb
import os
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from conversation_v01 import SeparatorStyle, get_conv_template
from PIL import Image
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

import json
import copy
DEFAULT_IMAGE_TOKEN = '<image>'
IGNORE_INDEX = -100

def tokenizer_image_token_(prompt, tokenizer, image_token_index, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    # len(prompt_chunks)=2

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    # return_tensors or not
    if return_tensors is not None:
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            return input_ids
    else:
        return input_ids

# LLaVA dataset
class LLaVADataset_for_instruction_tuning(Dataset):
    """ LLAVA-Dataset for instruction tuning """
    def __init__(self,
                 data_path,
                 image_folder,
                 LLM_tokenizer,
                 CLIPImageProcessor,
                 is_LLaMA,
                 LLaVADataset_resolution_ViT
                 ):
        super(LLaVADataset_for_instruction_tuning, self).__init__()
        # LLaVA dataset
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder

        # LLM tokenizer
        self.LLM_tokenizer = LLM_tokenizer
        self.is_LLaMA = is_LLaMA

        # CLIPImageProcessor
        self.CLIPImageProcessor = CLIPImageProcessor
        self.LLaVADataset_resolution_ViT = LLaVADataset_resolution_ViT

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        # 1. image -> [3, 224, 224]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_folder
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image = image.resize((self.LLaVADataset_resolution_ViT, self.LLaVADataset_resolution_ViT), resample=Image.Resampling.BICUBIC)
        image = self.CLIPImageProcessor.preprocess(image, return_tensors='pt')['pixel_values']
        image = image[0]

        # 2. preprocess_multimodal function
        # DEFAULT_IMAGE_TOKEN='<image>', DEFAULT_IM_START_TOKEN='<im_start>', DEFAULT_IM_END_TOKEN='<im_end>'
        sources = copy.deepcopy([e["conversations"] for e in sources])
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    # '<image>\nWhat are the colors of the bus in the image?'
                replace_token = DEFAULT_IMAGE_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
                # '<im_start> <image> <im_end>\nWhat are the colors of the bus in the image?'

        # 3. preprocess function
        # Step-1: choose conversation system message
        assert ('image' in self.list_data_dict[i]) == True
        conv = get_conv_template("vicuna_v1.3")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        # A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. -> {'human': 'USER', 'gpt': 'ASSISTANT'}

        # Step-2: Apply prompt templates for conversation
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # data processing for LLaMA
        input_ids_for_LLM, targets_for_LLM = None, None
        if self.is_LLaMA == True:
            # Step-3: Tokenize conversations
            input_ids_for_LLM = torch.stack([tokenizer_image_token_(prompt, self.LLM_tokenizer,
                                                                    image_token_index=self.LLM_tokenizer.img_start_token_id,
                                                                    return_tensors='pt') for prompt in conversations], dim=0)
            targets_for_LLM = input_ids_for_LLM.clone()
            assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

            # Step-4: Mask targets
            sep = conv.sep + conv.roles[1] + ": "
            for conversation, target_for_LLM in zip(conversations, targets_for_LLM):
                total_len = int(target_for_LLM.ne(self.LLM_tokenizer.pad_token_id).sum())
                rounds = conversation.split(conv.sep2)
                cur_len = 1
                target_for_LLM[:cur_len] = IGNORE_INDEX
                for i, rou in enumerate(rounds):
                    if rou == "":
                        break

                    parts = rou.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep

                    round_len = len(tokenizer_image_token_(rou, self.LLM_tokenizer, image_token_index=self.LLM_tokenizer.img_start_token_id))
                    instruction_len = len(tokenizer_image_token_(parts[0], self.LLM_tokenizer, image_token_index=self.LLM_tokenizer.img_start_token_id)) - 2

                    target_for_LLM[cur_len: cur_len + instruction_len] = IGNORE_INDEX
                    cur_len += round_len
                target_for_LLM[cur_len:] = IGNORE_INDEX

                if cur_len < self.LLM_tokenizer.model_max_length:
                    if cur_len != total_len:
                        target_for_LLM[:] = IGNORE_INDEX
                        print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

        # return dataloader -> 'original_img_for_vae' and 'edited_img' are placeholders
        original_img = image
        original_img_for_vae = torch.zeros([3, 256, 256], dtype=torch.float32)
        edited_img = torch.zeros([3, 256, 256], dtype=torch.float32)

        # For input_ids -> insert '<im_start>' and '<im_end>'
        input_ids_ = input_ids_for_LLM[0]
        LLM_img_start_token_id = self.LLM_tokenizer.img_start_token_id
        LLM_img_start_token_id_pos = (torch.where(input_ids_ == LLM_img_start_token_id)[0])[0].item()
        new_input_ids_ = torch.cat([input_ids_[:LLM_img_start_token_id_pos],
                                    torch.tensor([self.LLM_tokenizer.DEFAULT_IM_START_TOKEN]),
                                    input_ids_[LLM_img_start_token_id_pos:(LLM_img_start_token_id_pos + 1)],
                                    torch.tensor([self.LLM_tokenizer.DEFAULT_IM_END_TOKEN]),
                                    input_ids_[(LLM_img_start_token_id_pos + 1):]], dim=0)
        input_attention_mask = new_input_ids_.ne(self.LLM_tokenizer.pad_token_id)

        # For generated_caption_targets -> insert 2*IGNORE_INDEX
        generated_caption_targets = torch.cat([torch.tensor([IGNORE_INDEX]), torch.tensor([IGNORE_INDEX]),
                                               targets_for_LLM[0]], dim=0)
        generated_caption_encoder_attention_mask = new_input_ids_.ge(self.LLM_tokenizer.img_start_token_id)

        # task choosing
        is_editing_task = torch.zeros(1)

        # LLaVA-Dataset dataloader
        return {'original_img': original_img,
                'original_img_for_vae': original_img_for_vae,
                'edited_img': edited_img,
                'input_ids': new_input_ids_,
                'input_attention_mask': input_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}
