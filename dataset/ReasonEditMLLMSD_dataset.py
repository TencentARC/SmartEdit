import pdb

import copy
import json
import numpy as np
from conversation_v01 import SeparatorStyle, get_conv_template
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def convert_to_np(image, resolution):
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    return np.array(image).transpose(2, 0, 1)

# ReasoningEditing dataset
class ReasoningEditing_Dataset(Dataset):
    def __init__(self,
                 ReasoningEditingDataset_path,
                 ReasoningEditingDataset_resolution_ViT,
                 ReasoningEditingDataset_resolution_for_SD,
                 CLIPImageProcessor,
                 mm_projection_length,
                 editing_template,
                 editing_max_length,
                 llm_tokenizer=None
                 ):

        # ReasoningEditing Dataset path
        with open(ReasoningEditingDataset_path, 'r') as f:
            self.ReasoningEditing_data = json.load(f)

        # 224, 256
        self.ReasoningEditingDataset_resolution_ViT = ReasoningEditingDataset_resolution_ViT
        self.ReasoningEditingDataset_resolution_for_SD = ReasoningEditingDataset_resolution_for_SD

        # CLIPImageProcessor -> 没有flip操作
        self.CLIPImageProcessor = CLIPImageProcessor

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'

        # Vicuna conversation system for editing
        self.editing_template = editing_template
        self.editing_max_length = editing_max_length
        self.mm_projection_length = mm_projection_length

    def __len__(self,):
        return len(self.ReasoningEditing_data)

    def __getitem__(self, index):
        # load variables from json file
        key = f'{index:04d}'
        original_img_path = self.ReasoningEditing_data[key]['origin_img_path']
        original_image = Image.open(original_img_path).convert('RGB')
        target_img_path = self.ReasoningEditing_data[key]['target_img_path']
        target_image = Image.open(target_img_path).convert('RGB')

        # random select an instruction
        instruction_list = self.ReasoningEditing_data[key]['instruction']
        instruction = random.choice(instruction_list)

        # 1. Original Image for ViT input
        RE_original_image = copy.deepcopy(original_image)
        RE_original_image = RE_original_image.resize((self.ReasoningEditingDataset_resolution_ViT, self.ReasoningEditingDataset_resolution_ViT),
                                                     resample=Image.Resampling.BICUBIC)
        RE_original_image = self.CLIPImageProcessor.preprocess(RE_original_image, return_tensors='pt')['pixel_values']
        RE_original_image = RE_original_image[0]

        # 2. Original Image & 3. Edited Image for SD input
        RE_original_image_2 = convert_to_np(original_image, self.ReasoningEditingDataset_resolution_for_SD)
        RE_target_image = convert_to_np(target_image, self.ReasoningEditingDataset_resolution_for_SD)
        RE_SD_input = np.concatenate([RE_original_image_2, RE_target_image])
        RE_SD_input = torch.tensor(RE_SD_input)
        RE_SD_input = 2 * (RE_SD_input / 255) - 1
        RE_original_image_2, RE_target_image = RE_SD_input.chunk(2)

        ####################################################################################
        # Vicuna conversation system construction for image editing task...
        # Step 1. Choose Human-GPT templates
        conversation_templates = []
        with open(self.editing_template, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('Human: '):
                    d = dict()
                    d['Human'] = line[len("Human: "):]
                    conversation_templates.append(d)
                elif line.startswith('GPT: '):
                    conversation_templates[-1]['GPT'] = line[len("GPT: "):]

        # Step 2. Choose Vicuna_v1.3 system message
        conv = get_conv_template("vicuna_v1.3")
        roles = {"Human": conv.roles[0], "GPT": conv.roles[1]}

        # <img_i> tokens -> num_new_tokens=35: "<img>"(system message) + " <img_0> ... <img_31>"
        num_new_tokens = len(self.llm_tokenizer) - self.llm_tokenizer.vocab_size
        append_str = ""
        for i in range(num_new_tokens - 3):
            append_str += f" <img_{i}>"

        # Step 3. Vicuna conversation system construction
        """ "<img_0>" is a placeholder to find the text position and insert image embeddings """
        edited_prompt = instruction
        DEFAULT_IM_START_TOKEN = '<im_start>'
        DEFAULT_IM_END_TOKEN = '<im_end>'
        edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
        conversation_template = random.choice(conversation_templates)
        conv.messages = []
        conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
        conv.append_message(roles["GPT"], conversation_template["GPT"].replace(' [img].', append_str))
        conversation = conv.get_prompt()
        conversation = conversation.replace("\n", "")

        # 4. Edited Prompt input_ids -> Tokenize conversations
        input_ids_max_len = self.editing_max_length - self.mm_projection_length
        input_ids = self.llm_tokenizer(
            conversation,
            return_tensors="pt",
            padding="max_length",
            max_length=input_ids_max_len,
            truncation=True,
        ).input_ids[0]
        # [(editing_max_length-mm_projection_length)=256]

        # Step 4. Only show up tokens after 'ASSISTANT:'
        # IGNORE_TOKEN_ID=-100
        generated_caption_targets = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        generated_caption_targets[:1] = IGNORE_TOKEN_ID
        total_padding_len = int(generated_caption_targets.ne(self.llm_tokenizer.pad_token_id).sum())
        parts = conversation.split(sep)
        parts[0] += sep

        # 5. Generated caption targets for Language Model loss
        instruction_len = len(
            self.llm_tokenizer(
                parts[0],
                max_length=input_ids_max_len,
                truncation=True,
            ).input_ids) - 2
        generated_caption_targets[1:(1 + instruction_len)] = IGNORE_TOKEN_ID
        generated_caption_targets[total_padding_len:] = IGNORE_TOKEN_ID
        # [(editing_max_length-mm_projection_length)=256]
        ####################################################################################

        # 6. Edited Prompt attention_mask
        # ne(a, b) is a != b
        RE_instruction_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        # ge(a, b) is a >= b
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # Reasoning-Editing dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': RE_original_image,
                'original_img_for_vae': RE_original_image_2,
                'edited_img': RE_target_image,
                'input_ids': input_ids,
                'input_attention_mask': RE_instruction_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}
