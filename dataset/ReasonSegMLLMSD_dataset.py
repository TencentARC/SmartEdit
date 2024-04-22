import pdb
import copy
import os
import random
import torch
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from conversation_v01 import SeparatorStyle, get_conv_template
from einops import rearrange
import numpy as np
from PIL import Image
import json
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# ReasoningSegmentation dataset
class ReasoningSegmentation_Dataset(Dataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 ReasoningSegmentationDataset_json_path,
                 ReasoningSegmentationDataset_image_path,
                 ReasoningSegmentationDataset_binary_mask_path,
                 ReasoningSegmentationDataset_resolution_ViT,
                 ReasoningSegmentationDataset_resolution_for_SD,
                 transparency,
                 CLIPImageProcessor,
                 mm_projection_length,
                 editing_template,
                 editing_max_length,
                 llm_tokenizer=None,
                 InstructDiffusion_seg_template=None,
                 InstructDiffusion_color_template=None
                 ):

        # ReasoningSegmentation Dataset path
        # json + image + binary mask
        self.ReasoningSegmentationDataset_json_path = ReasoningSegmentationDataset_json_path
        self.ReasoningSegmentationDataset_image_path = ReasoningSegmentationDataset_image_path
        self.ReasoningSegmentationDataset_binary_mask_path = ReasoningSegmentationDataset_binary_mask_path
        self.transparency = transparency

        # segmentation template
        seg_diverse_prompt_path = InstructDiffusion_seg_template
        self.seg_diverse_prompt_list = []
        with open(seg_diverse_prompt_path) as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                self.seg_diverse_prompt_list.append(line)
                line = f.readline()

        # color template
        color_list_file_path = InstructDiffusion_color_template
        self.color_list = []
        with open(color_list_file_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(" ")
                if len(line_split) > 1:
                    temp = []
                    for i in range(4):
                        temp.append(line_split[i])
                    self.color_list.append(temp)
                line = f.readline()

        # get the json file -> len(jsonfiles)=239
        self.ReasoningSegmentation_jsonfiles = []
        for path, subdirs, files in os.walk(self.ReasoningSegmentationDataset_json_path):
            for name in files:
                if name.endswith('.json'):
                    self.ReasoningSegmentation_jsonfiles.append(os.path.join(path, name))

        # 224, 256
        self.ReasoningSegmentationDataset_resolution_ViT = ReasoningSegmentationDataset_resolution_ViT
        self.ReasoningSegmentationDataset_resolution_for_SD = ReasoningSegmentationDataset_resolution_for_SD

        # CLIPImageProcessor
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
        return len(self.ReasoningSegmentation_jsonfiles)

    def __getitem__(self, index):
        # read json and image path
        jsonfile = self.ReasoningSegmentation_jsonfiles[index]

        # meta-file name
        meta_file_name = jsonfile.split("train_new_231114/")[1].split('.json')[0]
        original_image_path = self.ReasoningSegmentationDataset_image_path + "/" + meta_file_name + ".jpg"
        mask_path = self.ReasoningSegmentationDataset_binary_mask_path + "/" + meta_file_name + ".jpg"
        with open(jsonfile, 'r') as f:
            data = json.load(f)
            random_element = random.choice(data['text'])
            object_name = data['Name']

        ####################################################################################
        # Vicuna conversation system construction for image editing task...
        # Step 0. Make prompts
        prompt_for_reasoning_seg = random.choice(self.seg_diverse_prompt_list)
        color = random.choice(self.color_list)
        color_name = color[0]
        prompt_for_reasoning_seg = random_element + " " + prompt_for_reasoning_seg.format(color=color_name.lower(), object=object_name.lower())
        # print(prompt_for_reasoning_seg)

        # 1. Original Image for ViT input
        original_image = Image.open(original_image_path).convert("RGB")
        original_image_ViT = copy.deepcopy(original_image)
        original_image_ViT = original_image_ViT.resize((self.ReasoningSegmentationDataset_resolution_ViT, self.ReasoningSegmentationDataset_resolution_ViT),
                                                       resample=Image.Resampling.BICUBIC)
        original_image_ViT = self.CLIPImageProcessor.preprocess(original_image_ViT, return_tensors='pt')['pixel_values']
        original_image_ViT = original_image_ViT[0]

        # 2. Original Image & 3. Edited Image for SD input
        original_image_SD = copy.deepcopy(original_image)
        original_image_SD = original_image_SD.resize((self.ReasoningSegmentationDataset_resolution_for_SD, self.ReasoningSegmentationDataset_resolution_for_SD), resample=Image.Resampling.BICUBIC)
        original_image_SD = np.asarray(original_image_SD, dtype=np.uint8)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.ReasoningSegmentationDataset_resolution_for_SD, self.ReasoningSegmentationDataset_resolution_for_SD), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask, dtype=np.int64)
        mask = (mask >= 255)

        # give masks
        original_image_SD_new = Image.fromarray(original_image_SD)
        edited_img_SD = copy.deepcopy(original_image_SD)
        R, G, B = color[3].split(",")
        R = int(R)
        G = int(G)
        B = int(B)
        edited_img_SD[:, :, 0][mask] = self.transparency * edited_img_SD[:, :, 0][mask] + (1 - self.transparency) * R
        edited_img_SD[:, :, 1][mask] = self.transparency * edited_img_SD[:, :, 1][mask] + (1 - self.transparency) * G
        edited_img_SD[:, :, 2][mask] = self.transparency * edited_img_SD[:, :, 2][mask] + (1 - self.transparency) * B
        edited_img_SD = Image.fromarray(edited_img_SD)
        original_image_SD_new = original_image_SD_new.resize((self.ReasoningSegmentationDataset_resolution_for_SD, self.ReasoningSegmentationDataset_resolution_for_SD), Image.Resampling.BICUBIC)
        edited_img_SD = edited_img_SD.resize((self.ReasoningSegmentationDataset_resolution_for_SD, self.ReasoningSegmentationDataset_resolution_for_SD), Image.Resampling.BICUBIC)
        original_image_SD_new = rearrange(2 * torch.tensor(np.array(original_image_SD_new)).float() / 255 - 1, "h w c -> c h w")
        edited_img_SD = rearrange(2 * torch.tensor(np.array(edited_img_SD)).float() / 255 - 1, "h w c -> c h w")

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
        edited_prompt = prompt_for_reasoning_seg
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
        input_ids_max_len = (self.editing_max_length + 100) - self.mm_projection_length
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

        # Reasoning-Segmentation dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': original_image_ViT,
                'original_img_for_vae': original_image_SD_new,
                'edited_img': edited_img_SD,
                'input_ids': input_ids,
                'input_attention_mask': RE_instruction_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}
