import pdb

from datasets import load_from_disk
import io
import numpy as np
from conversation_v01 import SeparatorStyle, get_conv_template
from PIL import Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def convert_to_np(image, resolution):
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    return np.array(image).transpose(2, 0, 1)

# InstructPix2Pix dataset
class InstructPix2Pix_Dataset(Dataset):
    '''
    according to InstructPix2Pix, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'edit_prompt'. 'original_image' can be used with the 'edit_prompt' and 'edited_image' denotes the image after applying the 'edit_prompt' on the 'original_image'.
    "original_image" + "edited_image" + "edit_prompt"
    '''
    def __init__(self,
                 InstructPix2PixDataset_path,
                 InstructPix2PixDataset_resolution_ViT,
                 InstructPix2PixDataset_resolution_for_SD,
                 CLIPImageProcessor,
                 mm_projection_length,
                 editing_template,
                 editing_max_length,
                 llm_tokenizer=None
                 ):

        # InstructPix2Pix Dataset path
        self.InstructPix2PixDataset_path = load_from_disk(InstructPix2PixDataset_path)
        # 224, 256
        self.InstructPix2PixDataset_resolution_ViT = InstructPix2PixDataset_resolution_ViT
        self.InstructPix2PixDataset_resolution_for_SD = InstructPix2PixDataset_resolution_for_SD

        # CLIPImageProcessor
        self.CLIPImageProcessor = CLIPImageProcessor
        # SD transformation
        self.SD_transform = transforms.Compose([transforms.CenterCrop(self.InstructPix2PixDataset_resolution_for_SD)])

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'

        # Vicuna conversation system for editing
        self.editing_template = editing_template
        self.editing_max_length = editing_max_length
        self.mm_projection_length = mm_projection_length

    def __len__(self,):
        return len(self.InstructPix2PixDataset_path)

    def __getitem__(self, index):
        # Loading Path...
        InstructPix2PixDataset_sample = self.InstructPix2PixDataset_path[index]
        # {'original_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E4C0>, 'edited_image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F3879D3E460>, 'edit_prompt': 'make the leaves yellow'}

        # convert into torch style
        instructpix2pix_original_img = InstructPix2PixDataset_sample['original_image']
        instructpix2pix_edited_img = InstructPix2PixDataset_sample['edited_image']
        instructpix2pix_original_img = Image.open(io.BytesIO(instructpix2pix_original_img['bytes'])).convert('RGB')
        instructpix2pix_edited_img = Image.open(io.BytesIO(instructpix2pix_edited_img['bytes'])).convert('RGB')

        # convert into numpy array first, then to torch tensor
        # 1. Original Image for ViT input
        instructpix2pix_original_img_1 = instructpix2pix_original_img
        instructpix2pix_original_img_1 = instructpix2pix_original_img_1.resize((self.InstructPix2PixDataset_resolution_ViT, self.InstructPix2PixDataset_resolution_ViT),
                                                                               resample=Image.Resampling.BICUBIC)
        instructpix2pix_original_img_1 = self.CLIPImageProcessor.preprocess(instructpix2pix_original_img_1, return_tensors='pt')['pixel_values']
        instructpix2pix_original_img_1 = instructpix2pix_original_img_1[0]

        # 2. Original Image & 3. Edited Image for SD input
        instructpix2pix_original_img_2 = convert_to_np(instructpix2pix_original_img, self.InstructPix2PixDataset_resolution_for_SD)
        instructpix2pix_edited_img = convert_to_np(instructpix2pix_edited_img, self.InstructPix2PixDataset_resolution_for_SD)
        instructpix2pix_SD_input = np.concatenate([instructpix2pix_original_img_2, instructpix2pix_edited_img])
        instructpix2pix_SD_input = torch.tensor(instructpix2pix_SD_input)
        instructpix2pix_SD_input = 2 * (instructpix2pix_SD_input / 255) - 1

        # transformation for SD
        instructpix2pix_SD_input = self.SD_transform(instructpix2pix_SD_input)
        instructpix2pix_original_img_2, instructpix2pix_edited_img = instructpix2pix_SD_input.chunk(2)

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
        edited_prompt = InstructPix2PixDataset_sample['edit_prompt']
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
        instructpix2pix_input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        # ge(a, b) is a >= b
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # check exception data -> pad_token
        if input_ids[-1] != self.llm_tokenizer.pad_token_id:
            print('Exception data sample:', InstructPix2PixDataset_sample['edit_prompt'])
            edited_prompt = ""
            DEFAULT_IM_START_TOKEN = '<im_start>'
            DEFAULT_IM_END_TOKEN = '<im_end>'
            edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
            conversation_template = random.choice(conversation_templates)
            conv.messages = []
            conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
            conv.append_message(roles["GPT"], conversation_template["GPT"].replace(' [img].', append_str))
            conversation = conv.get_prompt()
            conversation = conversation.replace("\n", "")
            # 1.
            input_ids = self.llm_tokenizer(
                conversation,
                return_tensors="pt",
                padding="max_length",
                max_length=input_ids_max_len,
                truncation=True,
            ).input_ids[0]
            # 2.
            generated_caption_targets = input_ids.clone()
            sep = conv.sep + conv.roles[1] + ": "
            generated_caption_targets[:1] = IGNORE_TOKEN_ID
            total_padding_len = int(generated_caption_targets.ne(self.llm_tokenizer.pad_token_id).sum())
            parts = conversation.split(sep)
            parts[0] += sep
            instruction_len = len(
                self.llm_tokenizer(
                    parts[0],
                    max_length=input_ids_max_len,
                    truncation=True,
                ).input_ids) - 2
            generated_caption_targets[1:(1 + instruction_len)] = IGNORE_TOKEN_ID
            generated_caption_targets[total_padding_len:] = IGNORE_TOKEN_ID
            # 3.
            instructpix2pix_input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)
            # 4.
            generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)
            # 5.
            is_editing_task = torch.zeros(1)

        # InstructPix2Pix dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['make the leaves yellow']
        return {'original_img': instructpix2pix_original_img_1,
                'original_img_for_vae': instructpix2pix_original_img_2,
                'edited_img': instructpix2pix_edited_img,
                'input_ids': input_ids,
                'input_attention_mask': instructpix2pix_input_ids_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}


# MagicBrush dataset
class MagicBrush_Dataset(Dataset):
    '''
    according to MagicBrush, the dataset can be used to train models to follow edit instructions.
    Edit instructions are available in the 'instruction'. 'source_img' can be used with the 'instruction' and 'target_img' denotes the image after applying the 'instruction' on the 'source_img'.
    "source_img" + "target_img" + "instruction"
    Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
    '''
    def __init__(self,
                 MagicBrushDataset_path,
                 MagicBrushDataset_resolution_ViT,
                 MagicBrushDataset_resolution_for_SD,
                 CLIPImageProcessor,
                 mm_projection_length,
                 editing_template,
                 editing_max_length,
                 llm_tokenizer=None
                 ):

        # MagicBrush Dataset path
        self.MagicBrushDataset_path = load_from_disk(MagicBrushDataset_path)
        # 224, 256
        self.MagicBrushDataset_resolution_ViT = MagicBrushDataset_resolution_ViT
        self.MagicBrushDataset_resolution_for_SD = MagicBrushDataset_resolution_for_SD

        # CLIPImageProcessor
        self.CLIPImageProcessor = CLIPImageProcessor
        # SD transformation
        self.SD_transform = transforms.Compose([transforms.CenterCrop(self.MagicBrushDataset_resolution_for_SD)])

        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'

        # Vicuna conversation system for editing
        self.editing_template = editing_template
        self.editing_max_length = editing_max_length
        self.mm_projection_length = mm_projection_length

    def __len__(self,):
        return len(self.MagicBrushDataset_path)

    def __getitem__(self, index):
        # Loading Path...
        MagicBrushDataset_sample = self.MagicBrushDataset_path[index]
        # {'source_img': <PIL.Image.Image image mode=RGB size=500x500 at 0x7F327BE01100>, 'target_img': <PIL.Image.Image image mode=RGB size=1024x1024 at 0x7F327BE010D0>, 'instruction': 'let the asparagus be replaced with sausages'}

        # convert into torch style
        MagicBrushDataset_source_img = MagicBrushDataset_sample['source_img']
        MagicBrushDataset_target_img = MagicBrushDataset_sample['target_img']
        MagicBrushDataset_source_img = Image.open(io.BytesIO(MagicBrushDataset_source_img['bytes'])).convert('RGB')
        MagicBrushDataset_target_img = Image.open(io.BytesIO(MagicBrushDataset_target_img['bytes'])).convert('RGB')

        # convert into numpy array first, then to torch tensor
        # 1. Original Image for ViT input
        MagicBrushDataset_source_img_1 = MagicBrushDataset_source_img
        MagicBrushDataset_source_img_1 = MagicBrushDataset_source_img_1.resize((self.MagicBrushDataset_resolution_ViT, self.MagicBrushDataset_resolution_ViT),
                                                                               resample=Image.Resampling.BICUBIC)
        MagicBrushDataset_source_img_1 = self.CLIPImageProcessor.preprocess(MagicBrushDataset_source_img_1, return_tensors='pt')['pixel_values']
        MagicBrushDataset_source_img_1 = MagicBrushDataset_source_img_1[0]

        # 2. Original Image & 3. Edited Image for SD input
        MagicBrushDataset_source_img_2 = convert_to_np(MagicBrushDataset_source_img, self.MagicBrushDataset_resolution_for_SD)
        MagicBrushDataset_target_img = convert_to_np(MagicBrushDataset_target_img, self.MagicBrushDataset_resolution_for_SD)
        MagicBrushDataset_SD_input = np.concatenate([MagicBrushDataset_source_img_2, MagicBrushDataset_target_img])
        MagicBrushDataset_SD_input = torch.tensor(MagicBrushDataset_SD_input)
        MagicBrushDataset_SD_input = 2 * (MagicBrushDataset_SD_input / 255) - 1

        # transformation for SD
        MagicBrushDataset_SD_input = self.SD_transform(MagicBrushDataset_SD_input)
        MagicBrushDataset_source_img_2, MagicBrushDataset_target_img = MagicBrushDataset_SD_input.chunk(2)

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
        edited_prompt = MagicBrushDataset_sample['instruction']
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
        MagicBrushDataset_instruction_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        # ge(a, b) is a >= b
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # MagicBrushDataset dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['let the asparagus be replaced with sausages']
        return {'original_img': MagicBrushDataset_source_img_1,
                'original_img_for_vae': MagicBrushDataset_source_img_2,
                'edited_img': MagicBrushDataset_target_img,
                'input_ids': input_ids,
                'input_attention_mask': MagicBrushDataset_instruction_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}
