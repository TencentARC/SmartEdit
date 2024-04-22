""" SmartEdit training """

import pdb
import os
import pathlib
import random
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Optional
from torch.utils.data import Dataset
import json

import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

########################################################################################################################################################################
from model.DS_SmartEdit_model import SmartEdit_model
from peft import LoraConfig
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import deepspeed
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

#############################################################################################################################
# save lora config
def save_llama_lora_config(llama_lora_config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # read original json
    with open('./data/adapter_config.json', 'r') as f:
        data = json.load(f)

    # modify json
    llama_lora_config_dict = vars(llama_lora_config)
    data['r'] = llama_lora_config_dict["r"]
    data['lora_alpha'] = llama_lora_config_dict["lora_alpha"]
    data['target_modules'] = llama_lora_config_dict["target_modules"]
    data['lora_dropout'] = llama_lora_config_dict["lora_dropout"]
    data['task_type'] = llama_lora_config_dict["task_type"]
    data['bias'] = llama_lora_config_dict["bias"]
    with open(os.path.join(output_dir, 'adapter_config.json'), 'w') as json_file:
        json.dump(data, json_file, indent=2)

# train_LLaMALoRA.py
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

# hugging face model for hugging face trainer
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """ Collects the state dict and dump to disk. """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# remove "module"
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).
    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

#############################################################################################################################
# inherit _save() in transformers trainer
class LLMSDTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Save the model
        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()

        # Original checkpoint save: 'mm_projector' + 'llm_proj' + ('sd_query_tokens' + 'sd_qformer') + 'lm_head'
        weight_to_save = {}
        sd_qformer_keys_to_match = ['sd_qformer', 'sd_query_tokens']
        connections_keys_to_match = ['mm_projector', 'lm_head']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in sd_qformer_keys_to_match):
                weight_to_save[k] = v
            if any(key_match in k for key_match in connections_keys_to_match):
                # lm_head.weight torch.Size([32035, 4096])
                # mm_projector.weight torch.Size([4096, 1024]) + mm_projector.bias torch.Size([4096])
                print(k, v.size())
                weight_to_save[k] = v

        # Stable Diffusion unet training... -> Unet checkpoint save
        weight_to_save_unet = {}
        unet_keys_to_match = ['unet']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in unet_keys_to_match):
                weight_to_save_unet[k] = v

        # LLM training... -> LLM checkpoint save
        weight_to_save_LLM = {}
        LLM_keys_to_match = ['model.base_model.model']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in LLM_keys_to_match):
                weight_to_save_LLM[k] = v

        # BIM module
        weight_to_save_BIM = {}
        BIM_keys_to_match = ['modulate_head', 'modulate_transformer', 'pe_layer', 'dimension_reduction_head_zeroconv', 'dimension_reduction_head_qformer_out']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in BIM_keys_to_match):
                weight_to_save_BIM[k] = v

        # len(weight_to_save.keys())=133, len(weight_to_save_unet.keys())=686, len(weight_to_save_LLM.keys())=450
        # checkpoint saving -> save_steps + training_finish
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if current_folder.startswith('checkpoint-'):
            current_step = int(current_folder[len('checkpoint-'):])
            # Original checkpoint save
            mm_projector_folder = os.path.join(parent_folder, "embeddings_qformer")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}_embeddings_qformer.bin'))
            # Unet checkpoint save
            unet_folder = os.path.join(parent_folder, "unet-%d" % current_step)
            os.makedirs(unet_folder, exist_ok=True)
            torch.save(weight_to_save_unet, os.path.join(unet_folder, 'adapter_model.bin'))
            # LLM checkpoint save
            LLM_folder = os.path.join(parent_folder, "LLM-%d" % current_step)
            os.makedirs(LLM_folder, exist_ok=True)
            torch.save(weight_to_save_LLM, os.path.join(LLM_folder, 'adapter_model.bin'))

            # BIM checkpoint save
            BIM_folder = os.path.join(parent_folder, "modulate-%d" % current_step)
            os.makedirs(BIM_folder, exist_ok=True)
            torch.save(weight_to_save_BIM, os.path.join(BIM_folder, 'adapter_model.bin'))

            # optimizer and scheduler...
            now_folder = parent_folder + '/' + current_folder
            os.makedirs(now_folder, exist_ok=True)
        else:
            # Original checkpoint save
            mm_projector_folder = os.path.join(output_dir, "embeddings_qformer")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, 'checkpoint-last_embeddings_qformer.bin'))
            # Unet checkpoint save
            unet_folder = os.path.join(output_dir, "unet-last")
            os.makedirs(unet_folder, exist_ok=True)
            torch.save(weight_to_save_unet, os.path.join(unet_folder, 'adapter_model.bin'))
            # LLM checkpoint save
            LLM_folder = os.path.join(output_dir, "LLM-last")
            os.makedirs(LLM_folder, exist_ok=True)
            torch.save(weight_to_save_LLM, os.path.join(LLM_folder, 'adapter_model.bin'))

            # modulate checkpoint save
            BIM_folder = os.path.join(output_dir, "modulate-last")
            os.makedirs(BIM_folder, exist_ok=True)
            torch.save(weight_to_save_BIM, os.path.join(BIM_folder, 'adapter_model.bin'))


#############################################################################################################################
@dataclass
class ModelArguments:
    # LLM -> Vicuna
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    # config for added token
    num_new_tokens: int = 32

    # config for sd
    sd_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"

    # config for pretrained clip text model
    clip_path: str = "openai/clip-vit-large-patch14"
    clip_max_length: int = 77

    # config for qformer that link to sd
    sd_qformer_num_layers: int = 6
    sd_qformer_cross_attention_freq: int = 2
    pretrain_sd_qformer: str = None

    # pretrained model
    sd_qformer_version: str = "v1.1-7b"
    LLaVA_00001: str = "./LLaVA-7B-v1/pytorch_model-00001-of-00002.bin"
    LLaVA_00002: str = "./LLaVA-7B-v1/pytorch_model-00002-of-00002.bin"
    LLaVA_model_path: str = "./LLaVA-7B-v1"
    pretrained_LLaMA: str = "./LLMSD_exp/stage2_MLLMSD_7b/LLM-5000/adapter_model.bin"
    pretrained_model: str = "./LLMSD_exp/stage2_MLLMSD_7b/embeddings_qformer/checkpoint-5000.bin"
    pretrained_unet: str = "./LLMSD_exp/stage2_MLLMSD_7b/unet-5000/adapter_model.bin"

#############################################################################################################################
@dataclass
class DataArguments:
    # InstructPix2Pix dataset
    InstructPix2PixDataset_path: str = "./Datasets/InstructPix2PixCLIPFiltered_HF"
    InstructPix2PixDataset_resolution_ViT: int = 224
    InstructPix2PixDataset_resolution_for_SD: int = 256

    # MagicBrush dataset
    MagicBrushDataset_path: str = "./Datasets/MagicBrush_HF"
    MagicBrushDataset_resolution_ViT: int = 224
    MagicBrushDataset_resolution_for_SD: int = 256

    # LLaVA dataset
    LLaVADataset_data_path: str = "./Datasets/LLaVA/llava_instruct_150k.json"
    LLaVADataset_image_folder: str = "./Datasets/coco/train2017"
    LLaVADataset_is_BERT: bool = True
    LLaVADataset_is_LLaMA: bool = True
    LLaVADataset_resolution_ViT: int = 224

    # Refcoco and gRefcoco dataset
    refcoco_path: str = "./Datasets/refcoco_dataset"
    refcoco_transparency: float = 0.5
    refcoco_resolution_ViT: int = 224
    refcoco_resolution_for_SD: int = 256
    grefcoco_path: str = "./Datasets/grefcoco_dataset"
    grefcoco_transparency: float = 0.5
    grefcoco_resolution_ViT: int = 224
    grefcoco_resolution_for_SD: int = 256
    coco_image_path: str = "./Datasets/coco"

    # ReasoningEditing dataset
    ReasoningEditingDataset_path: str = "./Datasets/ReasoningEditing_benchmark/gather_left_right_multiple_small_color_mirror_reason_v1.json"
    ReasoningEditingDataset_resolution_ViT: int = 224
    ReasoningEditingDataset_resolution_for_SD: int = 256

    # ReasoningSegmentation dataset
    ReasoningSegmentationDataset_json_path: str = "./Datasets/ReasonSeg/train_new"
    ReasoningSegmentationDataset_image_path: str = "./Datasets/LISA/reason_seg/ReasonSeg/train"
    ReasoningSegmentationDataset_binary_mask_path: str = "./Datasets/LISA/reason_seg/ReasonSeg/train_binary_mask"
    ReasoningSegmentationDataset_resolution_ViT: int = 224
    ReasoningSegmentationDataset_resolution_for_SD: int = 256
    ReasoningSegmentation_transparency: float = 0.5

    # COCOStuff dataset
    COCOStuff_mask_path: str = "./Datasets/cocostuff"
    COCOStuff_split: str = "train2017"
    COCOStuff_transparency: float = 0.5
    COCOStuff_empty_percentage: float = 0.2
    COCOStuff_resolution_ViT: int = 224
    COCOStuff_resolution_for_SD: int = 256

    # InstructDiffusion color and segmentation templates
    InstructDiffusion_color_template: str = os.getcwd() + '/data/LLMSD_InstructDiffusion_color.txt'
    InstructDiffusion_seg_template: str = os.getcwd() + '/data/LLMSD_InstructDiffusion_seg.txt'

    # Instruction tuning template
    editing_template: str = os.getcwd() + '/data/ConversationTemplateEditing_use.txt'

#############################################################################################################################
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    # max_length -> editing=280, LLaVA=2048
    model_max_length: int = 2048
    editing_max_length: int = 512
    mm_projection_length: int = 256

    # training settings
    llm_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0

########################################################################################################################################################################
# Merge InstructPix2Pix + MagicBrush + LLaVA + RefCOCO + GRefCOCO + COCO-stuff + reasoning segmentation + reasoning editing
class Merge_Dataset(torch.utils.data.Dataset):
    def __init__(self, InstructPix2PixDataset, MagicBrushDataset, ReasoningEditingDataset, LLaVADataset, RefcocoDataset, GRefcocoDataset, COCOStuffDataset, ReasoningSegmentationDataset):
        # initialize dataset
        self.InstructPix2PixDataset = InstructPix2PixDataset
        self.MagicBrushDataset = MagicBrushDataset
        self.ReasoningEditingDataset = ReasoningEditingDataset
        self.LLaVADataset = LLaVADataset
        self.RefcocoDataset = RefcocoDataset
        self.GRefcocoDataset = GRefcocoDataset
        self.COCOStuffDataset = COCOStuffDataset
        self.ReasoningSegmentationDataset = ReasoningSegmentationDataset
        # dataset length
        self.InstructPix2PixDataset_length = len(InstructPix2PixDataset)
        self.MagicBrushDataset_length = len(MagicBrushDataset)
        self.ReasoningEditingDataset_length = len(ReasoningEditingDataset)
        self.LLaVADataset_length = len(LLaVADataset)
        self.RefcocoDataset_length = len(RefcocoDataset)
        self.GRefcocoDataset_length = len(GRefcocoDataset)
        self.COCOStuffDataset_length = len(COCOStuffDataset)
        self.ReasoningSegmentationDataset_length = len(ReasoningSegmentationDataset)
        # choosing dataset
        self.editing_len = self.InstructPix2PixDataset_length + self.MagicBrushDataset_length + self.ReasoningEditingDataset_length
        self.segmentation_len = self.RefcocoDataset_length + self.GRefcocoDataset_length + self.COCOStuffDataset_length + self.ReasoningSegmentationDataset_length
        self.total_len = self.editing_len + self.LLaVADataset_length + self.segmentation_len

    def __getitem__(self, index):
        choose_RE = random.random()
        choose_ReasoningSeg_dataset = random.random()
        choose_llava_dataset = random.random()
        choose_editing_dataset = random.random()
        choose_MagicBrush_dataset = random.random()
        choose_COCOStuff_dataset = random.random()
        choose_Refcoco_dataset = random.random()
        # 1.
        if choose_RE < 0.15:
            ReasoningEditingDataset_data = self.ReasoningEditingDataset[random.randint(0, self.ReasoningEditingDataset_length - 1)]
            return ReasoningEditingDataset_data
        else:
            # 2.
            if choose_ReasoningSeg_dataset < 0.1:
                ReasoningSegmentationDataset_data = self.ReasoningSegmentationDataset[random.randint(0, self.ReasoningSegmentationDataset_length - 1)]
                return ReasoningSegmentationDataset_data
            else:
                # 3.
                if choose_llava_dataset < self.LLaVADataset_length / (self.total_len - self.ReasoningEditingDataset_length - self.ReasoningSegmentationDataset_length):
                    LLaVADataset_data = self.LLaVADataset[random.randint(0, self.LLaVADataset_length - 1)]
                    return LLaVADataset_data
                else:
                    # 4.
                    if choose_editing_dataset < self.editing_len / (self.editing_len + self.segmentation_len):
                        # 4.1.InstructPix2Pix:MagicBrush=1:1
                        if choose_MagicBrush_dataset < 0.5:
                            MagicBrushDataset_data = self.MagicBrushDataset[random.randint(0, self.MagicBrushDataset_length - 1)]
                            return MagicBrushDataset_data
                        else:
                            InstructPix2PixDataset_data = self.InstructPix2PixDataset[random.randint(0, self.InstructPix2PixDataset_length - 1)]
                            return InstructPix2PixDataset_data
                    else:
                        # 5.
                        if choose_COCOStuff_dataset < self.COCOStuffDataset_length / self.segmentation_len:
                            COCOStuffDataset_data = self.COCOStuffDataset[random.randint(0, self.COCOStuffDataset_length - 1)]
                            return COCOStuffDataset_data
                        else:
                            # 6.
                            if choose_Refcoco_dataset < self.RefcocoDataset_length / (self.RefcocoDataset_length + self.GRefcocoDataset_length):
                                RefcocoDataset_data = self.RefcocoDataset[random.randint(0, self.RefcocoDataset_length - 1)]
                                return RefcocoDataset_data
                            else:
                                GRefcocoDataset_data = self.GRefcocoDataset[random.randint(0, self.GRefcocoDataset_length - 1)]
                                return GRefcocoDataset_data

    def __len__(self):
        return self.total_len

########################################################################################################################################################################
from typing import Dict, Sequence
IGNORE_INDEX = -100
@dataclass
class DataCollatorForLLaVADataset(object):
    """ Collate examples for supervised fine-tuning. """

    LLM_tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # LLaVA: len(instances)=batch_size & instances[i].keys() -> dict_keys(['input_ids', 'labels', 'image'])
        original_img = [instance['original_img'] for instance in instances]
        original_img = torch.stack(original_img)
        original_img_for_vae = [instance['original_img_for_vae'] for instance in instances]
        original_img_for_vae = torch.stack(original_img_for_vae)
        edited_img = [instance['edited_img'] for instance in instances]
        edited_img = torch.stack(edited_img)
        is_editing_task = [instance['is_editing_task'] for instance in instances]
        is_editing_task = torch.stack(is_editing_task)

        # for LLaVA processing
        # 1. LLM tokenizer
        input_ids = tuple([instance['input_ids'] for instance in instances])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.LLM_tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.LLM_tokenizer.model_max_length]
        input_attention_mask = [input_id.ne(self.LLM_tokenizer.pad_token_id) for input_id in input_ids]
        input_attention_mask = torch.stack(input_attention_mask)
        input_attention_mask = input_attention_mask[:, :self.LLM_tokenizer.model_max_length]
        generated_caption_targets = tuple([instance['generated_caption_targets'] for instance in instances])
        generated_caption_targets = torch.nn.utils.rnn.pad_sequence(generated_caption_targets, batch_first=True, padding_value=IGNORE_INDEX)
        generated_caption_targets = generated_caption_targets[:, :self.LLM_tokenizer.model_max_length]
        generated_caption_encoder_attention_mask = tuple([instance['generated_caption_encoder_attention_mask'] for instance in instances])
        generated_caption_encoder_attention_mask = torch.nn.utils.rnn.pad_sequence(generated_caption_encoder_attention_mask, batch_first=True, padding_value=False)
        generated_caption_encoder_attention_mask = generated_caption_encoder_attention_mask[:, :self.LLM_tokenizer.model_max_length]

        # return from DataCollatorForLLaVADataset
        return {'original_img': original_img,
                'original_img_for_vae': original_img_for_vae,
                'edited_img': edited_img,
                'input_ids': input_ids,
                'input_attention_mask': input_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}


#############################################################################################################################
from model.DS_LoraLLaMAUnetPeftModel_new import get_peft_model
def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model_ = SmartEdit_model.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model_.config.use_cache = False
    dtype = torch.bfloat16
    # MLLMSD_model: sum([p.nelement() for p in model_.parameters()])

    # load first stage pre-training from corresponding LLaVA version
    sd_qformer_version = model_args.sd_qformer_version
    if sd_qformer_version == "v1.1-7b" or "v1.1-13b":
        LLaVA_model_path = model_args.LLaVA_model_path
        # init and freeze vit image encoder -> CLIP-ViT
        model_.init_visual_features_extractor(LLaVA_model_path=LLaVA_model_path, sd_qformer_version=sd_qformer_version)
        model_.vision_tower.requires_grad_(False)
        model_.vision_tower.to(torch.float32)
        # model_.vision_tower.eval()

    # init llm tokenizer -> LlamaTokenizer
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    # setup new llm tokens -> conversation system num_new_tokens=33: "<img>"(system message) + " <img_0> ... <img_31>" -> len(llm_tokenizer)=32033 -> original=32000
    model_.setup_tokens_for_conversation(llm_tokenizer, num_new_tokens=model_args.num_new_tokens, tune_new_embeddings=True, editing_template=data_args.editing_template, editing_max_length=training_args.editing_max_length)

    # freeze mm_projector
    for p in model_.mm_projector.parameters():
        p.requires_grad = False
    model_.mm_projector.to(torch.float32)
    # model_.vision_tower.dtype -> torch.float32
    # model_.mm_projector.weight.dtype -> torch.float32

    # LISA: lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_target_modules="q_proj,v_proj", bias="none", task_type="CAUSAL_LM"
    # LLaMA lora hyper-parameters
    lora_attention_dim_llama = 8
    lora_alpha_llama = 16
    lora_target_modules_llama = ['q_proj', 'v_proj']
    lora_dropout_llama = 0.05
    task_type_llama = "CAUSAL_LM"
    lora_bias_llama = 'none'
    llama_lora_config = LoraConfig(
        r=lora_attention_dim_llama,
        lora_alpha=lora_alpha_llama,
        target_modules=lora_target_modules_llama,
        lora_dropout=lora_dropout_llama,
        task_type=task_type_llama,
        bias=lora_bias_llama
    )
    save_llama_lora_config(llama_lora_config, training_args.output_dir)
    model_.model = get_peft_model(model_.model, llama_lora_config)
    model_.model.print_trainable_parameters()
    print('LoraLLaMA_model -> lora training')
    # trainable params: 4,194,304 || all params: 6,611,681,280 || trainable%: 0.06343778265125327

    # embed_tokens -> requires_grad
    for p in model_.get_model().embed_tokens.parameters():
        p.requires_grad = True

    # init LLM with LoRA and put on bfloat16 -> RuntimeError: Expected q_dtype == torch::kFloat16 || ((is_sm8x || is_sm90) && q_dtype == torch::kBFloat16) to be true, but got false
    model_.model.to(dtype)
    model_.model.embed_tokens.to(torch.float32)
    model_.lm_head.to(torch.float32)

    # init q-former that link SD
    model_.init_sd_qformer(
        num_query_token=model_args.clip_max_length,
        num_hidden_layers=model_args.sd_qformer_num_layers,
        cross_attention_freq=model_args.sd_qformer_cross_attention_freq
    )
    # model_.sd_qformer.encoder.layer[0].output_query.dense.weight.dtype -> torch.float32
    # model_.sd_query_tokens.dtype -> torch.float32

    # init and freeze vae -> "runwayml/stable-diffusion-v1-5"
    model_.init_sd_vae_unet(model_args.sd_model_name_or_path)
    model_.vae.requires_grad_(False)
    model_.vae.to(dtype)

    # load first stage pre-training from corresponding LLaVA version
    if sd_qformer_version == "v1.1-7b" or "v1.1-13b":
        LLaVA_00002 = model_args.LLaVA_00002
        print(LLaVA_00002)

    """ init BIM module """
    model_.init_BIM_module()
    model_.modulate_head.to(torch.float32)
    model_.modulate_transformer.to(torch.float32)
    model_.pe_layer.to(torch.float32)
    model_.dimension_reduction_head_zeroconv.to(torch.float32)
    model_.dimension_reduction_head_qformer_out.to(torch.float32)

    ####################################################################################
    """ initialize unet by InstructPix2Pix -> align with InstructPix2Pix hugging-face """
    from diffusers.models.attention_processor import AttnProcessor2_0
    in_channels = 8
    out_channels = model_.unet.conv_in.out_channels
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, model_.unet.conv_in.kernel_size, model_.unet.conv_in.stride, model_.unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(model_.unet.conv_in.weight)
        model_.unet.conv_in = new_conv_in
    model_.unet.set_attn_processor(AttnProcessor2_0())
    model_.unet.to(torch.float32)

    """ load pretrained checkpoint from MLLMSD-11 """
    pretrained_LLaMA = model_args.pretrained_LLaMA
    pretrained_model = model_args.pretrained_model
    pretrained_unet = model_args.pretrained_unet
    model_.load_pretrain_MLLMSD11(pretrained_LLaMA, pretrained_model, pretrained_unet, LLaVA_00002_weights=LLaVA_00002)
    model_.unet.requires_grad_(True)
    print('Fine-tuning total unet...')
    print(pretrained_LLaMA, pretrained_model, pretrained_unet)

    # init CLIP for null-text embeddings
    model_.init_CLIP_text_encoder(CLIP_path=model_args.clip_path)

    # setup loss weight
    model_.config.llm_loss_weight = training_args.llm_loss_weight
    model_.config.diffusion_loss_weight = training_args.diffusion_loss_weight

    """ check data_type"""
    print("1.model.vision_tower.dtype: ", model_.vision_tower.dtype)
    print("2.model.mm_projector.dtype: ", model_.mm_projector.weight.dtype)
    print("3.1.model.model.model(LLaMA).embed_tokens.dtype: ", model_.model.embed_tokens.weight.dtype)
    print("3.2.model.model.model(LLaMA).dtype: ", model_.model.layers[0].self_attn.q_proj.lora_A.default.weight.dtype, model_.model.layers[0].self_attn.q_proj.weight.dtype)
    print("3.3.model.lm_head.dtype: ", model_.lm_head.weight.data.dtype)
    print("4.1.model.sd_query_tokens.dtype: ", model_.sd_query_tokens.data.dtype)
    print("4.2.model.sd_qformer.dtype: ", model_.sd_qformer.dtype)
    print("5.1.model.vae.dtype: ", model_.vae.dtype)
    print("5.2.model.unet.dtype: ", model_.unet.dtype)
    print("model.get_input_embeddings().dtype: ", model_.get_input_embeddings().weight.data.dtype)

    params_no_grad = [n for n, p in model_.named_parameters() if not p.requires_grad]
    params_requires_grad = [n for n, p in model_.named_parameters() if p.requires_grad]
    print(params_requires_grad)
    print(sum([p.nelement() for p in model_.parameters()]))

    ####################################################################################
    ####################################################################################
    # load dataset
    from dataset.SegMLLMSD_dataset import RefCOCODataset, GrefCOCODataset, COCOStuffDataset
    from dataset.EditMLLMSD_dataset import InstructPix2Pix_Dataset, MagicBrush_Dataset
    from dataset.LLaVAMLLMSD_dataset import LLaVADataset_for_instruction_tuning
    from dataset.ReasonEditMLLMSD_dataset import ReasoningEditing_Dataset
    from dataset.ReasonSegMLLMSD_dataset import ReasoningSegmentation_Dataset

    # 1. len(InstructPix2Pix_train_Dataset)
    InstructPix2Pix_train_Dataset = InstructPix2Pix_Dataset(
        InstructPix2PixDataset_path=data_args.InstructPix2PixDataset_path,
        InstructPix2PixDataset_resolution_ViT=data_args.InstructPix2PixDataset_resolution_ViT,
        InstructPix2PixDataset_resolution_for_SD=data_args.InstructPix2PixDataset_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)
    InstructPix2Pix_train_dataloader = torch.utils.data.DataLoader(InstructPix2Pix_train_Dataset, batch_size=1, num_workers=8)

    # 2. len(MagicBrush_train_Dataset)
    MagicBrush_train_Dataset = MagicBrush_Dataset(
        MagicBrushDataset_path=data_args.MagicBrushDataset_path,
        MagicBrushDataset_resolution_ViT=data_args.MagicBrushDataset_resolution_ViT,
        MagicBrushDataset_resolution_for_SD=data_args.MagicBrushDataset_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)
    MagicBrush_train_dataloader = torch.utils.data.DataLoader(MagicBrush_train_Dataset, batch_size=1, num_workers=8)

    # 3. len(LLaVADataset)
    LLaVA_train_dataset = LLaVADataset_for_instruction_tuning(
        data_path=data_args.LLaVADataset_data_path,
        image_folder=data_args.LLaVADataset_image_folder,
        LLM_tokenizer=llm_tokenizer,
        CLIPImageProcessor=model_.image_processor,
        is_LLaMA=data_args.LLaVADataset_is_LLaMA,
        LLaVADataset_resolution_ViT=data_args.LLaVADataset_resolution_ViT)
    LLaVA_train_dataloader = torch.utils.data.DataLoader(LLaVA_train_dataset, batch_size=1, num_workers=8)

    # 4. len(Refcoco_train_dataset)
    Refcoco_train_dataset = RefCOCODataset(
        path=data_args.refcoco_path,
        path_coco=data_args.coco_image_path,
        transparency=data_args.refcoco_transparency,
        InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
        InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template,
        Refcoco_resolution_ViT=data_args.refcoco_resolution_ViT,
        Refcoco_resolution_for_SD=data_args.refcoco_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)
    Refcoco_train_dataloader = torch.utils.data.DataLoader(Refcoco_train_dataset, batch_size=1, num_workers=8)

    # 5. len(GRefcoco_train_dataset)
    GRefcoco_train_dataset = GrefCOCODataset(
        path=data_args.grefcoco_path,
        path_coco=data_args.coco_image_path,
        transparency=data_args.grefcoco_transparency,
        InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
        InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template,
        gRefcoco_resolution_ViT=data_args.grefcoco_resolution_ViT,
        gRefcoco_resolution_for_SD=data_args.grefcoco_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)
    GRefcoco_train_dataloader = torch.utils.data.DataLoader(GRefcoco_train_dataset, batch_size=1, num_workers=8)

    # 6. len(COCOStuff_train_dataset)
    COCOStuff_train_dataset = COCOStuffDataset(
        path_coco_image=data_args.coco_image_path,
        path_cocostuff_mask=data_args.COCOStuff_mask_path,
        split=data_args.COCOStuff_split,
        transparency=data_args.COCOStuff_transparency,
        InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
        InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template,
        empty_percentage=data_args.COCOStuff_empty_percentage,
        cocostuff_resolution_ViT=data_args.COCOStuff_resolution_ViT,
        cocostuff_resolution_for_SD=data_args.COCOStuff_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)
    COCOStuff_train_dataloader = torch.utils.data.DataLoader(COCOStuff_train_dataset, batch_size=1, num_workers=8)

    # 7. len(ReasoningEditing_train_Dataset)
    ReasoningEditing_train_Dataset = ReasoningEditing_Dataset(
        ReasoningEditingDataset_path=data_args.ReasoningEditingDataset_path,
        ReasoningEditingDataset_resolution_ViT=data_args.ReasoningEditingDataset_resolution_ViT,
        ReasoningEditingDataset_resolution_for_SD=data_args.ReasoningEditingDataset_resolution_for_SD,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer)
    ReasoningEditing_train_dataloader = torch.utils.data.DataLoader(ReasoningEditing_train_Dataset, batch_size=1, num_workers=8)
    print('Checking Reasoning-Editing-Dataset train dataset...')
    print('Reasoning-Editing-Dataset length:', len(ReasoningEditing_train_Dataset), 'Reasoning-Editing-Dataset json:', data_args.ReasoningEditingDataset_path)

    # 8. len(ReasoningSegmentation_train_Dataset)
    ReasoningSegmentation_train_Dataset = ReasoningSegmentation_Dataset(
        ReasoningSegmentationDataset_json_path=data_args.ReasoningSegmentationDataset_json_path,
        ReasoningSegmentationDataset_image_path=data_args.ReasoningSegmentationDataset_image_path,
        ReasoningSegmentationDataset_binary_mask_path=data_args.ReasoningSegmentationDataset_binary_mask_path,
        ReasoningSegmentationDataset_resolution_ViT=data_args.ReasoningSegmentationDataset_resolution_ViT,
        ReasoningSegmentationDataset_resolution_for_SD=data_args.ReasoningSegmentationDataset_resolution_for_SD,
        transparency=data_args.ReasoningSegmentation_transparency,
        CLIPImageProcessor=model_.image_processor,
        mm_projection_length=training_args.mm_projection_length,
        editing_template=data_args.editing_template,
        editing_max_length=training_args.editing_max_length,
        llm_tokenizer=llm_tokenizer,
        InstructDiffusion_color_template=data_args.InstructDiffusion_color_template,
        InstructDiffusion_seg_template=data_args.InstructDiffusion_seg_template)
    ReasoningSegmentation_train_dataloader = torch.utils.data.DataLoader(ReasoningSegmentation_train_Dataset, batch_size=1, num_workers=8)

    # 9. len(merged_train_dataset)
    merged_train_dataset = Merge_Dataset(InstructPix2PixDataset=InstructPix2Pix_train_Dataset,
                                         MagicBrushDataset=MagicBrush_train_Dataset,
                                         ReasoningEditingDataset=ReasoningEditing_train_Dataset,
                                         LLaVADataset=LLaVA_train_dataset,
                                         RefcocoDataset=Refcoco_train_dataset,
                                         GRefcocoDataset=GRefcoco_train_dataset,
                                         COCOStuffDataset=COCOStuff_train_dataset,
                                         ReasoningSegmentationDataset=ReasoningSegmentation_train_Dataset)
    merged_train_dataloader = torch.utils.data.DataLoader(merged_train_dataset, batch_size=1, num_workers=8)
    print(merged_train_dataset)

    # 10. DataCollatorForLLaVADataset
    data_collator_train_dataset = DataCollatorForLLaVADataset(LLM_tokenizer=llm_tokenizer)

    # add data_collator
    data_module_ = dict(train_dataset=merged_train_dataset, eval_dataset=None, data_collator=data_collator_train_dataset)
    trainer = LLMSDTrainer(model=model_, tokenizer=llm_tokenizer, args=training_args, **data_module_)

    # trainer for pretrained checkpoint or not
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    # save trainer_state.json
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # train_LLaMALoRA.py
    state_dict = get_peft_state_maybe_zero_3(
        model_.named_parameters(), lora_bias_llama)
    if training_args.local_rank == 0:
        model_.save_pretrained(training_args.output_dir, state_dict=state_dict)

####################################################################################
if __name__ == "__main__":
    from train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    train()
