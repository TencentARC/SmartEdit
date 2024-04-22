""" SmartEdit training for text alignment """

import copy
import csv
import json
import pdb

import numpy as np
import os
import pathlib
import random
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence

#############################################################################################################################
import transformers
from conversation_v01 import SeparatorStyle, get_conv_template
from transformers import CLIPTokenizer, Trainer
from transformers.trainer_pt_utils import LabelSmoother
from model.LLMSD_modelv01_conv import AlignLLMwithSDCLIP
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


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


class AlignLLMwithSDCLIPTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Save the model
        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()

        weight_to_save = {}
        keys_to_match = ['qformer', 'query_tokens']
        embedding_keys_to_match = ['embed_tokens', 'lm_head']
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in keys_to_match):
                weight_to_save[k] = v
            if any(key_match in k for key_match in embedding_keys_to_match):
                print(v.size())
                weight_to_save[k] = v.data[-self.model.config.num_new_tokens:, :].clone()

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if current_folder.startswith('checkpoint-'):
            mm_projector_folder = os.path.join(parent_folder, "embeddings_qformer")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f'embeddings_qformer.bin'))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    # config for pretrained clip text model
    clip_version: str = "openai/clip-vit-large-patch14"
    clip_hidden_size: int = 768
    clip_max_length: int = 77

    # config for qformer
    qformer_num_layers: int = 6
    qformer_cross_attention_freq: int = 2

    # config for added token
    num_new_tokens: int = 8

    # LLaVA settings -> v1.1-7b, v1.1-13b
    LLaVA_version: str = "v1.1-7b"
    LLaVA_model_v1_1_7b_path: str = "./LLaVA-7B-v1"
    LLaVA_model_v1_1_13b_path: str = "./LLaVA-13B-v1"
    pretrain_ckpt: str = ""


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    template_data_path: str = field(default=None, metadata={"help": "Path to the template data."})
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=256,  # for cc3m caption, the max caption token len is 125
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    llm_loss_weight: float = 1.0
    align_loss_weight: float = 1.0
    is_load_pretrain: bool = False


local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(sources, llm_tokenizer: transformers.PreTrainedTokenizer,
               clip_tokenizer: transformers.PreTrainedTokenizer, templates) -> Dict:
    conv = get_conv_template("vicuna_v1.2")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # <img_i> tokens
    num_new_tokens = len(llm_tokenizer) - llm_tokenizer.vocab_size
    append_str = ""
    for i in range(num_new_tokens - 1):
        append_str += f" <img_{i}>"

    max_try = 4
    source = sources[0]
    while True:
        template = random.choice(templates)
        conv.messages = []
        conv.append_message(roles["human"], template["human"].replace('[cap]', f'"{source}"'))
        conv.append_message(roles["gpt"], template["gpt"].replace(' [img].', append_str))

        conversation = conv.get_prompt()

        # Tokenize conversations
        input_ids = llm_tokenizer(
            conversation,
            return_tensors="pt",
            padding="max_length",
            max_length=llm_tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]

        if input_ids[-1] == llm_tokenizer.pad_token_id:
            break
        if input_ids[-1] == len(llm_tokenizer) - 1:
            break

        # half the caption if it is too long
        source = source[:len(source) // 2]
        max_try -= 1
        if max_try == 0:
            rank0_print(f'reach max try, the length for current caption is {len(source)}')
            break

    target = input_ids.clone()

    sep = conv.sep + conv.roles[1] + ": "
    target[:1] = IGNORE_TOKEN_ID
    total_len = int(target.ne(llm_tokenizer.pad_token_id).sum())

    parts = conversation.split(sep)
    parts[0] += sep
    instruction_len = len(
        llm_tokenizer(
            parts[0],
            max_length=llm_tokenizer.model_max_length,
            truncation=True,
        ).input_ids) - 2
    target[1:1 + instruction_len] = IGNORE_TOKEN_ID
    target[total_len:] = IGNORE_TOKEN_ID

    # note: 目前假定所有img token都会在序列中出现，而CC3M最大长度不超过128，所以tokenier最大长度一定要设为>128+8，否则就需要写边界处理代码
    input_ids_clip_format = clip_tokenizer(
        source,
        return_tensors="pt",
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]

    return dict(
        input_ids=input_ids,
        labels=target,
        input_ids_clip_format=input_ids_clip_format,
        attention_mask=input_ids.ne(llm_tokenizer.pad_token_id),
        encoder_attention_mask=input_ids.ge(llm_tokenizer.img_start_token_id),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        template_data_path: str,
        llm_tokenizer: transformers.PreTrainedTokenizer,
        clip_tokenizer: transformers.PreTrainedTokenizer,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.llm_tokenizer = llm_tokenizer
        self.clip_tokenizer = clip_tokenizer

        rank0_print(f"Loading templates from {template_data_path}...")
        templates = []
        with open(template_data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('human: '):
                    d = dict()
                    d['human'] = line[len("human: "):]
                    templates.append(d)
                elif line.startswith('gpt: '):
                    templates[-1]['gpt'] = line[len("gpt: "):]

        rank0_print("Loading data...")

        self.data_path = data_path
        tsv_dict = {}

        with open(self.data_path, "r") as f:
            tsv_reader = csv.reader(f, delimiter="\t")

            for index, line in enumerate(tsv_reader):
                tsv_dict[index] = line

        self.tsv_dict = tsv_dict

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.cached_data_dict = {}
        self.templates = templates

    def __len__(self):
        return len(self.tsv_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.tsv_dict[i][-1]], self.llm_tokenizer, self.clip_tokenizer, self.templates)
        ret = dict(
            input_ids=ret["input_ids"],
            labels=ret["labels"],
            input_ids_clip_format=ret["input_ids_clip_format"],
            attention_mask=ret["attention_mask"],
            encoder_attention_mask=ret["encoder_attention_mask"],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    llm_tokenizer: transformers.PreTrainedTokenizer,
    clip_tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    train_dataset = dataset_cls(data_args.data_path, data_args.template_data_path, llm_tokenizer, clip_tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None)


def train():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model = AlignLLMwithSDCLIP.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    dtype = torch.bfloat16
    # sum([p.nelement() for p in model.parameters()]) -> vicuna-7b-v1-1

    ####################################################################################
    # load LLaVA version
    LLaVA_version = model_args.LLaVA_version
    if LLaVA_version == "v1.1-7b":
        model.load_LLaVA_ckpt_v1_1(LLaVA_model_path_v1_1=model_args.LLaVA_model_v1_1_7b_path)
    elif LLaVA_version == "v1.1-13b":
        model.load_LLaVA_ckpt_v1_1_13b(LLaVA_model_path_v1_1_13b=model_args.LLaVA_model_v1_1_13b_path)
    print('LLaVA version:', LLaVA_version)

    # freeze LLM
    model.model.requires_grad_(False)
    model.model.embed_tokens.to(torch.float32)

    # init llm tokenizer
    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    # setup new llm tokens
    model.setup_tokens(llm_tokenizer, num_new_tokens=model_args.num_new_tokens, tune_new_embeddings=True)

    # init and freeze clip
    model.init_clip_text_model(model_args.clip_version)
    for name, param in model.clip_text_model.named_parameters():
        param.requires_grad = False
    # model.clip_text_model.to(dtype)
    model.clip_text_model.eval()

    # init clip tokenizer
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_args.clip_version)
    clip_tokenizer.model_max_length = model_args.clip_max_length

    # init qformer
    model.init_qformer(
        num_query_token=model_args.clip_max_length,
        num_hidden_layers=model_args.qformer_num_layers,
        cross_attention_freq=model_args.qformer_cross_attention_freq)

    # load pre-trained checkpoint
    is_load_pretrain = training_args.is_load_pretrain
    if is_load_pretrain == True:
        model.load_pretrain(model_args.pretrain_ckpt)
        print(model_args.pretrain_ckpt)

    # setup loss weight
    model.config.llm_loss_weight = training_args.llm_loss_weight
    model.config.align_loss_weight = training_args.align_loss_weight

    # check require gradients parameters
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print(params_requires_grad)
    print(sum([p.nelement() for p in model.parameters()]))

    """ check data_type"""
    print("1.1.model.model.model(LLaMA).embed_tokens.dtype: ", model.model.embed_tokens.weight.dtype)
    print("1.2.model.model.model(LLaMA).dtype: ", model.model.layers[0].self_attn.q_proj.weight.dtype)
    print("1.3.model.lm_head.dtype: ", model.lm_head.weight.data.dtype)
    print("2.1.model.query_tokens.dtype: ", model.query_tokens.data.dtype)
    print("2.2.model.qformer.dtype: ", model.qformer.dtype)

    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                    len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.
                      format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
            )

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func
            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    # len(data_module['train_dataset'])=12,423,374
    data_module = make_supervised_data_module(llm_tokenizer, clip_tokenizer, data_args=data_args)
    trainer = AlignLLMwithSDCLIPTrainer(model=model, tokenizer=llm_tokenizer, args=training_args, **data_module)
    print('trainer.qformer: ', trainer.model.qformer.dtype)
    print('trainer.query_tokens: ', trainer.model.query_tokens.dtype)
    print('trainer.clip_text_model: ', trainer.model.clip_text_model.dtype)

    # check CC12M text data
    CC12M_train_dataloader = torch.utils.data.DataLoader(data_module['train_dataset'], batch_size=1, num_workers=8)
    print('Checking CC12M train dataset...')
    index = 0
    for step, batch_data in enumerate(CC12M_train_dataloader):
        # dict_keys(['input_ids', 'labels', 'input_ids_clip_format', 'attention_mask', 'encoder_attention_mask'])
        print(batch_data['input_ids'], batch_data['input_ids'].shape, batch_data['input_ids'].dtype)  # LongTensor=int64
        print(batch_data['labels'], batch_data['labels'].shape, batch_data['labels'].dtype)  # LongTensor=int64
        print(batch_data['input_ids_clip_format'], batch_data['input_ids_clip_format'].shape, batch_data['input_ids_clip_format'].dtype)  # LongTensor=int64
        print(batch_data['attention_mask'], batch_data['attention_mask'].shape, batch_data['attention_mask'].dtype)  # LongTensor=int64
        print(batch_data['encoder_attention_mask'], batch_data['encoder_attention_mask'].shape, batch_data['encoder_attention_mask'].dtype)  # LongTensor=int64
        index = index + 1
        if index == 1:
            break

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    from train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    replace_llama_attn_with_flash_attn()
    train()
