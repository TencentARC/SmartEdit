""" SmartEdit model for text alignment """

import copy
import pdb
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import CLIPTextModel, LlamaModel, LlamaPreTrainedModel
from transformers.models.bert import BertConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import ModelOutput

########################################################################################################################################################################
# from fastchat.model.QFormer import BertModel
from model.LLMSD_QFormerv01 import BertModel
from llava.model import LlavaLlamaForCausalLM

class AlignLLMwithSDCLIPOutput(ModelOutput):
    query_output: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


########################################################################################################################################################################
class AlignLLMwithSDCLIP(LlamaPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    def __init__(self, config):
        super(AlignLLMwithSDCLIP, self).__init__(config)
        self.model = LlamaModel(config)
        # causal llm
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # "<img>" -> num_new_tokens=33 (<img> + <img_0>...<img_31>)
    def setup_tokens(
        self,
        tokenizer: PreTrainedTokenizer,
        num_new_tokens=8,
        tune_new_embeddings=True,
    ):
        # llm may need this special token in system to have the ability to generate img
        # without the token, llm needed to be finetuned to generate image in the conversation
        tokenizer.add_tokens(["<img>"], special_tokens=False)
        new_token_list = [f"<img_{i}>" for i in range(num_new_tokens)]
        tokenizer.add_tokens(new_token_list, special_tokens=False)
        self.config.num_new_tokens = num_new_tokens + 1
        self.resize_token_embeddings(len(tokenizer))
        tokenizer.img_start_token_id = tokenizer.convert_tokens_to_ids("<img_0>")

        # initialize new image tokens...
        # LlamaModel -> get_input_embeddings -> word embedding
        input_embeddings = self.model.get_input_embeddings().weight.data
        output_embeddings = self.lm_head.weight.data

        input_embeddings_avg = input_embeddings[:-self.config.num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-self.config.num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-self.config.num_new_tokens:] = input_embeddings_avg
        output_embeddings[-self.config.num_new_tokens:] = output_embeddings_avg

        if tune_new_embeddings:
            self.origin_inp_embedding = input_embeddings[:-self.config.num_new_tokens].clone()
            self.origin_out_embedding = output_embeddings[:-self.config.num_new_tokens].clone()

    def init_clip_text_model(self, clip_version="openai/clip-vit-large-patch14"):
        self.clip_text_model = CLIPTextModel.from_pretrained(clip_version)
        self.config.clip_version = clip_version

    def init_qformer(
        self,
        num_query_token=77,
        num_hidden_layers=12,
        cross_attention_freq=2,
    ):
        qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        qformer_config.encoder_width = self.config.hidden_size
        # insert cross-attention layer every other block
        qformer_config.add_cross_attention = True
        qformer_config.cross_attention_freq = cross_attention_freq
        qformer_config.query_length = num_query_token
        qformer_config.num_hidden_layers = num_hidden_layers
        self.config.qformer_config = qformer_config

        self.qformer = BertModel(config=qformer_config)
        self.qformer.embeddings.word_embeddings = None
        self.qformer.embeddings.position_embeddings = None
        for layer in self.qformer.encoder.layer:
            layer.output = None
            layer.intermediate = None
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, qformer_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
        self.query_tokens = query_tokens

    ########################################################################################################################################################################
    # 1. Load LLaVA-LLaMA-v1.1-7b checkpoint... -> llava-1.1-7b vocab_size=32003
    def load_LLaVA_ckpt_v1_1_7b(self, LLaVA_model_path_v1_1_7b):
        LLaVA_model = LlavaLlamaForCausalLM.from_pretrained(LLaVA_model_path_v1_1_7b)
        del LLaVA_model.model.mm_projector
        del LLaVA_model.model.vision_tower
        # sum([p.nelement() for p in LLaVA_model.parameters()]) -> 6,738,440,192
        LLaVA_model_LLM = LLaVA_model.cpu()
        LLM_ckpt_new = {}
        for k, v in LLaVA_model_LLM.state_dict().items():
            if 'model.embed_tokens.weight' in k:
                v1 = v[:-3]
                LLM_ckpt_new[k[len('model.'):]] = v1
            elif 'lm_head.weight' in k:
                v2 = v[:-3]
                LLM_ckpt_new[k] = v2
            else:
                LLM_ckpt_new[k[len('model.'):]] = v

        lm_head_weights = LLM_ckpt_new.pop('lm_head.weight')
        self.lm_head.weight.data = lm_head_weights
        self.model.load_state_dict(LLM_ckpt_new, strict=True)
        print('Load LLaVA LLaMA checkpoint:', self.model.load_state_dict(LLM_ckpt_new, strict=True))

        # original word embeddings and language model head values
        original_word_embedding = copy.deepcopy(self.model.embed_tokens)
        original_lm_head = copy.deepcopy(self.lm_head)
        self.original_word_embeding_value = original_word_embedding.weight.data
        self.original_lm_head_value = original_lm_head.weight.data
        # [32000, 4096], [32000, 4096]

    ########################################################################################################################################################################
    # 2. Load LLaVA-LLaMA-v1.1-13b checkpoint... -> llava-1.1-13b vocab_size=32003
    def load_LLaVA_ckpt_v1_1_13b(self, LLaVA_model_path_v1_1_13b):
        LLaVA_model = LlavaLlamaForCausalLM.from_pretrained(LLaVA_model_path_v1_1_13b)
        # sum([p.nelement() for p in LLaVA_model.parameters()]) -> 13,021,143,040

        del LLaVA_model.model.mm_projector
        del LLaVA_model.model.vision_tower
        LLaVA_model_LLM = LLaVA_model.cpu()
        LLM_ckpt_new = {}
        for k, v in LLaVA_model_LLM.state_dict().items():
            if 'model.embed_tokens.weight' in k:
                v1 = v[:-3]
                LLM_ckpt_new[k[len('model.'):]] = v1
            elif 'lm_head.weight' in k:
                v2 = v[:-3]
                LLM_ckpt_new[k] = v2
            else:
                LLM_ckpt_new[k[len('model.'):]] = v

        lm_head_weights = LLM_ckpt_new.pop('lm_head.weight')
        self.lm_head.weight.data = lm_head_weights
        self.model.load_state_dict(LLM_ckpt_new, strict=True)
        print('Load LLaVA LLaMA checkpoint:', self.model.load_state_dict(LLM_ckpt_new, strict=True))

        # original word embeddings and language model head values
        original_word_embedding = copy.deepcopy(self.model.embed_tokens)
        original_lm_head = copy.deepcopy(self.lm_head)
        self.original_word_embeding_value = original_word_embedding.weight.data
        self.original_lm_head_value = original_lm_head.weight.data
        # [32000, 4096], [32000, 4096]

    ########################################################################################################################################################################
    def filling_with_origin_embeddings(self):
        self.model.embed_tokens.weight.data[:-self.config.num_new_tokens] = self.original_word_embeding_value
        self.lm_head.weight.data[:-self.config.num_new_tokens] = self.original_lm_head_value

    ########################################################################################################################################################################
    # load inference checkpoints
    def load_pretrain(self, pretrain_model):
        weights = torch.load(pretrain_model, map_location="cpu")
        # 1. sd-query
        self.query_tokens.data = weights.pop('query_tokens')
        # 2. word-embedding
        self.model.embed_tokens.weight.data[:-self.config.num_new_tokens] = self.original_word_embeding_value
        self.model.embed_tokens.weight.data[-self.config.num_new_tokens:] = weights.pop('model.embed_tokens.weight')[-self.config.num_new_tokens:]
        # 3. language model head
        self.lm_head.weight.data[:-self.config.num_new_tokens] = self.original_lm_head_value
        self.lm_head.weight.data[-self.config.num_new_tokens:] = weights.pop('lm_head.weight')[-self.config.num_new_tokens:]
        # 4. sd-qformer
        self.qformer.load_state_dict({k[len("qformer."):]: v for k, v in weights.items()})
        print('Load stage-1 pretrained model:', self.qformer.load_state_dict({k[len("qformer."):]: v for k, v in weights.items()}, strict=True))

    @torch.no_grad()
    def get_clip_embedding(self, input_ids_clip_format):
        outputs = self.clip_text_model(input_ids=input_ids_clip_format)
        return outputs.last_hidden_state

    def get_model(self):
        return self.model

    def inference_llm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits, ) + outputs
            return output

        return AlignLLMwithSDCLIPOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # inference with q-former
    def inference_qformer(
        self,
        hidden_states: torch.LongTensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        # transform hidden state from LLM-space to CLIP-space, using BERT-like QFormer
        query_tokens = self.query_tokens.expand(hidden_states.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state

        return query_output

    ########################################################################################################################################################################
    # from llava/model/llava.py
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_clip_format: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        self.filling_with_origin_embeddings()

        # get LLM hidden state for the caption
        llm_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = llm_outputs[0]
        logits = self.lm_head(hidden_states)
        # llm_outputs.keys() -> odict_keys(['last_hidden_state']) -> [bs, LLM_model_max_length=256, hidden_size=4096] -> [bs, LLM_model_max_length=256, LLM_new_vocab_size=32033]

        # transform hidden state from LLM-space to CLIP-space, using BERT-like QFormer
        query_tokens = self.query_tokens.expand(hidden_states.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state
        # [bs, CLIP_model_max_length=77, dim_text=768]

        # get clip text features
        clip_text_feat = self.get_clip_embedding(input_ids_clip_format)
        # [bs, CLIP_model_max_length=77, dim_text=768]

        # next token prediction loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # [bs, (LLM_model_max_length-1)=255, LLM_new_vocab_size=32033], [bs, (LLM_model_max_length-1)=255]

        # Flatten the tokens
        ce_loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # 1. Next token prediction language model loss: Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = ce_loss_fct(shift_logits, shift_labels) * self.config.llm_loss_weight

        # 2. Align with CLIP text encoder loss: clip distill loss
        mse_loss_fct = nn.MSELoss()
        loss += (mse_loss_fct(query_output, clip_text_feat) * self.config.align_loss_weight)

        if not return_dict:
            output = (query_output, ) + llm_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return AlignLLMwithSDCLIPOutput(
            loss=loss,
            query_output=query_output,
            past_key_values=llm_outputs.past_key_values,
            hidden_states=llm_outputs.hidden_states,
            attentions=llm_outputs.attentions,
        )
