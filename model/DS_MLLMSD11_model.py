""" SmartEdit model for MLLM-SD """

import copy
import json
import os
import pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
# from torch.nn import LayerNorm
from typing import List, Optional, Tuple

from transformers import BertTokenizer, LlamaModel, LlamaPreTrainedModel
from transformers.models.bert import BertConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import ModelOutput

########################################################################################################################################################################
# SD_QFormer
from model.LLMSD_QFormerv01 import BertModel
# LLaVA model
from llava.model import LlavaLlamaForCausalLM
# CLIP text encoder
from transformers import CLIPTextModel, CLIPTokenizer

class AlignLLMwithSDOutput(ModelOutput):
    query_output: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

########################################################################################################################################################################
# new
class MLLMSD_model(LlamaPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    def __init__(self, config):
        super(MLLMSD_model, self).__init__(config)
        self.model = LlamaModel(config)
        # causal llm
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    # self.resize_token_embeddings(len(tokenizer))
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    # self.resize_token_embeddings(len(tokenizer))
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    #############################################################################################################################
    # conversation system num_new_tokens=33: "<img>"(system message) + " <img_0> ... <img_31>"
    def setup_tokens_for_conversation(
            self,
            tokenizer: PreTrainedTokenizer,
            num_new_tokens=8,
            tune_new_embeddings=True,
            editing_template=None,
            editing_max_length=None
    ):
        # 32000='<im_patch>', 32001='<im_start>', 32002='<im_end>'
        DEFAULT_IM_START_TOKEN = '<im_start>'
        DEFAULT_IM_END_TOKEN = '<im_end>'

        # 1. LLM has the special token "<img>" for system message to generate image -> add_tokens "<img>" -> 32000
        tokenizer.add_tokens(["<img>"], special_tokens=False)

        # 2. DEFAULT_IM_START_TOKEN and DEFAULT_IM_END_TOKEN in LLaVA -> 32001, 32002
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        tokenizer.DEFAULT_IM_START_TOKEN = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        tokenizer.DEFAULT_IM_END_TOKEN = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

        # 3. LLM contains 32 tokens to summarize image and text information for conversation system -> add_tokens "<img_0>...<img_31>" -> 32003~32034
        new_token_list = [f"<img_{i}>" for i in range(num_new_tokens)]
        tokenizer.add_tokens(new_token_list, special_tokens=False)

        # 4. count new tokens and resize tokenizer
        self.config.sd_qformer_new_tokens = num_new_tokens + 1
        self.config.llava_new_tokens = 2
        self.config.num_new_tokens = self.config.sd_qformer_new_tokens + self.config.llava_new_tokens
        self.resize_token_embeddings(len(tokenizer))
        tokenizer.img_start_token_id = tokenizer.convert_tokens_to_ids("<img_0>")

        # initialize LLaMA tokenizer + templates
        self.LLM_tokenizer = tokenizer
        self.editing_template = editing_template
        self.editing_max_length = editing_max_length

        input_embeddings = self.model.get_input_embeddings().weight.data
        output_embeddings = self.lm_head.weight.data
        self.original_LLM_word_embedding_0 = input_embeddings[0]
        self.original_LLM_language_model_head_0 = output_embeddings[0]

        input_embeddings_avg = input_embeddings[:-self.config.num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-self.config.num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-self.config.num_new_tokens:] = input_embeddings_avg
        output_embeddings[-self.config.num_new_tokens:] = output_embeddings_avg

        if tune_new_embeddings:
            self.origin_inp_embedding = input_embeddings[:-self.config.num_new_tokens].clone()
            self.origin_out_embedding = output_embeddings[:-self.config.num_new_tokens].clone()

    ####################################################################################
    # 0. initialize CLIP text encoder
    def init_CLIP_text_encoder(self, CLIP_path):
        CLIP_tokenizer = CLIPTokenizer.from_pretrained(CLIP_path)
        CLIP_text_encoder = CLIPTextModel.from_pretrained(CLIP_path)
        null_text_prompt = ""
        null_text_prompt = CLIP_tokenizer(null_text_prompt,
                                          max_length=CLIP_tokenizer.model_max_length,
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors="pt")
        null_text_prompt_ids = null_text_prompt.input_ids
        with torch.no_grad():
            null_text_embeddings_ = CLIP_text_encoder(input_ids=null_text_prompt_ids)
            self.null_text_embeddings = null_text_embeddings_.last_hidden_state
            self.null_text_embeddings = self.null_text_embeddings.to("cuda")
            # none-text embeddings: [1, 77, 768]

    #############################################################################################################################
    # 1. initialize Stable Diffusion
    def init_sd_vae_unet(self, model_name_or_path="runwayml/stable-diffusion-v1-5"):
        self.unet = UNet2DConditionModel.from_pretrained(model_name_or_path, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder="vae")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")

    #############################################################################################################################
    # 2. initialize LLaVA
    def init_visual_features_extractor(self, LLaVA_model_path, sd_qformer_version="v1.1-7b"):
        # Load LLaVA-LLaMA-v1.1 checkpoint... -> llava-1.1 vocab_size=32003
        if sd_qformer_version == "v1.1-7b" or "v1.1-13b":
            LLaVA_model = LlavaLlamaForCausalLM.from_pretrained(LLaVA_model_path)
            # 2.1. Load vision tower -> CLIPVisionTower
            self.vision_tower = LLaVA_model.get_vision_tower()
            self.vision_tower.load_model()
            # 2.2. CLIP image processor
            self.image_processor = self.vision_tower.image_processor
            # 2.3. connection from LLaVA
            self.mm_projector = nn.Linear(self.vision_tower.hidden_size, self.config.hidden_size)

            del LLaVA_model.model.mm_projector
            del LLaVA_model.model.vision_tower
            # sum([p.nelement() for p in LLaVA_model.parameters()])
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
            self.original_word_embedding_value = original_word_embedding.weight.data
            self.original_lm_head_value = original_lm_head.weight.data
            # [32000, 4096], [32000, 4096]

    #############################################################################################################################
    # 3. initialize SD q-former before Stable Diffusion
    def init_sd_qformer(
        self,
        num_query_token=77,
        num_hidden_layers=6,
        cross_attention_freq=2
    ):
        # q-former BERT config for LLM to SD
        qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        qformer_config.encoder_width = self.config.hidden_size
        # insert cross-attention layer every other block
        qformer_config.add_cross_attention = True
        qformer_config.cross_attention_freq = cross_attention_freq
        qformer_config.query_length = num_query_token
        qformer_config.num_hidden_layers = num_hidden_layers
        # self.config.sd_qformer_config = qformer_config

        # q-former model for LLM to SD
        self.sd_qformer = BertModel(config=qformer_config)
        self.sd_qformer.embeddings.word_embeddings = None
        self.sd_qformer.embeddings.position_embeddings = None
        for layer in self.sd_qformer.encoder.layer:
            layer.output = None
            layer.intermediate = None
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, qformer_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
        self.sd_query_tokens = query_tokens

    #############################################################################################################################
    # 4. load pretrained checkpoint
    def load_pretrain_MLLM_alignment(self, SD_QFormer_conversation_33tokens, LLaVA_00002):
        weights = torch.load(SD_QFormer_conversation_33tokens, map_location="cpu")
        LLaVA_00002_weights = torch.load(LLaVA_00002, map_location="cpu")
        print('mm_projector weight:', weights['mm_projector.weight'] == LLaVA_00002_weights['model.mm_projector.weight'])
        print('mm_projector bias:', weights['mm_projector.bias'] == LLaVA_00002_weights['model.mm_projector.bias'])

        # 1. vec2word: Linear(in_features=4096, out_features=32035, bias=False)
        LLaMA_lm_haed = weights.pop('lm_head.weight')
        LLaMA_lm_haed = LLaMA_lm_haed[-self.config.num_new_tokens:]
        self.lm_head.weight.data[-self.config.num_new_tokens:] = LLaMA_lm_haed
        original_LLaMA_lm_head = self.original_lm_head_value
        self.lm_head.weight.data[:-self.config.num_new_tokens] = original_LLaMA_lm_head
        print('Matching language model head:', self.lm_head.weight.data[0] == self.original_LLM_language_model_head_0)

        # 2. word2vec: Embedding(32035, 4096)
        LLaMA_word2vec = weights.pop('model.embed_tokens.weight')
        LLaMA_word2vec = LLaMA_word2vec[-self.config.num_new_tokens:]
        self.model.embed_tokens.weight.data[-self.config.num_new_tokens:] = LLaMA_word2vec
        original_LLaMA_embed_tokens = self.origin_inp_embedding
        self.model.embed_tokens.weight.data[:-self.config.num_new_tokens] = original_LLaMA_embed_tokens
        print('Matching word embedding:', self.model.embed_tokens.weight.data[0] == self.original_LLM_word_embedding_0)

        # 3. mm_projector
        mm_projector_param = {'weight': weights.pop('mm_projector.weight'), 'bias': weights.pop('mm_projector.bias')}
        self.mm_projector.load_state_dict(mm_projector_param, strict=True)

        # 4. SD_Query and SD_Qformer -> remove 'sd_qformer.'
        self.sd_query_tokens.data = weights.pop('sd_query_tokens').float()
        self.sd_qformer.load_state_dict({k[len('sd_qformer.'):]: v for k, v in weights.items()})
        print('Loading embeddings for Qformer checkpoint:', self.sd_qformer.load_state_dict({k[len('sd_qformer.'):]: v for k, v in weights.items()}, strict=True))

    # return embed_tokens -> requires_grad
    def get_model(self):
        return self.model

    ########################################################################################################################################################################
    # inference 1.with pretrained LLaMA+LoRA
    def load_pretrained_LLaMA_for_inference(self, pretrained_LLaMA):
        LLaMA_weights = torch.load(pretrained_LLaMA, map_location="cpu")
        # 1.
        LLM_ckpt_new = {}
        for k, v in LLaMA_weights.items():
            if 'model.base_model.model.' in k:
                LLM_ckpt_new[k[len('model.'):]] = v
                # remove 'model.'
        # LLM_ckpt_new['base_model.model.embed_tokens.weight']
        self.model.load_state_dict(LLM_ckpt_new, strict=True)
        print('Loading LLaMA with LoRA checkpoint:', self.model.load_state_dict(LLM_ckpt_new, strict=True))
        # 2.
        original_LLaMA_embed_tokens = self.original_word_embedding_value
        self.model.embed_tokens.weight.data[:-self.config.num_new_tokens] = original_LLaMA_embed_tokens
        print('Matching word embedding:', self.model.embed_tokens.weight.data[0] == self.original_LLM_word_embedding_0)

    ########################################################################################################################################################################
    # inference 2.with pretrained parts
    def load_pretrained_for_inference(self, pretrain_model, LLaVA_00002_weights):
        weights = torch.load(pretrain_model, map_location="cpu")
        LLaVA_00002_weights = torch.load(LLaVA_00002_weights, map_location="cpu")
        print('mm_projector weight:', weights['mm_projector.weight'] == LLaVA_00002_weights['model.mm_projector.weight'])
        print('mm_projector bias:', weights['mm_projector.bias'] == LLaVA_00002_weights['model.mm_projector.bias'])

        # 1. vec2word
        LLaMA_lm_haed = weights.pop('lm_head.weight')
        LLaMA_lm_haed = LLaMA_lm_haed[-self.config.num_new_tokens:]
        self.lm_head.weight.data[-self.config.num_new_tokens:] = LLaMA_lm_haed
        original_LLaMA_lm_head = self.original_lm_head_value
        self.lm_head.weight.data[:-self.config.num_new_tokens] = original_LLaMA_lm_head
        print('Matching language model head:', self.lm_head.weight.data[0] == self.original_LLM_language_model_head_0)

        # 2. mm_projector
        mm_projector_param = {'weight': weights.pop('mm_projector.weight'), 'bias': weights.pop('mm_projector.bias')}
        self.mm_projector.load_state_dict(mm_projector_param, strict=True)

        # remove embed_tokens
        for k, v in weights.items():
            if 'model.base_model.model.embed_tokens.weight' in k:
                weights.pop('model.base_model.model.embed_tokens.weight')
                break

        # 3. SD_Query and SD_Qformer
        self.sd_query_tokens.data = weights.pop('sd_query_tokens').float()
        self.sd_qformer.load_state_dict({k[len('sd_qformer.'):]: v for k, v in weights.items()})
        print('Loading embeddings for Qformer checkpoint:', self.sd_qformer.load_state_dict({k[len('sd_qformer.'):]: v for k, v in weights.items()}, strict=True))

    ########################################################################################################################################################################
    # inference 3.with LLM
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

        return AlignLLMwithSDOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    ########################################################################################################################################################################
    # inference 4.with sd_qformer
    def inference_sd_qformer(
        self,
        hidden_states: torch.LongTensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        # transform hidden state from LLM-space to CLIP-space, using BERT-like QFormer
        query_tokens = self.sd_query_tokens.expand(hidden_states.shape[0], -1, -1)
        query_output = self.sd_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state
        return query_output

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    @staticmethod
    def numpy_to_pil(images):
        """ Convert a numpy image or a batch of images to a PIL image. """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    ########################################################################################################################################################################
    # initialize original embeddings
    def filling_with_origin_embeddings(self):
        self.model.embed_tokens.weight.data[:-self.config.num_new_tokens] = self.original_word_embedding_value
        self.lm_head.weight.data[:-self.config.num_new_tokens] = self.original_lm_head_value

    ########################################################################################################################################################################
    # code from llava/model/llava.py
    def forward(
            self,
            ####################################################################################
            original_img: torch.FloatTensor = None,
            original_img_for_vae: torch.FloatTensor = None,
            edited_img: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            input_attention_mask: torch.Tensor = None,
            generated_caption_targets: torch.LongTensor = None,
            generated_caption_encoder_attention_mask: torch.Tensor = None,
            is_editing_task: torch.FloatTensor = None,
            ####################################################################################
            # input_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            images: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        self.filling_with_origin_embeddings()

        # 1. get LLM embeddings
        batch_size = original_img.shape[0]
        IGNORE_TOKEN_ID = generated_caption_targets[0][0].item()

        # 2. LLaVA extracts image features
        CLIP_image_features_llm_input = self.vision_tower(original_img)
        CLIP_image_features_llm_input = self.mm_projector(CLIP_image_features_llm_input)

        # 3. Prepare new (input embeddings + attention masks + generated caption targets + encoder attention masks for SD-QFormer) for LLM
        # preparing new generation targets and new attention masks for SD-QFormer
        CLIP_llm_atts = torch.ones(CLIP_image_features_llm_input.size()[:-1], dtype=torch.long, device=CLIP_image_features_llm_input.device)
        CLIP_llm_target_masks = torch.ones_like(CLIP_llm_atts, dtype=torch.long, device=CLIP_image_features_llm_input.device)
        CLIP_llm_target_masks[:, :] = IGNORE_TOKEN_ID
        insert_SDLLM_attention_masks = torch.ones_like(CLIP_llm_atts, dtype=torch.long, device=CLIP_image_features_llm_input.device)
        insert_SDLLM_attention_masks[:, :] = False

        # 4. merge new embeddings for LLM
        LLM_img_start_token_id = self.LLM_tokenizer.img_start_token_id
        LLM_new_inputs_embeds = []
        LLM_new_attention_masks = []
        LLM_new_targets = []
        SDLLM_attention_masks = []
        for sample in range(batch_size):
            """ find the first "<img_0>=32003" """
            LLM_img_start_token_id_pos = (torch.where(input_ids[sample] == LLM_img_start_token_id)[0])[0].item()
            if int(is_editing_task[sample].item()) == 1:
                assert torch.where(input_ids[sample] == LLM_img_start_token_id)[0].shape[0] == 2

            # 4.1. input_ids
            original_input_ids = input_ids[sample]
            new_input_ids = torch.cat([original_input_ids[:LLM_img_start_token_id_pos],
                                       original_input_ids[(LLM_img_start_token_id_pos + 1):],
                                       torch.tensor([self.LLM_tokenizer.pad_token_id], device=input_ids.device)], dim=0)
            LLM_inputs_embeds = self.get_input_embeddings()(new_input_ids)
            # 4.2. attention masks
            original_LLM_attention_mask = input_attention_mask[sample]
            new_LLM_attention_mask = torch.cat([original_LLM_attention_mask[:LLM_img_start_token_id_pos],
                                                original_LLM_attention_mask[(LLM_img_start_token_id_pos + 1):],
                                                torch.tensor([False], device=input_ids.device)], dim=0)
            # 4.3. generated caption targets
            original_generated_caption_targets = generated_caption_targets[sample]
            new_generated_caption_target = torch.cat([original_generated_caption_targets[:LLM_img_start_token_id_pos],
                                                      original_generated_caption_targets[(LLM_img_start_token_id_pos + 1):],
                                                      torch.tensor([IGNORE_TOKEN_ID], device=input_ids.device)], dim=0)
            # 4.4. encoder attention masks for SD-QFormer
            original_generated_caption_encoder_attention_mask = generated_caption_encoder_attention_mask[sample]
            new_generated_caption_encoder_attention_mask = torch.cat([original_generated_caption_encoder_attention_mask[:LLM_img_start_token_id_pos],
                                                                      original_generated_caption_encoder_attention_mask[(LLM_img_start_token_id_pos + 1):],
                                                                      torch.tensor([False], device=input_ids.device)], dim=0)

            # 4.1.1. new input embeddings for LLM
            LLM_embedding_BeforeStart = LLM_inputs_embeds[:LLM_img_start_token_id_pos]
            insert_SPE = CLIP_image_features_llm_input[sample]
            LLM_embedding_AfterStart = LLM_inputs_embeds[LLM_img_start_token_id_pos:]
            LLM_new_inputs_embed = torch.cat([LLM_embedding_BeforeStart.unsqueeze(0), insert_SPE.unsqueeze(0), LLM_embedding_AfterStart.unsqueeze(0)], dim=1)
            LLM_new_inputs_embeds.append(LLM_new_inputs_embed)

            # 4.2.1. new attention masks for LLM
            LLM_attention_mask = new_LLM_attention_mask
            insert_SPE_attention_mask = CLIP_llm_atts[sample]
            LLM_new_attention_mask = torch.cat([insert_SPE_attention_mask.unsqueeze(0), LLM_attention_mask.unsqueeze(0)], dim=1)
            LLM_new_attention_masks.append(LLM_new_attention_mask)

            # 4.3.1. new generated caption targets for LLM
            LLM_target = new_generated_caption_target
            insert_SPE_targets = CLIP_llm_target_masks[sample]
            LLM_new_target = torch.cat([insert_SPE_targets.unsqueeze(0), LLM_target.unsqueeze(0)], dim=1)
            LLM_new_targets.append(LLM_new_target)

            # 4.4.1. new attention masks for SD-QFormer
            original_SDLLM_attention_mask = new_generated_caption_encoder_attention_mask
            insert_SPE_targets_SDLLM_attention_mask = insert_SDLLM_attention_masks[sample]
            SDLLM_attention_mask = torch.cat([insert_SPE_targets_SDLLM_attention_mask.unsqueeze(0), original_SDLLM_attention_mask.unsqueeze(0)], dim=1)
            SDLLM_attention_masks.append(SDLLM_attention_mask)

        LLM_new_inputs_embeds = torch.cat(LLM_new_inputs_embeds, dim=0)
        LLM_new_attention_masks = torch.cat(LLM_new_attention_masks, dim=0)
        LLM_new_targets = torch.cat(LLM_new_targets, dim=0)
        SDLLM_attention_masks = torch.cat(SDLLM_attention_masks, dim=0)
        # [bs, editing_max_length, LLM_hidden_size], [bs, editing_max_length], [bs, editing_max_length], [bs, editing_max_length]

        llm_outputs = self.model(
            attention_mask=LLM_new_attention_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=LLM_new_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = llm_outputs[0]
        shift_labels = LLM_new_targets[..., 1:].contiguous()
        # [bs, editing_max_length, LLM_hidden_size], [bs, (editing_max_length-1)]

        # 5. Next token prediction language model loss: Enable model parallelism
        hidden_states = hidden_states.to(torch.float32)
        logits = self.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()

        # Flatten the tokens
        ce_loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        LM_loss = ce_loss_fct(shift_logits, shift_labels) * self.config.llm_loss_weight

        # 6. hidden_states into SD-QFormer -> transform hidden state from LLM-space to CLIP-space, using BERT-like QFormer
        hidden_states = hidden_states[:, :self.editing_max_length, :]
        SDLLM_attention_masks = SDLLM_attention_masks[:, :self.editing_max_length]
        query_tokens = self.sd_query_tokens.expand(hidden_states.shape[0], -1, -1)
        query_output = self.sd_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=SDLLM_attention_masks,
            return_dict=True,
        ).last_hidden_state
        # [bs, CLIP_model_max_length, SD_qformer_hidden_size=CLIP_test_dim]

        ####################################################################################
        # classifier-free guidance for image and text embeddings
        random_p = torch.rand(batch_size, device=original_img.device)
        InstructPix2Pix_dropout_prob = 0.05

        # Final text conditioning
        null_text_embeddings = self.null_text_embeddings
        prompt_mask = random_p < 2 * InstructPix2Pix_dropout_prob
        prompt_mask_embeds = prompt_mask.reshape(batch_size, 1, 1)
        query_output = torch.where(prompt_mask_embeds, null_text_embeddings, query_output)

        # 7. Diffusion loss: Convert images to latent space
        edited_img = edited_img.to(torch.bfloat16)
        latents = self.vae.encode(edited_img).latent_dist.sample().detach()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz, ), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        latents = latents.to(torch.float32)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        ####################################################################################
        # CFG-2. Sample masks for the original images
        # Get the additional image embedding for conditioning -> Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_img_for_vae = original_img_for_vae.to(torch.bfloat16)
        original_image_embeds_vae = self.vae.encode(original_img_for_vae).latent_dist.mode()
        image_mask_dtype = original_img.dtype
        image_mask_CFG = 1 - ((random_p >= InstructPix2Pix_dropout_prob).to(image_mask_dtype) * (random_p < 3 * InstructPix2Pix_dropout_prob).to(image_mask_dtype))
        image_mask_CFG = image_mask_CFG.reshape(batch_size, 1, 1, 1)
        original_image_embeds_vae = image_mask_CFG * original_image_embeds_vae
        original_image_embeds_vae = original_image_embeds_vae.to(torch.float32)

        # Concatenate the original_image_embeds with the noisy_latents
        concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds_vae], dim=1)

        # Predict the noise residual
        model_pred = self.unet(concatenated_noisy_latents, timesteps, query_output).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        ####################################################################################
        # LLaVA dataset loss is always 0 -> settings for deepspeed engine
        is_editing_task = is_editing_task.view(batch_size, 1, 1, 1).expand(batch_size, model_pred.shape[1], model_pred.shape[2], model_pred.shape[3])
        model_pred = model_pred * is_editing_task
        target = target * is_editing_task
        SD_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") * self.config.diffusion_loss_weight

        # Final loss: Language model loss + Diffusion loss
        loss = LM_loss + SD_loss

        if not return_dict:
            output = (query_output, ) + llm_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return AlignLLMwithSDOutput(
            loss=loss,
            query_output=query_output,
            past_key_values=llm_outputs.past_key_values,
            hidden_states=llm_outputs.hidden_states,
            attentions=llm_outputs.attentions,
        )
