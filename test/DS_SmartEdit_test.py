""" SmartEdit inference """

import pdb
import argparse
import glob
import numpy as np
import os
import torch.nn as nn
import random
import torch
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DConditionModel
from torchvision.transforms.functional import InterpolationMode
import transformers
from diffusers.utils import PIL_INTERPOLATION
torch.set_grad_enabled(False)

# negative prompt for inference
DEFAULT_NEGATIVE_PROMPT = 'nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
editing_template = os.getcwd() + '/data/ConversationTemplateEditing_use.txt'

########################################################################################################################################################################
def get_prompt_embedding_LLM(img_embeddings,
                             text_prompt,
                             conversation_template,
                             LLM_tokenizer,
                             model_,
                             condition_prompt='positive'):
    # Step 2. Choose Vicuna_v1.3 system message
    conv = get_conv_template("vicuna_v1.3")
    roles = {"Human": conv.roles[0], "GPT": conv.roles[1]}

    # Step 3. Vicuna conversation system construction
    edited_prompt = text_prompt
    DEFAULT_IM_START_TOKEN = '<im_start>'
    DEFAULT_IM_END_TOKEN = '<im_end>'
    edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
    conv.messages = []
    conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
    conv.append_message(roles["GPT"], None)
    conversation = conv.get_prompt()
    conversation = conversation.replace("\n", "")

    # Step 4. Save conversation into discrete index and list
    text_prompt_input_ids = LLM_tokenizer(conversation).input_ids
    text_prompt_input_ids = torch.as_tensor(text_prompt_input_ids, device="cuda")
    output_text_ids = text_prompt_input_ids.tolist()

    # decode loop
    llm_img_token_states = []
    token_id = None
    max_new_tokens = 512
    for i in range(max_new_tokens):
        # create following tokens
        if i == 0:
            if condition_prompt == 'positive':
                images_llm_input = img_embeddings

                # Remove the first placeholder "<img_0>"
                original_input_ids = text_prompt_input_ids
                LLM_img_start_token_id = LLM_tokenizer.img_start_token_id
                LLM_img_start_token_id_pos = (torch.where(text_prompt_input_ids == LLM_img_start_token_id)[0])[0].item()
                new_input_ids = torch.cat([original_input_ids[:LLM_img_start_token_id_pos], original_input_ids[(LLM_img_start_token_id_pos + 1):]], dim=0)

                # prepare input embedding rather than input_ids
                inputs_embeds = model_.model.get_input_embeddings()(new_input_ids.unsqueeze(0))
                LLM_embedding_BeforeStart = inputs_embeds[0][:LLM_img_start_token_id_pos]
                insert_SPE = images_llm_input[0]
                LLM_embedding_AfterStart = inputs_embeds[0][LLM_img_start_token_id_pos:]
                inputs_embeds = torch.cat([LLM_embedding_BeforeStart.unsqueeze(0), insert_SPE.unsqueeze(0), LLM_embedding_AfterStart.unsqueeze(0)], dim=1)
                # [Before Padding, Insert Subject Prompt Embedding, After Padding]

                # next token prediction
                out = model_.inference_llm(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=True)
                # out.keys() -> odict_keys(['logits', 'last_hidden_state', 'past_key_values', 'hidden_states', 'attentions'])

            elif condition_prompt == 'negative':
                inputs_embeds = model_.model.get_input_embeddings()(text_prompt_input_ids.unsqueeze(0))
                out = model_.inference_llm(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
            # [1, (ViT_qformer_query_length + len(input_ids)), LLM_new_vocab_size], len(past_key_values)=32
        else:
            # iterative creation of tokena
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device="cuda")
            out = model_.inference_llm(
                input_ids=torch.as_tensor([[token_id]], device="cuda"),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values
            # [1, 1, LLM_new_vocab_size]

        # save embeddings in order
        if token_id is not None and token_id >= LLM_tokenizer.img_start_token_id:
            print('Saving LLM embeddings...', token_id)
            llm_img_token_states.append(out.last_hidden_state)

        # mapping to vocabulary
        temperature = args.temperature
        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token_id = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token_id = int(torch.multinomial(probs, num_samples=1))
            # [LLM_new_vocab_size], [LLM_new_vocab_size]

        # output_ids decode
        output_text_ids.append(token_id)
        if token_id == LLM_tokenizer.eos_token_id or len(llm_img_token_states) == model_.config.num_new_tokens:
            break

    print(output_text_ids)
    print(LLM_tokenizer.decode(output_text_ids))
    llm_img_token_states = torch.cat(llm_img_token_states, dim=1)
    num_new_tokens = 32
    assert llm_img_token_states.shape[1] == num_new_tokens
    print('llm_img_token_states:', llm_img_token_states, llm_img_token_states.shape)
    # [1, query_length, LLM_hidden_size]

    return model_.inference_sd_qformer(llm_img_token_states)

########################################################################################################################################################################
# LLaVA inference
@torch.inference_mode()
def generate_LLaVA_image(CLIP_image_features_llm_input, text_prompt, LLM_tokenizer, model_):
    # Step 1. Choose Human-GPT templates
    conversation_templates = []
    with open(editing_template, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Human: '):
                d = dict()
                d['Human'] = line[len("Human: "):]
                conversation_templates.append(d)
            elif line.startswith('GPT: '):
                conversation_templates[-1]['GPT'] = line[len("GPT: "):]
    conversation_template = random.choice(conversation_templates)

    # image embeddings + text embeddings for edited prompt
    original_image_embeddings = CLIP_image_features_llm_input
    edited_prompt = text_prompt
    both_condition_embeddings = get_prompt_embedding_LLM(img_embeddings=original_image_embeddings,
                                                         text_prompt=edited_prompt,
                                                         conversation_template=conversation_template,
                                                         LLM_tokenizer=LLM_tokenizer,
                                                         model_=model_,
                                                         condition_prompt='positive')
    both_condition_embeddings = both_condition_embeddings.to(torch.float16)
    print('both conditional prompt_embedding:', both_condition_embeddings, both_condition_embeddings.shape)
    # [1, CLIP_model_max_length, SD_qformer_hidden_size=CLIP_test_dim]
    return both_condition_embeddings


########################################################################################################################################################################
import PIL
import shutil
from InstructPix2PixSD_SM import StableDiffusionInstructPix2PixPipeline_modulated
from model.DS_SmartEdit_model import SmartEdit_model
from model.unet_2d_condition_ZeroConv import UNet2DConditionModel_ZeroConv
from conversation_v01 import SeparatorStyle, get_conv_template

def main():
    # Unet and LLM with LoRA
    os.makedirs(args.save_dir, exist_ok=True)
    adapter_name = 'default'
    total_dir = args.total_dir
    LLM_sub_dir = total_dir + '/LLM-' + f'{args.steps}'
    embeddings_qformer_Path = total_dir + '/embeddings_qformer/checkpoint-' + f'{args.steps}' + '_embeddings_qformer.bin'

    # model path
    sd_qformer_version = args.sd_qformer_version
    model_name_or_path = args.model_name_or_path
    LLaVA_model_path = args.LLaVA_model_path
    SD_IP2P_path = "timbrooks/instruct-pix2pix"
    sd_model_name_or_path = "runwayml/stable-diffusion-v1-5"

    # "v1.1-7b"
    if sd_qformer_version == "v1.1-7b":
        LLaVA_00002_weights = LLaVA_model_path + "/pytorch_model-00002-of-00002.bin"
    # "v1.1-13b"
    elif sd_qformer_version == "v1.1-13b":
        LLaVA_00002_weights = LLaVA_model_path + "/pytorch_model-00003-of-00003.bin"

    # Load SmartEdit_model
    model_ = SmartEdit_model.from_pretrained(
        model_name_or_path,
        cache_dir=None
    )

    # init llm tokenizer
    model_max_length = 512
    cache_dir = None
    LLM_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # TODO: ablation on pad_token
    LLM_tokenizer.pad_token = LLM_tokenizer.unk_token

    # init CLIP-ViT feature extractor
    model_.init_visual_features_extractor(LLaVA_model_path=LLaVA_model_path, sd_qformer_version=sd_qformer_version)

    # setup new llm tokens -> conversation system num_new_tokens=35: "<img>"(system message) + 32001='<im_start>', 32002='<im_end>' + " <img_0> ... <img_31>" -> len(llm_tokenizer)=32035
    editing_max_length = 512
    num_new_tokens = 32
    model_.setup_tokens_for_conversation(
        LLM_tokenizer,
        num_new_tokens=num_new_tokens,
        tune_new_embeddings=True,
        editing_template=editing_template,
        editing_max_length=editing_max_length)

    # init q-former that link SD
    model_.init_sd_qformer(
        num_hidden_layers=6
    )

    # Add LoRA for LLaMA
    # Load LLM with lora checkpoint -> type(model_.model)
    from DS_PeftForLoRA import PeftModel_for_LLM
    src_lora_file = total_dir + "/adapter_config.json"
    dst_lora_file = LLM_sub_dir + "/adapter_config.json"
    shutil.copy(src_lora_file, dst_lora_file)
    model_.model = PeftModel_for_LLM.from_pretrained(model_.model, LLM_sub_dir, adapter_name=adapter_name)
    LLM_sub_dir = LLM_sub_dir + '/adapter_model.bin'
    model_.load_pretrained_LLaMA_for_inference(pretrained_LLaMA=LLM_sub_dir)
    # pretrained checkpoint for SD-QFormer
    model_.load_pretrained_for_inference(pretrain_model=embeddings_qformer_Path, LLaVA_00002_weights=LLaVA_00002_weights)

    # load BIM module
    BIM_path = total_dir + f'/modulate-{args.steps}/adapter_model.bin'
    model_.init_BIM_module()
    model_.load_BIM_module_for_inference(modulate_path=BIM_path)

    ########################################################################################################################################################################
    # inference preparation
    print('LLM vocabulary size:', LLM_tokenizer.vocab_size)
    model_.to(dtype=torch.float32, device="cuda")
    model_.eval()
    # sum([p.nelement() for p in model_.parameters()])

    # loading inference type
    is_understanding_scenes = args.is_understanding_scenes
    is_reasoning_scenes = args.is_reasoning_scenes
    original_image_ViT_resolution = 224
    resize_resolution = args.resize_resolution

    # 1. is_understanding_scenes
    if is_understanding_scenes == 'True':
        # save 5 different types
        ReasonEdit_benckmark_dir = args.ReasonEdit_benckmark_dir
        benckmark_understanding_scenes_dir = []
        for root, dirs, files in os.walk(ReasonEdit_benckmark_dir):
            for dir in dirs:
                if dir.endswith('1-Left-Right') or dir.endswith('2-Relative-Size') or dir.endswith('3-Mirror') or dir.endswith('4-Color') or dir.endswith('5-Multiple-Objects'):
                    print(os.path.join(root, dir))
                    sub_path = os.path.join(root, dir)
                    benckmark_understanding_scenes_dir.append(sub_path)

        # choose sub-path
        image_SD_1, both_1, modulated_image_feature_list_1 = [], [], []
        image_SD_2, both_2, modulated_image_feature_list_2 = [], [], []
        image_SD_3, both_3, modulated_image_feature_list_3 = [], [], []
        image_SD_4, both_4, modulated_image_feature_list_4 = [], [], []
        image_SD_5, both_5, modulated_image_feature_list_5 = [], [], []
        for sub_dir in benckmark_understanding_scenes_dir:
            if sub_dir.endswith("1-Left-Right") == True:
                test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
                with open(sub_dir + "/Left_Right_text.txt", 'r') as f:
                    prompt = f.readlines()

            if sub_dir.endswith("2-Relative-Size") == True:
                test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
                with open(sub_dir + "/Size_text.txt", 'r') as f:
                    prompt = f.readlines()

            if sub_dir.endswith("3-Mirror") == True:
                test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
                with open(sub_dir + "/Mirror_text.txt", 'r') as f:
                    prompt = f.readlines()

            if sub_dir.endswith("4-Color") == True:
                test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
                with open(sub_dir + "/Color_text.txt", 'r') as f:
                    prompt = f.readlines()

            if sub_dir.endswith("5-Multiple-Objects") == True:
                test_img_list = sorted(glob.glob(f'{sub_dir}/*.png'))
                with open(sub_dir + "/MultipleObjects_text.txt", 'r') as f:
                    prompt = f.readlines()

            for idx, img_path in enumerate(test_img_list):
                # original image for ViT
                text_prompt = prompt[idx]
                text_prompt = text_prompt.split("CLIP: ")[0].rstrip()
                # text_prompt = text_prompt.replace('\n', '')

                original_image_for_ViT = PIL.Image.open(img_path).convert("RGB")
                original_image_for_ViT = original_image_for_ViT.resize((original_image_ViT_resolution, original_image_ViT_resolution), resample=Image.Resampling.BICUBIC)
                original_image_for_ViT = model_.vision_tower.image_processor.preprocess(original_image_for_ViT, return_tensors='pt')['pixel_values'].to(dtype=torch.float32, device="cuda")
                CLIP_image_features = model_.vision_tower(original_image_for_ViT)
                CLIP_image_features_llm_input = model_.mm_projector(CLIP_image_features)
                # [1, mm_projection_length, LLM_hidden_size]

                # original image for SD
                original_image_for_SD = PIL.Image.open(img_path)
                original_image_for_SD = PIL.ImageOps.exif_transpose(original_image_for_SD)
                original_image_for_SD = original_image_for_SD.convert("RGB")
                print(img_path)

                # BIM module -> latent_w and latent_h
                both_condition_embeddings_origin = generate_LLaVA_image(CLIP_image_features_llm_input, text_prompt, LLM_tokenizer, model_)
                resize_resolution = resize_resolution
                latent_w = resize_resolution // 8
                latent_h = resize_resolution // 8
                modulated_image_features, modulated_SD_features = model_.inference_BIM_module(CLIP_image_features, both_condition_embeddings_origin.to(torch.float32),
                                                                                              latent_w=latent_w, latent_h=latent_h)
                modulated_image_features = modulated_image_features.to(torch.float16)
                modulated_SD_features = modulated_SD_features.to(torch.float16)
                both_condition_embeddings = both_condition_embeddings_origin + 0.5 * modulated_SD_features

                if sub_dir.endswith("1-Left-Right") == True:
                    image_SD_1.append(original_image_for_SD)
                    both_1.append(both_condition_embeddings)
                    modulated_image_feature_list_1.append(modulated_image_features)

                if sub_dir.endswith("2-Relative-Size") == True:
                    image_SD_2.append(original_image_for_SD)
                    both_2.append(both_condition_embeddings)
                    modulated_image_feature_list_2.append(modulated_image_features)

                if sub_dir.endswith("3-Mirror") == True:
                    image_SD_3.append(original_image_for_SD)
                    both_3.append(both_condition_embeddings)
                    modulated_image_feature_list_3.append(modulated_image_features)

                if sub_dir.endswith("4-Color") == True:
                    image_SD_4.append(original_image_for_SD)
                    both_4.append(both_condition_embeddings)
                    modulated_image_feature_list_4.append(modulated_image_features)

                if sub_dir.endswith("5-Multiple-Objects") == True:
                    image_SD_5.append(original_image_for_SD)
                    both_5.append(both_condition_embeddings)
                    modulated_image_feature_list_5.append(modulated_image_features)

    # 2. is_reasoning_scenes
    if is_reasoning_scenes == 'True':
        image_SD_reasoning, both_reasoning, modulated_image_feature_list_reasoning = [], [], []
        ReasonEdit_benckmark_dir = args.ReasonEdit_benckmark_dir
        test_dir = ReasonEdit_benckmark_dir + "/6-Reasoning"
        test_img_list = sorted(glob.glob(f'{test_dir}/*.png'))
        with open(test_dir + "/Reason_test.txt", 'r') as f:
            prompt = f.readlines()
        for idx, img_path in enumerate(test_img_list):
            text_prompt = prompt[idx]
            text_prompt = text_prompt.split("CLIP: ")[0].rstrip()
            # text_prompt = text_prompt.replace('\n', '')

            # original image for ViT
            original_image_for_ViT = PIL.Image.open(img_path).convert("RGB")
            original_image_for_ViT = original_image_for_ViT.resize((original_image_ViT_resolution, original_image_ViT_resolution), resample=Image.Resampling.BICUBIC)
            original_image_for_ViT = model_.vision_tower.image_processor.preprocess(original_image_for_ViT, return_tensors='pt')['pixel_values'].to(dtype=torch.float32, device="cuda")
            CLIP_image_features = model_.vision_tower(original_image_for_ViT)
            CLIP_image_features_llm_input = model_.mm_projector(CLIP_image_features)
            # [1, mm_projection_length, LLM_hidden_size]

            # original image for SD
            original_image_for_SD = PIL.Image.open(img_path)
            original_image_for_SD = PIL.ImageOps.exif_transpose(original_image_for_SD)
            original_image_for_SD = original_image_for_SD.convert("RGB")
            print(img_path)

            # BIM module -> latent_w and latent_h
            both_condition_embeddings_origin = generate_LLaVA_image(CLIP_image_features_llm_input, text_prompt, LLM_tokenizer, model_)
            resize_resolution = resize_resolution
            latent_w = resize_resolution // 8
            latent_h = resize_resolution // 8
            modulated_image_features, modulated_SD_features = model_.inference_BIM_module(CLIP_image_features, both_condition_embeddings_origin.to(torch.float32),
                                                                                          latent_w=latent_w, latent_h=latent_h)
            modulated_image_features = modulated_image_features.to(torch.float16)
            modulated_SD_features = modulated_SD_features.to(torch.float16)
            both_condition_embeddings = both_condition_embeddings_origin + 0.5 * modulated_SD_features

            image_SD_reasoning.append(original_image_for_SD)
            both_reasoning.append(both_condition_embeddings)
            modulated_image_feature_list_reasoning.append(modulated_image_features)

    ####################################################################################
    # Delete the first model and clear GPU memory cache
    del model_
    torch.cuda.empty_cache()

    # Load SD-v1.5
    pipe = StableDiffusionInstructPix2PixPipeline_modulated.from_pretrained(
        SD_IP2P_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    # initialize unet from UNet2DConditionModel_ZeroConv
    pipe.unet = UNet2DConditionModel_ZeroConv.from_pretrained(
        sd_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True)

    # initialize unet for concatenation
    in_channels = 8
    out_channels = pipe.unet.conv_in.out_channels
    new_conv_in = nn.Conv2d(in_channels, out_channels, pipe.unet.conv_in.kernel_size, pipe.unet.conv_in.stride, pipe.unet.conv_in.padding)
    pipe.unet.conv_in = new_conv_in

    # load unet checkpoint
    unet_sub_dir = total_dir + '/unet-' + f'{args.steps}'
    unet_ckpt = unet_sub_dir + '/adapter_model.bin'
    unet_ckpt = torch.load(unet_ckpt)
    unet_ckpt_new = {}
    for k, v in unet_ckpt.items():
        if 'unet.' in k:
            unet_ckpt_new[k[len('unet.'):]] = v
            # remove 'unet.'
    pipe.unet.load_state_dict(unet_ckpt_new, strict=True)
    pipe.unet.to(dtype=torch.float16, device="cuda")
    pipe.unet.eval()
    print('Loading unet checkpoint:', pipe.unet.load_state_dict(unet_ckpt_new, strict=True))

    # 1. understanding scenes inference pipeline
    if is_understanding_scenes == 'True':
        total_save_path = args.save_dir
        # choose sub-path
        for sub_dir in benckmark_understanding_scenes_dir:
            if sub_dir.endswith("1-Left-Right") == True:
                original_image_for_SD_generation = image_SD_1
                both_condition_embeddings_generation = both_1
                modulated_image_embeddings_generation = modulated_image_feature_list_1
                final_save_dir = total_save_path + "/LeftRight_1"
                final_save_dir_15 = total_save_path + "/LeftRight_metrics_1"

            if sub_dir.endswith("2-Relative-Size") == True:
                original_image_for_SD_generation = image_SD_2
                both_condition_embeddings_generation = both_2
                modulated_image_embeddings_generation = modulated_image_feature_list_2
                final_save_dir = total_save_path + "/RelativeSize_2"
                final_save_dir_15 = total_save_path + "/RelativeSize_metrics_2"

            if sub_dir.endswith("3-Mirror") == True:
                original_image_for_SD_generation = image_SD_3
                both_condition_embeddings_generation = both_3
                modulated_image_embeddings_generation = modulated_image_feature_list_3
                final_save_dir = total_save_path + "/Mirror_3"
                final_save_dir_15 = total_save_path + "/Mirror_metrics_3"

            if sub_dir.endswith("4-Color") == True:
                original_image_for_SD_generation = image_SD_4
                both_condition_embeddings_generation = both_4
                modulated_image_embeddings_generation = modulated_image_feature_list_4
                final_save_dir = total_save_path + "/Color_4"
                final_save_dir_15 = total_save_path + "/Color_metrics_4"

            if sub_dir.endswith("5-Multiple-Objects") == True:
                original_image_for_SD_generation = image_SD_5
                both_condition_embeddings_generation = both_5
                modulated_image_embeddings_generation = modulated_image_feature_list_5
                final_save_dir = total_save_path + "/MultipleObjects_5"
                final_save_dir_15 = total_save_path + "/MultipleObjects_metrics_5"

            # make different dir for both metrics and instruction-alignment
            os.makedirs(final_save_dir, exist_ok=True)
            os.makedirs(final_save_dir_15, exist_ok=True)
            for idx in range(len(original_image_for_SD_generation)):
                original_image_for_SD = original_image_for_SD_generation[idx]
                both_condition_embeddings = both_condition_embeddings_generation[idx]
                modulated_image_features = modulated_image_embeddings_generation[idx]

                # resize to the original training size or not
                is_resize = args.is_resize
                if is_resize == True:
                    original_image_for_SD = original_image_for_SD.resize((resize_resolution, resize_resolution), resample=Image.Resampling.BICUBIC)
                    text_guidance_scale_list = [7.5]
                    image_guidance_scale_list = [1.4, 1.5, 1.6]
                    # 3-CFG
                else:
                    text_guidance_scale_list = [7.5]
                    image_guidance_scale_list = [1.5]

                # inference
                seed = random.randint(0, 100000)
                generator = torch.Generator("cuda").manual_seed(seed)
                for text_ in text_guidance_scale_list:
                    for image_ in image_guidance_scale_list:
                        text_guidance_scale = text_
                        image_guidance_scale = image_
                        image_output = pipe(
                            prompt_embeds=both_condition_embeddings,
                            image=original_image_for_SD,
                            num_inference_steps=100,
                            guidance_scale=text_guidance_scale,
                            image_guidance_scale=image_guidance_scale,
                            modulated_image_feature=modulated_image_features,
                            generator=generator).images[0]
                        image_output.save(os.path.join(final_save_dir, f'{(idx + 1):04d}_T%s_I%s.png') % (text_guidance_scale, image_guidance_scale))
                        if image_guidance_scale == 1.5:
                            image_output.save(os.path.join(final_save_dir_15, f'{(idx + 1):04d}_T%s_I%s.png') % (text_guidance_scale, image_guidance_scale))
                print('Understanding Scenes Editing image %d' % (idx + 1), sub_dir)

    # 2. reasoning scenes inference pipeline
    if is_reasoning_scenes == 'True':
        original_image_for_SD_generation = image_SD_reasoning
        both_condition_embeddings_generation = both_reasoning
        modulated_image_embeddings_generation = modulated_image_feature_list_reasoning
        total_save_path = args.save_dir
        final_save_dir = total_save_path + "/Reasoning"
        final_save_dir_15 = total_save_path + "/Reasoning_metrics_1"

        # make different dir for both metrics and instruction-alignment
        os.makedirs(final_save_dir, exist_ok=True)
        os.makedirs(final_save_dir_15, exist_ok=True)
        for idx in range(len(original_image_for_SD_generation)):
            original_image_for_SD = original_image_for_SD_generation[idx]
            both_condition_embeddings = both_condition_embeddings_generation[idx]
            modulated_image_features = modulated_image_embeddings_generation[idx]

            # resize to the original training size or not
            is_resize = args.is_resize
            if is_resize == True:
                original_image_for_SD = original_image_for_SD.resize((resize_resolution, resize_resolution), resample=Image.Resampling.BICUBIC)
                text_guidance_scale_list = [7.5]
                image_guidance_scale_list = [1.4, 1.5, 1.6]
                # 3-CFG
            else:
                text_guidance_scale_list = [7.5]
                image_guidance_scale_list = [1.5]

            # inference
            seed = random.randint(0, 100000)
            generator = torch.Generator("cuda").manual_seed(seed)
            for text_ in text_guidance_scale_list:
                for image_ in image_guidance_scale_list:
                    text_guidance_scale = text_
                    image_guidance_scale = image_
                    image_output = pipe(
                        prompt_embeds=both_condition_embeddings,
                        image=original_image_for_SD,
                        num_inference_steps=100,
                        guidance_scale=text_guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        modulated_image_feature=modulated_image_features,
                        generator=generator).images[0]
                    image_output.save(os.path.join(final_save_dir, f'{(idx + 1):04d}_T%s_I%s.png') % (text_guidance_scale, image_guidance_scale))
                    if image_guidance_scale == 1.5:
                        image_output.save(os.path.join(final_save_dir_15, f'{(idx + 1):04d}_T%s_I%s.png') % (text_guidance_scale, image_guidance_scale))
            print('Reasoning Scenes Editing image %d' % (idx + 1))


########################################################################################################################################################################
"""
# execute DS_SmartEdit_test.py
* RE-1. is_understanding_scenes
python test/DS_SmartEdit_test.py --is_understanding_scenes True --model_name_or_path "./checkpoints/vicuna-7b-v1-1" --LLaVA_model_path "./checkpoints/LLaVA-7B-v1" --save_dir './checkpoints/SmartEdit-7B/Understand-15000' --steps 15000 --total_dir "./checkpoints/SmartEdit-7B" --sd_qformer_version "v1.1-7b" --resize_resolution 256
* RE-2. is_reasoning_scenes
python test/DS_SmartEdit_test.py --is_reasoning_scenes True --model_name_or_path "./checkpoints/vicuna-7b-v1-1" --LLaVA_model_path "./checkpoints/LLaVA-7B-v1" --save_dir './checkpoints/SmartEdit-7B/Reason-15000' --steps 15000 --total_dir "./checkpoints/SmartEdit-7B" --sd_qformer_version "v1.1-7b" --resize_resolution 256
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. vicuna-7b
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./vicuna-7b-v1-1",
    )
    # 2. llava-7b
    parser.add_argument(
        "--LLaVA_model_path",
        type=str,
        default="./LLaVA-7B-v1",
    )
    # 3. save_dir
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
    )
    # 4. total_dir
    parser.add_argument(
        '--total_dir',
        type=str,
        required=True,
    )
    # 5. steps
    parser.add_argument(
        '--steps',
        type=int,
        required=True,
    )
    # 6. temperature
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
    )
    # 7. is_understanding_scenes
    parser.add_argument(
        '--is_understanding_scenes',
        default=False
    )
    # 8. is_reasoning_scenes
    parser.add_argument(
        '--is_reasoning_scenes',
        default=False
    )
    # 9. sd_qformer_version
    parser.add_argument(
        '--sd_qformer_version',
        type=str,
        default="v1.1-7b"
    )
    # 10. is_resize
    parser.add_argument(
        '--is_resize',
        default=True
    )
    # 11. ReasonEdit_benckmark_dir
    parser.add_argument(
        '--ReasonEdit_benckmark_dir',
        type=str,
        default="./ReasonEdit_benckmark_dir"
    )
    # 12. resize_resolution
    parser.add_argument(
        '--resize_resolution',
        type=int,
        default=256
    )
    args = parser.parse_args()
    main()
