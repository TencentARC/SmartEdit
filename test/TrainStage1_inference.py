""" SmartEdit inference for text alignment """

import argparse
import numpy as np
import os
import random
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import transformers
from conversation_v01 import get_conv_template

# load stage-1 model
from model.LLMSD_modelv01_conv import AlignLLMwithSDCLIP

test_prompt_list = [
    "a photo of an astronaut riding a horse on mars",
    "A photo of a teddy bear made of water.",
    "A close-up of two chameleons wearing karate uniforms and fighting, jumping over a waterfall.",
    "A horse sitting on an astronaut's shoulders.",
    "a girl wearing a red dress, she is dancing.",
    "A boy wearing glasses, he is reading a thick book.",
    "A little cut boy.",
    "A woman wearing a green sportswear, she is running.",
    "A woman wearing a purple hat and a yellow scarf.",
    "A man wearing a black leather jacket and a red tie.",
    "A little boy with glasses and a watch.",
    "A smiling little girl.",
    "A cat wearing a hat.",
    "A dog in a bucket.",
    "A black and white panda.",
    "A little girl holding flowers.",
    "A cube made of brick",
    "A painting by grant wood of an astronaut couple,american gothic style",
    "A sign that says text to image",
    "Darth vader playing-with raccoon in mars during sunset",
    "A person walking in afield of red flowers with a yellow dress",
    "A person walking in afield of yellow flowers with a red dress",
    "A horse getting wet",
    "An astronaut riding a horse as a pencil drawing",
    "A virus monster is playing guitar, oil on canvas",
    "There is a penguin with a dog head standing",
    "A green colored banana",
    "A pink colored car",
    "A stop sign on the right of a refrigerator",
    "An elephant under the sea.",
    "A blanket is on a dog with spots.",
    "A child with a fake beard and an adult.",
    "A human being on a device that intercepts the sun's rays.",
    "A parked car is partially inside a tree.",
    "A person is close to the water and in the sand.",
    "A person stands and a dog sits.",
    "Blue pants and green top",
    "Wood floors with concrete-walls.",
    "A pink car",
    "A shark in the dessert."
]
DEFAULT_NEGATIVE_PROMPT = 'worst quality, low quality'
negative_prompt_embedding = None


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
    Args:
        seed (`int`): The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_img_tokens(prompt):
    ret = prompt
    for i in range(args.num_new_tokens):
        ret += f" <img_{i}>"
    return ret


def get_prompt_embedding(caption, llm_tokenizer, llm_model):
    conv = get_conv_template("vicuna_v1.2")
    inp = f'Can you create a picture based on the description "{caption}"'
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    temperature = args.temperature
    max_new_tokens = 256

    input_ids = llm_tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    llm_img_token_states = []
    token = None
    for i in range(max_new_tokens):
        if i == 0:
            out = llm_model.inference_llm(torch.as_tensor([input_ids], device="cuda"), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device="cuda")
            out = llm_model.inference_llm(
                input_ids=torch.as_tensor([[token]], device="cuda"),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        if token is not None and token >= llm_tokenizer.img_start_token_id:
            print('Saving LLM embeddings...', token)
            llm_img_token_states.append(out.last_hidden_state)

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)
        if token == llm_tokenizer.eos_token_id:
            break

    print(llm_tokenizer.decode(output_ids))
    llm_img_token_states = torch.cat(llm_img_token_states, dim=1)
    assert llm_img_token_states.shape[-2] == args.num_new_tokens
    print('llm_img_token_states: ', llm_img_token_states)

    return llm_model.inference_qformer(llm_img_token_states)


@torch.inference_mode()
def generate_image(prompt, idx, llm_tokenizer, llm_model, pipe):
    prompt_embedding = get_prompt_embedding(prompt, llm_tokenizer, llm_model)
    print('prompt_embedding: ', prompt_embedding)

    global negative_prompt_embedding
    if negative_prompt_embedding is None:
        negative_prompt_embedding = get_prompt_embedding(args.neg_prompt, llm_tokenizer, llm_model)

    set_seed(42)
    image = pipe(
        height=512,
        width=512,
        prompt_embeds=prompt_embedding,
        negative_prompt_embeds=negative_prompt_embedding,
        guidance_scale=args.cfg_scale).images[0]

    if not args.get_orig_out:
        image.save(os.path.join(args.save_dir, f'{idx:04d}.png'))
    else:
        set_seed(42)
        orig_image = pipe(
            height=512,
            width=512,
            prompt=prompt,
            negative_prompt=args.neg_prompt,
            guidance_scale=args.cfg_scale,
        ).images[0]
        new_image = Image.new(image.mode, (1026, 512))
        new_image.paste(orig_image, box=(0, 0))
        new_image.paste(image, box=(514, 0))
        image_name = '_'.join(prompt.split(' '))
        new_image.save(os.path.join(args.save_dir, f'{image_name}.png'))


#############################################################################################################################
def main():
    os.makedirs(args.save_dir, exist_ok=True)
    # load stable diffusion
    sd_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_name_or_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
        padding_side="right",
        use_fast=False,
    )
    # TODO: ablation on pad_token
    llm_tokenizer.pad_token = llm_tokenizer.unk_token

    # load llm
    llm_model = AlignLLMwithSDCLIP.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
    )

    ####################################################################################
    # load LLaVA version for checkpoint
    LLaVA_version = args.LLaVA_version
    if LLaVA_version == "v1.1-7b":
        LLaVA_model_v1_1_7b_path = args.LLaVA_model_path
        llm_model.load_LLaVA_ckpt_v1_1_7b(LLaVA_model_path_v1_1_7b=LLaVA_model_v1_1_7b_path)
    elif LLaVA_version == "v1.1-13b":
        LLaVA_model_v1_1_13b_path = args.LLaVA_model_path
        llm_model.load_LLaVA_ckpt_v1_1_13b(LLaVA_model_path_v1_1_13b=LLaVA_model_v1_1_13b_path)

    # load sd_qformer
    llm_model.init_qformer(
        num_hidden_layers=6
    )

    # setup new llm tokens
    llm_model.setup_tokens(llm_tokenizer, num_new_tokens=args.num_new_tokens)

    # load pretrain
    llm_model.load_pretrain(args.pretrain_model)
    llm_model.eval()
    llm_model.to(dtype=torch.float32, device="cuda")

    # generation
    prompt_list = test_prompt_list
    for idx, prompt in enumerate(prompt_list):
        generate_image(prompt, idx, llm_tokenizer, llm_model, pipe)


#############################################################################################################################
"""
# execute TrainStage1_inference.py
python test/TrainStage1_inference.py --model_name_or_path "./checkpoints/vicuna-7b-v1-1" --LLaVA_model_path "./checkpoints/LLaVA-7B-v1" --save_dir './checkpoints/stage1_CC12M_alignment_7b/Results-100000' --pretrain_model "./checkpoints/stage1_CC12M_alignment_7b/embeddings_qformer/checkpoint-150000.bin" --get_orig_out --LLaVA_version "v1.1-7b"
python test/TrainStage1_inference.py --model_name_or_path "./checkpoints/vicuna-13b-v1-1" --LLaVA_model_path "./checkpoints/LLaVA-13B-v1" --save_dir './checkpoints/stage1_CC12M_alignment_13b/Results-100000' --pretrain_model "./checkpoints/stage1_CC12M_alignment_13b/embeddings_qformer/checkpoint-150000.bin" --get_orig_out --LLaVA_version "v1.1-13b"
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. model_name_or_path
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./vicuna-7b-v1-1",
    )
    # 2. LLaVA_model_path
    parser.add_argument(
        "--LLaVA_model_path",
        type=str,
        default="./LLaVA-7B-v1",
    )
    # 3. neg_prompt
    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )
    # 4. save_dir
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
    )
    # 5. pretrain_model for stage-1
    parser.add_argument(
        '--pretrain_model',
        type=str,
        required=True,
    )
    # 6. get_orig_out -> create original sd-1.5 results
    parser.add_argument(
        '--get_orig_out',
        action='store_true',
    )
    # 7. num_new_tokens
    parser.add_argument(
        '--num_new_tokens',
        type=int,
        default=32,
    )
    # 8. cfg_scale
    parser.add_argument(
        '--cfg_scale',
        type=float,
        default=7.5,
    )
    # 9. temperature
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
    )
    # 10. LLaVA_version
    parser.add_argument(
        '--LLaVA_version',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main()
