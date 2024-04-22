import argparse
import pdb

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


#############################################################################################################################
def main(args):
    # Model
    disable_torch_init()

    # llava.model.builder.py
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    # Conversation(system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    # roles=('USER', 'ASSISTANT'), messages=[], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    # [1, 3, 224, 224]

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end == False:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                print('1:', inp)
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                print('2:', inp)
            # 1. '<im_start><image><im_end>\nDescribe the image.'
            # 2. '<image>\nDescribe the image.'
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <im_start><image><im_end>\nDescribe the image. ASSISTANT:"

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

"""
# python3 -m llava.model.apply_delta --base /path/to/llama-7b --target /output/path/to/LLaVA-7B-v0 --delta liuhaotian/LLaVA-7b-delta-v0
# python3 -m llava.model.apply_delta --base /group/30098/liangbinxie/Projects/LLM/LLaVA/models/llama-7b-new --target /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-7B-v1_1 --delta /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-delta-v1-1

export PYTHONPATH=$PYTHONPATH:/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLaVA230730
export PYTHONPATH=$PYTHONPATH:/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLaVA101
cd /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA101
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/view_llava1.jpg"
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-7B-v1_1 --image-file "./images/view_llava1.jpg"


# 1. two birds 一般
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0002.png"
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-7B-v1_1 --image-file "./images/0002.png"
Describe the image.
What are the differences between these two birds?
ASSISTANT: The image shows two small birds sitting on a tree branch. They are perched on a branch with green leaves, and their presence creates a peaceful and serene atmosphere.
ASSISTANT: The two birds have different appearances, with one being a brown bird and the other being a white bird. They are sitting close to each other on the tree branch, which adds to the overall harmony of the scene.

# 2. cat-dog 成功
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0007.png"
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-7B-v1_1 --image-file "./images/0007.png"
What is the animal on the left?
What is the left animal?
What is the animal on the right?
What is the right animal?
Describe the image.
ASSISTANT: The animal on the left is a cat.
ASSISTANT: The left animal is a cat.
ASSISTANT: The animal on the right is a dog.
ASSISTANT: The right animal is a dog.
ASSISTANT: The image is a close-up of a dog and a cat standing next to each other in a grassy field. They are both looking at the camera, and their expressions seem curious and engaged. The scene captures a moment of interaction and curiosity between the two animals, showcasing their natural curiosity and fascination with each other.

# 3. cat-dog 成功
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0012.png"
CUDA_VISIBLE_DEVICES=1 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-7B-v1_1 --image-file "./images/0012.png"
What is the animal on the left?
What is the left animal?
What is the animal on the right?
What is the right animal?
Describe the image.
Where is the cat among the dogs?
ASSISTANT: The animal on the left is a cat.
ASSISTANT: The left animal is a cat.
ASSISTANT: The animal on the right is a dog.
ASSISTANT: The right animal is a dog.
ASSISTANT: The image is a close-up of two animals, a cat and a dog, laying down together in a grassy field. They are both resting comfortably, with the cat on the left and the dog on the right. The scene captures a peaceful moment between the two animals, showcasing their companionship and relaxation in a natural setting.
ASSISTANT: The cat is laying next to the dog, with the dog on the right and the cat on the left.

# 4. cat-dog 失败
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0013.png"
CUDA_VISIBLE_DEVICES=1 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-7B-v1_1 --image-file "./images/0013.png"
What is the animal on the left?
What is the left animal?
What is the animal on the right?
What is the right animal?
Describe the image.
ASSISTANT: The animal on the left is a dog.
ASSISTANT: The left animal is a dog.
ASSISTANT: The animal on the right is a cat.
ASSISTANT: The right animal is a cat.
ASSISTANT: The image is a black and white photograph of a dog and a cat standing together in a field of yellow flowers. The two animals are looking at the camera, and they appear to be enjoying their time together in the field. The scene is peaceful and serene, with the yellow flowers adding a touch of natural beauty to the image. The photograph captures a moment of harmony and companionship between the two animals, showcasing their ability to coexist and share a common space.

# 5. baby in the mirror 还行
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0030.png"
Describe the image.
How to differentiate the baby and his reflection?
Describe the baby in the mirror.
ASSISTANT: The image is a close-up of a baby's face, with the baby looking at its own reflection in a mirror. The baby is holding up its hand, possibly trying to touch its reflection.
ASSISTANT: The baby and his reflection can be differentiated by observing the baby's facial expression and body language. The baby's reflection will appear as a mirror image of the baby, but the baby's facial expression and body language will be unique to the baby himself.
ASSISTANT: The baby in the mirror is a young child, likely a toddler, who is looking at his reflection and holding up his hand. The baby's reflection is a mirror image of the child, but the baby's facial expression and body language will be unique to the child himself.

# 6. baby in the mirror 失败
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0031.png"
Describe the image.
How to differentiate the baby and his reflection?
Describe the baby in the mirror.
ASSISTANT: The image is a close-up of a baby boy and a baby girl, both wearing blue pajamas, smiling and touching each other's hands. They are standing next to each other, possibly in a bedroom setting.

# 7. baby in the mirror 成功
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0032.png"
Describe the image.
How to differentiate the baby and his reflection?
ASSISTANT: The image features a baby boy looking at his reflection in a mirror. The baby is sitting on a table, and the mirror is placed in front of him, allowing him to see his own reflection.
ASSISTANT: The baby and his reflection can be differentiated by observing the baby's facial expressions and body language. The baby will likely be actively engaged in looking at his reflection, whereas the reflection will be a static image of the baby's reflection in the mirror. Additionally, the baby may be making facial expressions or gestures that are not visible in the reflection.

# 8. cat in the mirror 成功
CUDA_VISIBLE_DEVICES=1 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/0033.png"
Describe the image.
How to differentiate the cat and its reflection?
ASSISTANT: The image features a white and gray cat sitting on a wooden counter or table, looking at its reflection in a mirror.
ASSISTANT: The cat and its reflection can be differentiated by their positions and the fact that the cat is a real, physical cat, while the reflection is an image of the cat in the mirror.

# 9. cat in the mirror 成功
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/mirrorcat.jpg"
Describe the image.
How to differentiate the cat and its reflection?
Where is the cat in the mirror?
ASSISTANT: The image features a cat sitting on a yellow floor, looking at its reflection in a mirror.
ASSISTANT: The cat and its reflection can be differentiated by observing the cat's body position and the angle at which it is looking at the mirror. The cat is sitting on the yellow floor, while its reflection is shown in the mirror, which is positioned above the cat. Additionally, the cat's reflection might appear slightly blurry or distorted due to the mirror's reflection properties.
ASSISTANT: The cat is sitting on the yellow floor in the reflection, which is shown in the mirror above it.

# 10. cat-dog 成功
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/cat_dog.jpg"
What is the animal on the left?
What is the left animal?
What is the animal on the right?
What is the right animal?
Describe the image.
ASSISTANT: The animal on the left is a dog.
ASSISTANT: The left animal is a dog.
ASSISTANT: The animal on the right is a cat.
ASSISTANT: The right animal is a cat.
ASSISTANT: The image is a close-up of two animals, a dog and a cat, laying down together on a floor. They are both relaxed and appear to be comfortable in each other's company. The scene captures a moment of harmony and companionship between the two animals, showcasing their ability to coexist peacefully.

# 11. goat-cat 成功
CUDA_VISIBLE_DEVICES=1 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/goat_and_cat.jpg"
What is the animal on the left?
What is the left animal?
What is the animal on the right?
What is the right animal?
Describe the image.
ASSISTANT: The animal on the left is a goat.
ASSISTANT: The left animal is a goat.
ASSISTANT: The animal on the right is a cat.
ASSISTANT: The right animal is a cat.
ASSISTANT: The image is a black and white photograph of a goat and a cat standing next to each other on a rocky surface, such as a rocky beach or a rocky cliff. The two animals are posing together, creating a unique and interesting scene.

# 12. two horses 失败
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/pexels-photo-5732892.jpg"
What color is the left horse?
What color is the right horse?
ASSISTANT: The left horse is brown.
ASSISTANT: The right horse is black.

# 13. two dogs 成功
CUDA_VISIBLE_DEVICES=1 python -m llava.serve.cli --model-path /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 --image-file "./images/two_dogs_with_checkered_shirts1.jpg"
Describe the image.
What color is the left dog?
What color is the right dog?
ASSISTANT: The image shows two dogs, one black and one brown, sitting on a wooden dock or pier next to a body of water. They are wearing matching plaid jackets, which adds a cute and coordinated touch to their appearance.
ASSISTANT: The left dog is black.
ASSISTANT: The right dog is brown.


cp -r /group/30042/xlai/LLaVA-Lightning-7B-delta-v1-1 /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730
cp -r /group/30042/xlai/LLaVA/LLaVA-Lightning-7B-v1-1 /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730
cp -r /group/30042/xlai/clip-vit-large-patch14 /group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730
scp -r -P 36000 D:\X_Python_1\vilmedic-main\diffusion_priors-main\LLMSD_x1\LLaVA101 root@9.134.230.214:/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1
scp -r -P 36000 root@9.134.230.214:/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1/config.json D:\
scp -r -P 36000 D:\config.json root@9.134.230.214:/group/30098/yuzhouhuang/X_Python_1/vilmedic-main/diffusion_priors-main/LLMSD_x1/LLaVA230730/LLaVA-Lightning-7B-v1-1 
"""
