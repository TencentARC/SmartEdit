#############################################################################################################################
# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# --------------------------------------------------------

from __future__ import annotations
import os
import pdb
import random
import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

import sys
import os.path as osp
import json
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
from pycocotools import mask

"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google
The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

class REFER:
    def __init__(self, data_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'), fix_imports=True)

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    # 1.
    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    # 2.
    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    # 3.
    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    # 4.
    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']
        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


#############################################################################################################################
#############################################################################################################################
from torchvision import transforms
from diffusers.utils import PIL_INTERPOLATION
from torchvision.transforms.functional import InterpolationMode
from conversation_v01 import SeparatorStyle, get_conv_template
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# 1. RefCOCODataset
class RefCOCODataset(Dataset):
    def __init__(
            self,
            path,
            path_coco,
            split="train",
            transparency=0.0,
            # Newly added...
            InstructDiffusion_color_template='./LLMSD_InstructDiffusion_color.txt',
            InstructDiffusion_seg_template='./LLMSD_InstructDiffusion_seg.txt',
            Refcoco_resolution_ViT=224,
            Refcoco_resolution_for_SD=256,
            CLIPImageProcessor=None,
            mm_projection_length=256,
            editing_template=None,
            editing_max_length=None,
            llm_tokenizer=None
    ):
        # initialize settings
        assert split in ("train", "val", "test")
        self.path = path
        self.transparency = transparency

        # refcoco loading
        self.G_ref_dataset = REFER(data_root=path)
        self.IMAGE_DIR = os.path.join(path_coco, 'train2014')
        self.list_ref = self.G_ref_dataset.getRefIds(split=split)

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

        # 224, 256
        self.Refcoco_resolution_ViT = Refcoco_resolution_ViT
        self.Refcoco_resolution_for_SD = Refcoco_resolution_for_SD

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

    def __len__(self):
        return len(self.list_ref)

    def __getitem__(self, i):
        ref_ids = self.list_ref[i]
        ref = self.G_ref_dataset.loadRefs(ref_ids)[0]

        ####################################################################################
        # Vicuna conversation system construction for image editing task...
        # Step 0. Make prompts for Refcoco
        sentences = random.choice(ref['sentences'])['sent']
        # 'lady with back to us' or 'blue shirt' or 'the lady with the blue shirt'
        prompt_for_refcoco = random.choice(self.seg_diverse_prompt_list)
        color = random.choice(self.color_list)
        color_name = color[0]
        prompt_for_refcoco = prompt_for_refcoco.format(color=color_name.lower(), object=sentences.lower())
        # Update the pixels of {object} to {color}, but leave the other pixels untouched. + 'Black' + 'blue shirt' -> 'Update the pixels of blue shirt to black, but leave the other pixels untouched.'

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
        edited_prompt = prompt_for_refcoco
        DEFAULT_IM_START_TOKEN = '<im_start>'
        DEFAULT_IM_END_TOKEN = '<im_end>'
        edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
        conversation_template = random.choice(conversation_templates)
        conv.messages = []
        conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
        conv.append_message(roles["GPT"], conversation_template["GPT"].replace(' [img].', append_str))
        conversation = conv.get_prompt()
        conversation = conversation.replace("\n", "")

        # 1. Edited Prompt input_ids -> Tokenize conversations
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

        # 2. Generated caption targets for Language Model loss
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

        # read image and mask
        image_name = self.G_ref_dataset.loadImgs(ref['image_id'])[0]['file_name']
        image_path = os.path.join(self.IMAGE_DIR, image_name)
        image = Image.open(image_path).convert("RGB")

        # 3. Original Image for ViT
        refcoco_original_img_ViT = image
        refcoco_original_img_ViT = refcoco_original_img_ViT.resize((self.Refcoco_resolution_ViT, self.Refcoco_resolution_ViT), resample=Image.Resampling.BICUBIC)
        refcoco_original_img_ViT = self.CLIPImageProcessor.preprocess(refcoco_original_img_ViT, return_tensors='pt')['pixel_values']
        refcoco_original_img_ViT = refcoco_original_img_ViT[0]

        # no augmentation crop, instead resize directly
        image = image.resize((self.Refcoco_resolution_for_SD, self.Refcoco_resolution_for_SD), resample=Image.Resampling.BICUBIC)
        image = np.asarray(image, dtype=np.uint8)
        mask = self.G_ref_dataset.getMask(ref=ref)['mask']
        mask = Image.fromarray(mask).resize((self.Refcoco_resolution_for_SD, self.Refcoco_resolution_for_SD), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask, dtype=np.int64)
        mask = (mask == 1)

        # 4. Edited Image for SD & 5. Original Image for SD
        refcoco_original_img_SD = Image.fromarray(image)
        refcoco_edited_img_SD = copy.deepcopy(image)
        R, G, B = color[3].split(",")
        R = int(R)
        G = int(G)
        B = int(B)
        refcoco_edited_img_SD[:, :, 0][mask] = self.transparency * refcoco_edited_img_SD[:, :, 0][mask] + (1 - self.transparency) * R
        refcoco_edited_img_SD[:, :, 1][mask] = self.transparency * refcoco_edited_img_SD[:, :, 1][mask] + (1 - self.transparency) * G
        refcoco_edited_img_SD[:, :, 2][mask] = self.transparency * refcoco_edited_img_SD[:, :, 2][mask] + (1 - self.transparency) * B
        refcoco_edited_img_SD = Image.fromarray(refcoco_edited_img_SD)
        refcoco_original_img_SD = refcoco_original_img_SD.resize((self.Refcoco_resolution_for_SD, self.Refcoco_resolution_for_SD), Image.Resampling.BICUBIC)
        refcoco_edited_img_SD = refcoco_edited_img_SD.resize((self.Refcoco_resolution_for_SD, self.Refcoco_resolution_for_SD), Image.Resampling.BICUBIC)
        refcoco_original_img_SD = rearrange(2 * torch.tensor(np.array(refcoco_original_img_SD)).float() / 255 - 1, "h w c -> c h w")
        refcoco_edited_img_SD = rearrange(2 * torch.tensor(np.array(refcoco_edited_img_SD)).float() / 255 - 1, "h w c -> c h w")

        # 6. Edited Prompt attention_mask
        # ne(a, b) is a != b
        input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        # ge(a, b) is a >= b
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # check exception data -> pad_token
        if input_ids[-1] != self.llm_tokenizer.pad_token_id:
            print('Exception data sample:', prompt_for_refcoco)
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
            input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)
            # 4.
            generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)
            # 5.
            is_editing_task = torch.zeros(1)

        # Refcoco dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['Update the pixels of blue shirt to black, but leave the other pixels untouched.']
        return {'original_img': refcoco_original_img_ViT,
                'original_img_for_vae': refcoco_original_img_SD,
                'edited_img': refcoco_edited_img_SD,
                'input_ids': input_ids,
                'input_attention_mask': input_ids_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}


#############################################################################################################################
# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# --------------------------------------------------------

import pdb
import os
import random
import copy
import math
from pathlib import Path
from typing import Any

import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

import os.path as osp
import json
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from pycocotools import mask

"""
grefer v0.1
This interface provides access to gRefCOCO.
The following API functions are defined:
G_REFER      - REFER api class
getRefIds    - get ref ids that satisfy given filter conditions.
getAnnIds    - get ann ids that satisfy given filter conditions.
getImgIds    - get image ids that satisfy given filter conditions.
getCatIds    - get category ids that satisfy given filter conditions.
loadRefs     - load refs with the specified ref ids.
loadAnns     - load anns with the specified ann ids.
loadImgs     - load images with the specified image ids.
loadCats     - load category names with the specified category ids.
getRefBox    - get ref's bounding box [x, y, w, h] given the ref_id
showRef      - show image, segmentation or box of the referred object with the ref
getMaskByRef - get mask and area of the referred object given ref or ref ids
getMask      - get mask and area of the referred object given ref
showMask     - show mask of the referred object given ref
"""

class G_REFER:
    def __init__(self, data_root, dataset='grefcoco', splitBy='unc'):
        # provide data_root folder which contains grefcoco
        print('loading dataset %s into memory...' % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['grefcoco']:
            self.IMAGE_DIR = osp.join(data_root, 'images/train2014')
        else:
            raise KeyError('No refer dataset is called [%s]' % dataset)

        tic = time.time()

        # load refs from data/dataset/refs(dataset).json
        self.data = {}
        self.data['dataset'] = dataset

        ref_file = osp.join(self.DATA_DIR, f'grefs({splitBy}).p')
        if osp.exists(ref_file):
            self.data['refs'] = pickle.load(open(ref_file, 'rb'), fix_imports=True)
        else:
            ref_file = osp.join(self.DATA_DIR, f'grefs({splitBy}).json')
            if osp.exists(ref_file):
                self.data['refs'] = json.load(open(ref_file, 'rb'))
            else:
                raise FileNotFoundError('JSON file not found')

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    @staticmethod
    def _toList(x):
        return x if isinstance(x, list) else [x]

    @staticmethod
    def match_any(a, b):
        a = a if isinstance(a, list) else [a]
        b = b if isinstance(b, list) else [b]
        return set(a) & set(b)

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        Anns[-1] = None
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        availableSplits = []
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            if ref['split'] not in availableSplits:
                availableSplits.append(ref['split'])

            # add mapping related to ref
            if ref_id in Refs:
                print('Duplicate ref id')
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]

            category_id = self._toList(category_id)
            added_cats = []
            for cat in category_id:
                if cat not in added_cats:
                    added_cats.append(cat)
                    catToRefs[cat] = catToRefs.get(cat, []) + [ref]

            ann_id = self._toList(ann_id)
            refToAnn[ref_id] = [Anns[ann] for ann in ann_id]
            for ann_id_n in ann_id:
                annToRef[ann_id_n] = annToRef.get(ann_id_n, []) + [ref]

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        self.availableSplits = availableSplits
        print('index created.')

    # 1.
    def getRefIds(self, image_ids=[], cat_ids=[], split=[]):
        image_ids = self._toList(image_ids)
        cat_ids = self._toList(cat_ids)
        split = self._toList(split)

        for s in split:
            if s not in self.availableSplits:
                raise ValueError(f'Invalid split name: {s}')

        refs = self.data['refs']

        if len(image_ids) > 0:
            lists = [self.imgToRefs[image_id] for image_id in image_ids]
            refs = list(itertools.chain.from_iterable(lists))
        if len(cat_ids) > 0:
            refs = [ref for ref in refs if self.match_any(ref['category_id'], cat_ids)]
        if len(split) > 0:
            refs = [ref for ref in refs if ref['split'] in split]

        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], ref_ids=[]):
        image_ids = self._toList(image_ids)
        ref_ids = self._toList(ref_ids)

        if any([len(image_ids), len(ref_ids)]):
            if len(image_ids) > 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            ann_ids = [ann['id'] for ann in anns]
            if len(ref_ids) > 0:
                lists = [self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]
                anns_by_ref_id = list(itertools.chain.from_iterable(lists))
                ann_ids = list(set(ann_ids).intersection(set(anns_by_ref_id)))
        else:
            ann_ids = [ann['id'] for ann in self.data['annotations']]

        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = self._toList(ref_ids)

        if len(ref_ids) > 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    # 2.
    def loadRefs(self, ref_ids=[]):
        return [self.Refs[ref_id] for ref_id in self._toList(ref_ids)]

    def loadAnns(self, ann_ids=[]):
        if isinstance(ann_ids, str):
            ann_ids = int(ann_ids)
        return [self.Anns[ann_id] for ann_id in self._toList(ann_ids)]

    # 3.
    def loadImgs(self, image_ids=[]):
        return [self.Imgs[image_id] for image_id in self._toList(image_ids)]

    def loadCats(self, cat_ids=[]):
        return [self.Cats[cat_id] for cat_id in self._toList(cat_ids)]

    def getRefBox(self, ref_id):
        anns = self.refToAnn[ref_id]
        return [ann['bbox'] for ann in anns]  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ann):
        if not ann:
            return None
        if ann['iscrowd']:
            raise ValueError('Crowd object')
        image = self.Imgs[ann['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}

    # 4.
    def getMaskByRef(self, ref=None, ref_id=None, merge=False):
        if not ref and not ref_id:
            raise ValueError
        if ref:
            ann_ids = ref['ann_id']
            ref_id = ref['ref_id']
        else:
            ann_ids = self.getAnnIds(ref_ids=ref_id)

        if ann_ids == [-1]:
            img = self.Imgs[self.Refs[ref_id]['image_id']]
            return {
                'mask': np.zeros([img['height'], img['width']], dtype=np.uint8),
                'empty': True
            }

        anns = self.loadAnns(ann_ids)
        mask_list = [self.getMask(ann) for ann in anns if not ann['iscrowd']]

        if merge:
            merged_masks = sum([mask['mask'] for mask in mask_list])
            merged_masks[np.where(merged_masks > 1)] = 1
            return {
                'mask': merged_masks,
                'empty': False
            }
        else:
            return mask_list

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


#############################################################################################################################
#############################################################################################################################
from torchvision import transforms
from diffusers.utils import PIL_INTERPOLATION
from torchvision.transforms.functional import InterpolationMode
from transformers.trainer_pt_utils import LabelSmoother

# 2. GrefCOCODataset
class GrefCOCODataset(Dataset):
    def __init__(
            self,
            path,
            path_coco,
            split="train",
            transparency=0.0,
            # Newly added...
            InstructDiffusion_color_template='./LLMSD_InstructDiffusion_color.txt',
            InstructDiffusion_seg_template='./LLMSD_InstructDiffusion_seg.txt',
            gRefcoco_resolution_ViT=224,
            gRefcoco_resolution_for_SD=256,
            CLIPImageProcessor=None,
            mm_projection_length=256,
            editing_template=None,
            editing_max_length=None,
            llm_tokenizer=None
    ):
        # initialize settings
        assert split in ("train", "val", "test")
        self.path = path
        self.transparency = transparency

        # refcoco loading
        self.G_ref_dataset = G_REFER(data_root=path)
        # self.IMAGE_DIR = os.path.join(path_coco, 'images/train2014')
        self.IMAGE_DIR = os.path.join(path_coco, 'train2014')
        self.list_ref = self.G_ref_dataset.getRefIds(split=split)

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

        # 224, 256
        self.gRefcoco_resolution_ViT = gRefcoco_resolution_ViT
        self.gRefcoco_resolution_for_SD = gRefcoco_resolution_for_SD

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

    def __len__(self):
        return len(self.list_ref)

    def __getitem__(self, i):
        ref_ids = self.list_ref[i]
        ref = self.G_ref_dataset.loadRefs(ref_ids)[0]

        ####################################################################################
        # Vicuna conversation system construction for image editing task...
        # Step 0. Make prompts for gRefcoco
        sentences = random.choice(ref['sentences'])['sent']
        # giraffe on left
        prompt_for_grefcoco = random.choice(self.seg_diverse_prompt_list)
        color = random.choice(self.color_list)
        color_name = color[0]
        prompt_for_grefcoco = prompt_for_grefcoco.format(color=color_name.lower(), object=sentences.lower())
        # Set the {object} pixels to {color} and keep the other pixels in their original state. +  -> 'giraffe on left' + 'white' -> 'Set the giraffe on left pixels to white and keep the other pixels in their original state.'

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
        edited_prompt = prompt_for_grefcoco
        DEFAULT_IM_START_TOKEN = '<im_start>'
        DEFAULT_IM_END_TOKEN = '<im_end>'
        edited_prompt = DEFAULT_IM_START_TOKEN + f" <img_0> " + DEFAULT_IM_END_TOKEN + edited_prompt
        conversation_template = random.choice(conversation_templates)
        conv.messages = []
        conv.append_message(roles["Human"], conversation_template["Human"].replace('[cap]', f'"{edited_prompt}"'))
        conv.append_message(roles["GPT"], conversation_template["GPT"].replace(' [img].', append_str))
        conversation = conv.get_prompt()
        conversation = conversation.replace("\n", "")

        # 1. Edited Prompt input_ids -> Tokenize conversations
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

        # 2. Generated caption targets for Language Model loss
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

        # read image and mask
        image_name = self.G_ref_dataset.loadImgs(ref['image_id'])[0]['file_name']
        image_path = os.path.join(self.IMAGE_DIR, image_name)
        image = Image.open(image_path).convert("RGB")

        # 3. Original Image for ViT
        grefcoco_original_img_ViT = image
        grefcoco_original_img_ViT = grefcoco_original_img_ViT.resize((self.gRefcoco_resolution_ViT, self.gRefcoco_resolution_ViT), resample=Image.Resampling.BICUBIC)
        grefcoco_original_img_ViT = self.CLIPImageProcessor.preprocess(grefcoco_original_img_ViT, return_tensors='pt')['pixel_values']
        grefcoco_original_img_ViT = grefcoco_original_img_ViT[0]

        # no augmentation crop, instead resize directly
        image = image.resize((self.gRefcoco_resolution_for_SD, self.gRefcoco_resolution_for_SD), resample=Image.Resampling.LANCZOS)
        image = np.asarray(image, dtype=np.uint8)
        mask = self.G_ref_dataset.getMaskByRef(ref=ref, merge=True)['mask']
        mask = Image.fromarray(mask).resize((self.gRefcoco_resolution_for_SD, self.gRefcoco_resolution_for_SD), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask, dtype=np.int64)
        mask = (mask == 1)

        # 4. Edited Image for SD & 5. Original Image for SD
        grefcoco_original_img_SD = Image.fromarray(image)
        grefcoco_edited_img_SD = copy.deepcopy(image)
        R, G, B = color[3].split(",")
        R = int(R)
        G = int(G)
        B = int(B)
        grefcoco_edited_img_SD[:, :, 0][mask] = self.transparency * grefcoco_edited_img_SD[:, :, 0][mask] + (1 - self.transparency) * R
        grefcoco_edited_img_SD[:, :, 1][mask] = self.transparency * grefcoco_edited_img_SD[:, :, 1][mask] + (1 - self.transparency) * G
        grefcoco_edited_img_SD[:, :, 2][mask] = self.transparency * grefcoco_edited_img_SD[:, :, 2][mask] + (1 - self.transparency) * B
        grefcoco_edited_img_SD = Image.fromarray(grefcoco_edited_img_SD)
        grefcoco_original_img_SD = grefcoco_original_img_SD.resize((self.gRefcoco_resolution_for_SD, self.gRefcoco_resolution_for_SD), Image.Resampling.BICUBIC)
        grefcoco_edited_img_SD = grefcoco_edited_img_SD.resize((self.gRefcoco_resolution_for_SD, self.gRefcoco_resolution_for_SD), Image.Resampling.BICUBIC)
        grefcoco_original_img_SD = rearrange(2 * torch.tensor(np.array(grefcoco_original_img_SD)).float() / 255 - 1, "h w c -> c h w")
        grefcoco_edited_img_SD = rearrange(2 * torch.tensor(np.array(grefcoco_edited_img_SD)).float() / 255 - 1, "h w c -> c h w")

        # 6. Edited Prompt attention_mask
        # ne(a, b) is a != b
        input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        # ge(a, b) is a >= b
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # check exception data -> pad_token
        if input_ids[-1] != self.llm_tokenizer.pad_token_id:
            print('Exception data sample:', prompt_for_grefcoco)
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
            input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)
            # 4.
            generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)
            # 5.
            is_editing_task = torch.zeros(1)

        # gRefcoco dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['Set the giraffe on left pixels to white and keep the other pixels in their original state.']
        return {'original_img': grefcoco_original_img_ViT,
                'original_img_for_vae': grefcoco_original_img_SD,
                'edited_img': grefcoco_edited_img_SD,
                'input_ids': input_ids,
                'input_attention_mask': input_ids_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}


#############################################################################################################################
#############################################################################################################################
import pdb
import json
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import os
import random
import copy
from glob import glob

# 3. COCOStuffDataset
class COCOStuffDataset(Dataset):
    def __init__(
            self,
            path_coco_image,
            path_cocostuff_mask,
            split="train2017",
            # Newly added...
            InstructDiffusion_color_template='./LLMSD_InstructDiffusion_color.txt',
            InstructDiffusion_seg_template='./LLMSD_InstructDiffusion_seg.txt',
            transparency=0.5,
            empty_percentage=0.2,
            cocostuff_resolution_ViT=224,
            cocostuff_resolution_for_SD=256,
            CLIPImageProcessor=None,
            mm_projection_length=256,
            editing_template=None,
            editing_max_length=None,
            llm_tokenizer=None
    ):
        # path from coco and coco-stuff
        assert split in ("train2017", "val2017")
        self.split = split
        self.path_coco_image = path_coco_image
        self.path_cocostuff_mask = path_cocostuff_mask

        # settings for coco-stuff
        self.empty_percentage = empty_percentage
        self.transparency = transparency

        # path from coco-image
        if self.split in ["train2017", "val2017"]:
            file_list = sorted(glob(os.path.join(self.path_coco_image, self.split, "*.jpg")))
            assert len(file_list) > 0, "{} has no image".format(
                os.path.join(self.path_coco_image, self.split)
            )
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

        # LLMSD_InstructDiffusion_seg.txt
        seg_diverse_prompt_path = InstructDiffusion_seg_template
        self.seg_diverse_prompt_list = []
        with open(seg_diverse_prompt_path) as f:
            line = f.readline()
            while line:
                line = line.strip('\n')
                self.seg_diverse_prompt_list.append(line)
                line = f.readline()

        # LLMSD_InstructDiffusion_color.txt
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

        # coco-stuff label
        coco_label_list_path = self.path_cocostuff_mask + '/ft_local/coco_stuff_labels.txt'
        self.label_dict = {}
        with open(coco_label_list_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(": ")
                self.label_dict[int(line_split[0])] = line_split[1]
                line = f.readline()

        # 224, 256
        self.cocostuff_resolution_ViT = cocostuff_resolution_ViT
        self.cocostuff_resolution_for_SD = cocostuff_resolution_for_SD

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

    def __len__(self) -> int:
        length = len(self.files)
        return length

    def _augmentation_new(self, image, label):
        # Cropping
        h, w = label.shape
        if h > w:
            start_h = random.randint(0, h - w)
            end_h = start_h + w
            image = image[start_h:end_h]
            label = label[start_h:end_h]
        elif h < w:
            start_w = random.randint(0, w - h)
            end_w = start_w + h
            image = image[:, start_w:end_w]
            label = label[:, start_w:end_w]
        else:
            pass
        image = Image.fromarray(image).resize((self.cocostuff_resolution_for_SD, self.cocostuff_resolution_for_SD), resample=Image.Resampling.LANCZOS)
        image = np.asarray(image, dtype=np.uint8)
        label = Image.fromarray(label).resize((self.cocostuff_resolution_for_SD, self.cocostuff_resolution_for_SD), resample=Image.Resampling.NEAREST)
        label = np.asarray(label, dtype=np.int64)
        return image, label

    def __getitem__(self, i):
        image_id = self.files[i]

        # path from coco and coco-stuff
        img_path = os.path.join(self.path_coco_image, self.split, image_id + ".jpg")
        mask_path = os.path.join(self.path_cocostuff_mask, self.split, image_id + ".png")
        assert img_path.split('/')[-1][:-4] == mask_path.split('/')[-1][:-4]

        # label.dtype = dtype('uint8')
        image = Image.open(img_path).convert("RGB")

        # 1. Original Image for ViT
        cocostuff_original_img_ViT = copy.deepcopy(image)
        cocostuff_original_img_ViT = cocostuff_original_img_ViT.resize((self.cocostuff_resolution_ViT, self.cocostuff_resolution_ViT), resample=Image.Resampling.BICUBIC)
        cocostuff_original_img_ViT = self.CLIPImageProcessor.preprocess(cocostuff_original_img_ViT, return_tensors='pt')['pixel_values']
        cocostuff_original_img_ViT = cocostuff_original_img_ViT[0]

        # _augmentation_new
        image = np.asarray(image)
        label = Image.open(mask_path).convert("L")
        label = np.asarray(label)
        image, label = self._augmentation_new(image, label)

        # preserve 0-182
        label_list = np.unique(label)
        label_list = list(label_list)
        label_list_rest = [i for i in range(182)]
        for item in label_list_rest:
            if item in label_list:
                label_list_rest.remove(item)

        if 255 in label_list:
            label_list.remove(255)

        # find class and make instruction
        if len(label_list) != 0:
            label_idx = random.choice(label_list)
            if random.uniform(0, 1) < self.empty_percentage:
                label_idx = random.choice(label_list_rest)

            # find class and make instruction
            class_name = self.label_dict[label_idx + 1]
            prompt = random.choice(self.seg_diverse_prompt_list)
            color = random.choice(self.color_list)
            color_name = color[0]
            prompt = prompt.format(color=color_name.lower(), object=class_name.lower())
            R, G, B = color[3].split(",")
            R = int(R)
            G = int(G)
            B = int(B)
        else:
            label_idx = 200
            prompt = "leave the picture as it is."

        # give masks
        mask = (label == label_idx)
        image_1 = copy.deepcopy(image)
        if len(label_list) != 0:
            image_1[:, :, 0][mask] = self.transparency * image_1[:, :, 0][mask] + (1 - self.transparency) * R
            image_1[:, :, 1][mask] = self.transparency * image_1[:, :, 1][mask] + (1 - self.transparency) * G
            image_1[:, :, 2][mask] = self.transparency * image_1[:, :, 2][mask] + (1 - self.transparency) * B

        # 2. Edited Image for SD & 3. Original Image for SD
        image_0 = Image.fromarray(image)
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = Image.fromarray(image_1)
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        cocostuff_original_img_SD = image_0
        cocostuff_edited_img_SD = image_1

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
        edited_prompt = prompt
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
        input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)

        # 7. Generated caption targets attention mask
        # ge(a, b) is a >= b
        generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)

        # 8. task choosing
        is_editing_task = torch.ones(1)

        # check exception data -> pad_token
        if input_ids[-1] != self.llm_tokenizer.pad_token_id:
            print('Exception data sample:', prompt)
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
            input_ids_attention_mask = input_ids.ne(self.llm_tokenizer.pad_token_id)
            # 4.
            generated_caption_encoder_attention_mask = input_ids.ge(self.llm_tokenizer.img_start_token_id)
            # 5.
            is_editing_task = torch.zeros(1)

        # COCO-Stuff dataloader -> 3 parts -> [bs, 3, 224, 224] + [bs, 3, 256, 256], [bs, 3, 256, 256], ['Paint the pixels of truck in green and maintain the current appearance of the other pixels.']
        return {'original_img': cocostuff_original_img_ViT,
                'original_img_for_vae': cocostuff_original_img_SD,
                'edited_img': cocostuff_edited_img_SD,
                'input_ids': input_ids,
                'input_attention_mask': input_ids_attention_mask,
                'generated_caption_targets': generated_caption_targets,
                'generated_caption_encoder_attention_mask': generated_caption_encoder_attention_mask,
                'is_editing_task': is_editing_task}
