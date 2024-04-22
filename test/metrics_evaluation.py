# -*- coding: utf-8 -*-

import pdb
import argparse
import glob
import numpy as np
import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import transformers

########################################################################################################################################################################
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

# 1. MetricsCalculator
class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device

        # CLIP similarity
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)

        # background preservation
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ####################################################################################
    # 1. CLIP similarity
    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)

        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)

        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        return score

    # 2. PSNR
    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score

    # 3. LPIPS
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        score = score.cpu().item()
        return score

    # 4. MSE
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)
        score = self.mse_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous())
        score = score.cpu().item()
        return score

    # 5. SSIM
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score


########################################################################################################################################################################
# 2. compute metrics
def main_metrics():
    # initialize MetricsCalculator
    metrics_calculator = MetricsCalculator("cuda")
    clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")

    # save 5 different types
    benckmark_dir = args.benckmark_dir
    benckmark_understanding_scenes_dir = []
    for root, dirs, files in os.walk(benckmark_dir):
        for dir in dirs:
            if dir.endswith('1-Left-Right') or dir.endswith('2-Relative-Size') or dir.endswith('3-Mirror') or dir.endswith('4-Color') or dir.endswith('5-Multiple-Objects') or dir.endswith('Reasoning-231115'):
                print(os.path.join(root, dir))
                sub_path = os.path.join(root, dir)
                benckmark_understanding_scenes_dir.append(sub_path)

    # understanding and reasoning scenes...
    metrics_size = 256
    for sub_dir in benckmark_understanding_scenes_dir:
        # 1.
        if sub_dir.endswith("1-Left-Right") == True:
            PSNR_1, LPIPS_1, MSE_1, SSIM_1, CLIP_score_1 = [], [], [], [], []
            original_image_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            mask_image_list = sorted(glob.glob(f'{sub_dir}/*_mask.jpg'))
            edited_image_dir = args.edited_image_understanding_dir + "/LeftRight_metrics_1"
            edited_image_list = sorted(glob.glob(f'{edited_image_dir}/*.png'))
            with open(sub_dir + "/Left_Right_text.txt", 'r') as f:
                prompt = f.readlines()
            for idx, img_path in enumerate(original_image_list):
                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_image_list[idx]).convert("L")
                edited_image = Image.open(edited_image_list[idx]).convert("RGB")
                text_prompt = prompt[idx]

                # rstrip() remove space
                instruction = text_prompt.split("CLIP: ")[0].rstrip()
                CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')

                # read images
                original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

                # crop edited images -> find mask boundary
                mask_array = np.array(mask_image)
                y, x = np.where(mask_array)
                top = np.min(y)
                bottom = np.max(y)
                left = np.min(x)
                right = np.max(x)
                cropped_edited_image = edited_image.crop((left, top, right, bottom))
                cropped_edited_image = np.array(cropped_edited_image)
                cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to("cuda")
                CLIP_score = clip_metric_calculator(cropped_edited_image, CLIP_text)
                CLIP_score = CLIP_score.cpu().item()

                # process mask
                mask_image = np.asarray(mask_image, dtype=np.int64) / 255
                mask_image = 1 - mask_image
                mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

                # 1.1. PSNR
                psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.2. LPIPS
                lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.3. MSE
                mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.4. SSIM
                ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                PSNR_1.append(psnr_unedit_part)
                LPIPS_1.append(lpips_unedit_part)
                MSE_1.append(mse_unedit_part)
                SSIM_1.append(ssim_unedit_part)
                CLIP_score_1.append(CLIP_score)
                # print((idx + 1), round(psnr_unedit_part, 3), round(ssim_unedit_part, 3), round(lpips_unedit_part, 3), round(CLIP_score, 3))

            # show all results
            PSNR_1_mean = sum(PSNR_1) / len(PSNR_1)
            LPIPS_1_mean = sum(LPIPS_1) / len(LPIPS_1)
            MSE_1_mean = sum(MSE_1) / len(MSE_1)
            SSIM_1_mean = sum(SSIM_1) / len(SSIM_1)
            CLIP_score_1_mean = sum(CLIP_score_1) / len(CLIP_score_1)
            print('1-Left-Right metrics finished...', round(PSNR_1_mean, 3), round(LPIPS_1_mean, 3), round(MSE_1_mean, 3), round(SSIM_1_mean, 3), round(CLIP_score_1_mean, 3))

        # 2.
        if sub_dir.endswith("2-Relative-Size") == True:
            PSNR_2, LPIPS_2, MSE_2, SSIM_2, CLIP_score_2 = [], [], [], [], []
            original_image_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            mask_image_list = sorted(glob.glob(f'{sub_dir}/*_mask.jpg'))
            edited_image_dir = args.edited_image_understanding_dir + "/RelativeSize_metrics_2"
            edited_image_list = sorted(glob.glob(f'{edited_image_dir}/*.png'))
            with open(sub_dir + "/Size_text.txt", 'r') as f:
                prompt = f.readlines()
            for idx, img_path in enumerate(original_image_list):
                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_image_list[idx]).convert("L")
                edited_image = Image.open(edited_image_list[idx]).convert("RGB")
                text_prompt = prompt[idx]

                # rstrip() remove space
                instruction = text_prompt.split("CLIP: ")[0].rstrip()
                CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')

                # read images
                original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

                # crop edited images -> find mask boundary
                mask_array = np.array(mask_image)
                y, x = np.where(mask_array)
                top = np.min(y)
                bottom = np.max(y)
                left = np.min(x)
                right = np.max(x)
                cropped_edited_image = edited_image.crop((left, top, right, bottom))
                cropped_edited_image = np.array(cropped_edited_image)
                cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to("cuda")
                CLIP_score = clip_metric_calculator(cropped_edited_image, CLIP_text)
                CLIP_score = CLIP_score.cpu().item()

                # process mask
                mask_image = np.asarray(mask_image, dtype=np.int64) / 255
                mask_image = 1 - mask_image
                mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

                # 1.1. PSNR
                psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.2. LPIPS
                lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.3. MSE
                mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.4. SSIM
                ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                PSNR_2.append(psnr_unedit_part)
                LPIPS_2.append(lpips_unedit_part)
                MSE_2.append(mse_unedit_part)
                SSIM_2.append(ssim_unedit_part)
                CLIP_score_2.append(CLIP_score)
                # print((idx + 1), round(psnr_unedit_part, 3), round(ssim_unedit_part, 3), round(lpips_unedit_part, 3), round(CLIP_score, 3))

            # show all results
            PSNR_2_mean = sum(PSNR_2) / len(PSNR_2)
            LPIPS_2_mean = sum(LPIPS_2) / len(LPIPS_2)
            MSE_2_mean = sum(MSE_2) / len(MSE_2)
            SSIM_2_mean = sum(SSIM_2) / len(SSIM_2)
            CLIP_score_2_mean = sum(CLIP_score_2) / len(CLIP_score_2)
            print('2-Relative-Size metrics finished...', round(PSNR_2_mean, 3), round(LPIPS_2_mean, 3), round(MSE_2_mean, 3), round(SSIM_2_mean, 3), round(CLIP_score_2_mean, 3))

        # 3.
        if sub_dir.endswith("3-Mirror") == True:
            PSNR_3, LPIPS_3, MSE_3, SSIM_3, CLIP_score_3 = [], [], [], [], []
            original_image_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            mask_image_list = sorted(glob.glob(f'{sub_dir}/*_mask.jpg'))
            edited_image_dir = args.edited_image_understanding_dir + "/Mirror_metrics_3"
            edited_image_list = sorted(glob.glob(f'{edited_image_dir}/*.png'))
            with open(sub_dir + "/Mirror_text.txt", 'r') as f:
                prompt = f.readlines()
            for idx, img_path in enumerate(original_image_list):
                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_image_list[idx]).convert("L")
                edited_image = Image.open(edited_image_list[idx]).convert("RGB")
                text_prompt = prompt[idx]

                # rstrip() remove space
                instruction = text_prompt.split("CLIP: ")[0].rstrip()
                CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')

                # read images
                original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

                # crop edited images -> find mask boundary
                mask_array = np.array(mask_image)
                y, x = np.where(mask_array)
                top = np.min(y)
                bottom = np.max(y)
                left = np.min(x)
                right = np.max(x)
                cropped_edited_image = edited_image.crop((left, top, right, bottom))
                cropped_edited_image = np.array(cropped_edited_image)
                cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to("cuda")
                CLIP_score = clip_metric_calculator(cropped_edited_image, CLIP_text)
                CLIP_score = CLIP_score.cpu().item()

                # process mask
                mask_image = np.asarray(mask_image, dtype=np.int64) / 255
                mask_image = 1 - mask_image
                mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

                # 1.1. PSNR
                psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.2. LPIPS
                lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.3. MSE
                mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.4. SSIM
                ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                PSNR_3.append(psnr_unedit_part)
                LPIPS_3.append(lpips_unedit_part)
                MSE_3.append(mse_unedit_part)
                SSIM_3.append(ssim_unedit_part)
                CLIP_score_3.append(CLIP_score)
                # print((idx + 1), round(psnr_unedit_part, 3), round(ssim_unedit_part, 3), round(lpips_unedit_part, 3), round(CLIP_score, 3))

            # show all results
            PSNR_3_mean = sum(PSNR_3) / len(PSNR_3)
            LPIPS_3_mean = sum(LPIPS_3) / len(LPIPS_3)
            MSE_3_mean = sum(MSE_3) / len(MSE_3)
            SSIM_3_mean = sum(SSIM_3) / len(SSIM_3)
            CLIP_score_3_mean = sum(CLIP_score_3) / len(CLIP_score_3)
            print('3-Mirror metrics finished...', round(PSNR_3_mean, 3), round(LPIPS_3_mean, 3), round(MSE_3_mean, 3), round(SSIM_3_mean, 3), round(CLIP_score_3_mean, 3))

        # 4.
        if sub_dir.endswith("4-Color") == True:
            PSNR_4, LPIPS_4, MSE_4, SSIM_4, CLIP_score_4 = [], [], [], [], []
            original_image_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            mask_image_list = sorted(glob.glob(f'{sub_dir}/*_mask.jpg'))
            edited_image_dir = args.edited_image_understanding_dir + "/Color_metrics_4"
            edited_image_list = sorted(glob.glob(f'{edited_image_dir}/*.png'))
            with open(sub_dir + "/Color_text.txt", 'r') as f:
                prompt = f.readlines()
            for idx, img_path in enumerate(original_image_list):
                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_image_list[idx]).convert("L")
                edited_image = Image.open(edited_image_list[idx]).convert("RGB")
                text_prompt = prompt[idx]

                # rstrip() remove space
                instruction = text_prompt.split("CLIP: ")[0].rstrip()
                CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')

                # read images
                original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

                # crop edited images -> find mask boundary
                mask_array = np.array(mask_image)
                y, x = np.where(mask_array)
                top = np.min(y)
                bottom = np.max(y)
                left = np.min(x)
                right = np.max(x)
                cropped_edited_image = edited_image.crop((left, top, right, bottom))
                cropped_edited_image = np.array(cropped_edited_image)
                cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to("cuda")
                CLIP_score = clip_metric_calculator(cropped_edited_image, CLIP_text)
                CLIP_score = CLIP_score.cpu().item()

                # process mask
                mask_image = np.asarray(mask_image, dtype=np.int64) / 255
                mask_image = 1 - mask_image
                mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

                # 1.1. PSNR
                psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.2. LPIPS
                lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.3. MSE
                mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.4. SSIM
                ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                PSNR_4.append(psnr_unedit_part)
                LPIPS_4.append(lpips_unedit_part)
                MSE_4.append(mse_unedit_part)
                SSIM_4.append(ssim_unedit_part)
                CLIP_score_4.append(CLIP_score)
                # print((idx + 1), round(psnr_unedit_part, 3), round(ssim_unedit_part, 3), round(lpips_unedit_part, 3), round(CLIP_score, 3))

            # show all results
            PSNR_4_mean = sum(PSNR_4) / len(PSNR_4)
            LPIPS_4_mean = sum(LPIPS_4) / len(LPIPS_4)
            MSE_4_mean = sum(MSE_4) / len(MSE_4)
            SSIM_4_mean = sum(SSIM_4) / len(SSIM_4)
            CLIP_score_4_mean = sum(CLIP_score_4) / len(CLIP_score_4)
            print('4-Color finished...', round(PSNR_4_mean, 3), round(LPIPS_4_mean, 3), round(MSE_4_mean, 3), round(SSIM_4_mean, 3), round(CLIP_score_4_mean, 3))

        # 5.
        if sub_dir.endswith("5-Multiple-Objects") == True:
            PSNR_5, LPIPS_5, MSE_5, SSIM_5, CLIP_score_5 = [], [], [], [], []
            original_image_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            mask_image_list = sorted(glob.glob(f'{sub_dir}/*_mask.jpg'))
            edited_image_dir = args.edited_image_understanding_dir + "/MultipleObjects_metrics_5"
            edited_image_list = sorted(glob.glob(f'{edited_image_dir}/*.png'))
            with open(sub_dir + "/MultipleObjects_text.txt", 'r') as f:
                prompt = f.readlines()
            for idx, img_path in enumerate(original_image_list):
                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_image_list[idx]).convert("L")
                edited_image = Image.open(edited_image_list[idx]).convert("RGB")
                text_prompt = prompt[idx]

                # rstrip() remove space
                instruction = text_prompt.split("CLIP: ")[0].rstrip()
                CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')

                # read images
                original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

                # crop edited images -> find mask boundary
                mask_array = np.array(mask_image)
                y, x = np.where(mask_array)
                top = np.min(y)
                bottom = np.max(y)
                left = np.min(x)
                right = np.max(x)
                cropped_edited_image = edited_image.crop((left, top, right, bottom))
                cropped_edited_image = np.array(cropped_edited_image)
                cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to("cuda")
                CLIP_score = clip_metric_calculator(cropped_edited_image, CLIP_text)
                CLIP_score = CLIP_score.cpu().item()

                # process mask
                mask_image = np.asarray(mask_image, dtype=np.int64) / 255
                mask_image = 1 - mask_image
                mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

                # 1.1. PSNR
                psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.2. LPIPS
                lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.3. MSE
                mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.4. SSIM
                ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                PSNR_5.append(psnr_unedit_part)
                LPIPS_5.append(lpips_unedit_part)
                MSE_5.append(mse_unedit_part)
                SSIM_5.append(ssim_unedit_part)
                CLIP_score_5.append(CLIP_score)
                # print((idx + 1), round(psnr_unedit_part, 3), round(ssim_unedit_part, 3), round(lpips_unedit_part, 3), round(CLIP_score, 3))

            # show all results
            PSNR_5_mean = sum(PSNR_5) / len(PSNR_5)
            LPIPS_5_mean = sum(LPIPS_5) / len(LPIPS_5)
            MSE_5_mean = sum(MSE_5) / len(MSE_5)
            SSIM_5_mean = sum(SSIM_5) / len(SSIM_5)
            CLIP_score_5_mean = sum(CLIP_score_5) / len(CLIP_score_5)
            print('5-Multiple-Objects finished...', round(PSNR_5_mean, 3), round(LPIPS_5_mean, 3), round(MSE_5_mean, 3), round(SSIM_5_mean, 3), round(CLIP_score_5_mean, 3))

        # 6.
        if sub_dir.endswith("6-Reasoning") == True:
            PSNR_reason, LPIPS_reason, MSE_reason, SSIM_reason, CLIP_score_reason = [], [], [], [], []
            original_image_list = sorted(glob.glob(f'{sub_dir}/*.png'))
            mask_image_list = sorted(glob.glob(f'{sub_dir}/*_mask.jpg'))
            edited_image_dir = args.edited_image_reasoning_dir + "/Reasoning_metrics_1"
            edited_image_list = sorted(glob.glob(f'{edited_image_dir}/*.png'))
            with open(sub_dir + "/Reason_test.txt", 'r') as f:
                prompt = f.readlines()
            for idx, img_path in enumerate(original_image_list):
                original_image = Image.open(img_path).convert("RGB")
                mask_image = Image.open(mask_image_list[idx]).convert("L")
                edited_image = Image.open(edited_image_list[idx]).convert("RGB")
                text_prompt = prompt[idx]

                # rstrip() remove space
                instruction = text_prompt.split("CLIP: ")[0].rstrip()
                CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')

                # read images
                original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
                mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)

                # crop edited images -> find mask boundary
                mask_array = np.array(mask_image)
                y, x = np.where(mask_array)
                top = np.min(y)
                bottom = np.max(y)
                left = np.min(x)
                right = np.max(x)
                cropped_edited_image = edited_image.crop((left, top, right, bottom))
                cropped_edited_image = np.array(cropped_edited_image)
                cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to("cuda")
                CLIP_score = clip_metric_calculator(cropped_edited_image, CLIP_text)
                CLIP_score = CLIP_score.cpu().item()

                # process mask
                mask_image = np.asarray(mask_image, dtype=np.int64) / 255
                mask_image = 1 - mask_image
                mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

                # 1.1. PSNR
                psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.2. LPIPS
                lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.3. MSE
                mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                # 1.4. SSIM
                ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                                     mask_pred=mask_image, mask_gt=mask_image)
                PSNR_reason.append(psnr_unedit_part)
                LPIPS_reason.append(lpips_unedit_part)
                MSE_reason.append(mse_unedit_part)
                SSIM_reason.append(ssim_unedit_part)
                CLIP_score_reason.append(CLIP_score)

            # show all results
            PSNR_reason_mean = sum(PSNR_reason) / len(PSNR_reason)
            LPIPS_reason_mean = sum(LPIPS_reason) / len(LPIPS_reason)
            MSE_reason_mean = sum(MSE_reason) / len(MSE_reason)
            SSIM_reason_mean = sum(SSIM_reason) / len(SSIM_reason)
            CLIP_score_reason_mean = sum(CLIP_score_reason) / len(CLIP_score_reason)
            print('6-Reasoning finished...', round(PSNR_reason_mean, 3), round(LPIPS_reason_mean, 3), round(MSE_reason_mean, 3), round(SSIM_reason_mean, 3), round(CLIP_score_reason_mean, 3))

    # Final metrics
    print('Understanding total PSNR:', (sum(PSNR_1) + sum(PSNR_2) + sum(PSNR_3) + sum(PSNR_4) + sum(PSNR_5)) / (len(PSNR_1) + len(PSNR_2) + len(PSNR_3) + len(PSNR_4) + len(PSNR_5)))
    print('Understanding total LPIPS:', (sum(LPIPS_1) + sum(LPIPS_2) + sum(LPIPS_3) + sum(LPIPS_4) + sum(LPIPS_5)) / (len(LPIPS_1) + len(LPIPS_2) + len(LPIPS_3) + len(LPIPS_4) + len(LPIPS_5)))
    print('Understanding total MSE:', (sum(MSE_1) + sum(MSE_2) + sum(MSE_3) + sum(MSE_4) + sum(MSE_5)) / (len(MSE_1) + len(MSE_2) + len(MSE_3) + len(MSE_4) + len(MSE_5)))
    print('Understanding total SSIM:', (sum(SSIM_1) + sum(SSIM_2) + sum(SSIM_3) + sum(SSIM_4) + sum(SSIM_5)) / (len(SSIM_1) + len(SSIM_2) + len(SSIM_3) + len(SSIM_4) + len(SSIM_5)))
    print('Understanding total CLIP:', (sum(CLIP_score_1) + sum(CLIP_score_2) + sum(CLIP_score_3) + sum(CLIP_score_4) + sum(CLIP_score_5)) / (len(CLIP_score_1) + len(CLIP_score_2) + len(CLIP_score_3) + len(CLIP_score_4) + len(CLIP_score_5)))


########################################################################################################################################################################
"""
# 1. SmartEdit-7B
python test/metrics_evaluation.py --edited_image_understanding_dir "./checkpoints/SmartEdit-7B/Understand-15000" --edited_image_reasoning_dir "./checkpoints/SmartEdit-7B/Reason-15000"
# 2. SmartEdit-13B
python test/metrics_evaluation.py --edited_image_understanding_dir "./checkpoints/SmartEdit-13B/Understand-15000" --edited_image_reasoning_dir "./checkpoints/SmartEdit-13B/Reason-15000"
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 1. benckmark_dir
    parser.add_argument(
        "--benckmark_dir",
        type=str,
        default="./dataset/ReasonEdit_benckmark_dir",
    )
    # 2. edited_image_understanding_dir
    parser.add_argument(
        '--edited_image_understanding_dir',
        type=str,
        default=None,
    )
    # 3. edited_image_reasoning_dir
    parser.add_argument(
        '--edited_image_reasoning_dir',
        type=str,
        default=None,
    )

    args = parser.parse_args()
    main_metrics()
