"""
# https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered
# https://huggingface.co/datasets/osunlp/MagicBrush
python process_HF.py
"""

# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import os

# Define a generator function that loads Parquet files one by one and converts them into a dataset
def parquet_to_dataset_generator(file_paths):
    index = 0
    for file_path in file_paths:
        print('Number:', index)
        df = pd.read_parquet(file_path)
        dataset = Dataset.from_pandas(df)
        index = index + 1
        yield dataset

# InstructPix2Pix
InstructPix2Pix_file_pattern = './Datasets/InstructPix2PixCLIPFiltered_HF/*.parquet'
InstructPix2Pix_file_paths = glob.glob(InstructPix2Pix_file_pattern)

# Load Parquet files one by one using a generator function and convert them into a dataset
InstructPix2Pix_parquet_datasets = list(parquet_to_dataset_generator(InstructPix2Pix_file_paths))

# Concatenate multiple datasets using the concatenate_datasets function
InstructPix2Pix_merged_dataset = concatenate_datasets(InstructPix2Pix_parquet_datasets)
print(InstructPix2Pix_merged_dataset)
# Dataset({features: ['original_prompt', 'original_image', 'edit_prompt', 'edited_prompt', 'edited_image'], num_rows: 313010})

# Save the dataset to disk using the save_to_disk method
InstructPix2Pix_HF_path = './Datasets/InstructPix2PixCLIPFiltered_HF'
InstructPix2Pix_merged_dataset.save_to_disk(InstructPix2Pix_HF_path)

# load_from_disk
InstructPix2Pix_HF_path = load_from_disk(InstructPix2Pix_HF_path)
print(InstructPix2Pix_HF_path)

# same for MagicBrush
MagicBrush_file_pattern = './Datasets/MagicBrush_HF/train-*.parquet'
MagicBrush_file_paths = glob.glob(MagicBrush_file_pattern)
MagicBrush_parquet_datasets = list(parquet_to_dataset_generator(MagicBrush_file_paths))
MagicBrush_merged_dataset = concatenate_datasets(MagicBrush_parquet_datasets)
print(MagicBrush_merged_dataset)

# load
MagicBruth_HF_path = './Datasets/MagicBruth_HF'
MagicBrush_merged_dataset.save_to_disk(MagicBruth_HF_path)
MagicBruth_HF_path = load_from_disk(MagicBruth_HF_path)
# Dataset({features: ['img_id', 'turn_index', 'source_img', 'mask_img', 'instruction', 'target_img'], num_rows: 8807})
