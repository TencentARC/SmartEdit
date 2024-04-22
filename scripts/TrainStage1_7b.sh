"""
wget https://huggingface.co/lmsys/vicuna-7b-v1.1/resolve/main/*
bash scripts/TrainStage1_7b.sh
"""

# CC12M + llava1.1-7b
torchrun --nproc_per_node=8 --master_port=20001 fastchat/train/TrainStage1.py \
    --max_steps 150000 \
    --model_name_or_path ./checkpoints/vicuna-7b-v1-1 \
    --LLaVA_model_v1_1_7b_path ./checkpoints/LLaVA-7B-v1 \
    --data_path ./dataset/cc12m.tsv \
    --template_data_path ./data/conv_template_cap_to_img.txt \
    --bf16 True \
    --output_dir ./checkpoints/stage1_CC12M_alignment_7b \
    --num_new_tokens 32 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 256 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --LLaVA_version "v1.1-7b" \
    --report_to "none"
