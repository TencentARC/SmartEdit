""" bash scripts/MLLMSD_7b.sh """

# train MLLM-7b + SD
wandb disabled
export WANDB_DISABLED=true
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train/DS_MLLMSD11_train.py \
    --max_steps 5000 \
    --model_name_or_path ./checkpoints/vicuna-7b-v1-1 \
    --LLaVA_00001 "./checkpoints/LLaVA-7B-v1/pytorch_model-00001-of-00002.bin" \
    --LLaVA_00002 "./checkpoints/LLaVA-7B-v1/pytorch_model-00002-of-00002.bin" \
    --LLaVA_model_path "./checkpoints/LLaVA-7B-v1" \
    --sd_qformer_version "v1.1-7b" \
    --unet_ckpt "./checkpoints/InstructDiffusion_diffusers/unet/diffusion_pytorch_model.bin" \
    --bf16 True \
    --tf32 True \
    --output_dir ./checkpoints/stage2_MLLMSD_7b \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --lr_scheduler_type 'cosine' \
    --weight_decay 0. \
    --warmup_ratio 0.001 \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --ddp_find_unused_parameters True \
    --SD_QFormer_conversation_33tokens "./checkpoints/stage1_CC12M_alignment_7b/embeddings_qformer/checkpoint-150000_embeddings_qformer.bin" \
    --InstructPix2PixDataset_path "./dataset/InstructPix2PixCLIPFiltered_HF" \
    --MagicBrushDataset_path "./dataset/MagicBrush_HF" \
    --LLaVADataset_data_path "./dataset/LLaVA/llava_instruct_150k.json" \
    --LLaVADataset_image_folder "./dataset/coco/train2017" \
    --refcoco_path "./dataset/refcoco" \
    --grefcoco_path "./dataset/grefcoco" \
    --coco_image_path "./dataset/coco" \
    --COCOStuff_mask_path "./dataset/cocostuff" \
    --ReasoningEditingDataset_path "./dataset/SyntheticData/SyntheticData_info_new.json" \
    --ReasoningSegmentationDataset_json_path "./dataset/reason_seg/train" \
    --ReasoningSegmentationDataset_image_path "./dataset/reason_seg/train" \
    --ReasoningSegmentationDataset_binary_mask_path "./dataset/reason_seg/train_binary_mask" \
    --deepspeed scripts/zero2_mixed.json \
