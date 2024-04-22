""" bash scripts/SmartEdit_7b.sh """

# train SmartEdit-7b
wandb disabled
export WANDB_DISABLED=true
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28458 train/DS_SmartEdit_train.py \
    --max_steps 15000 \
    --model_name_or_path ./checkpoints/vicuna-7b-v1-1 \
    --LLaVA_00001 "./checkpoints/LLaVA-7B-v1/pytorch_model-00001-of-00002.bin" \
    --LLaVA_00002 "./checkpoints/LLaVA-7B-v1/pytorch_model-00002-of-00002.bin" \
    --LLaVA_model_path "./checkpoints/LLaVA-7B-v1" \
    --sd_qformer_version "v1.1-7b" \
    --pretrained_LLaMA "./checkpoints/stage2_MLLMSD_7b/LLM-5000/adapter_model.bin" \
    --pretrained_model "./checkpoints/stage2_MLLMSD_7b/embeddings_qformer/checkpoint-5000_embeddings_qformer.bin" \
    --pretrained_unet "./checkpoints/stage2_MLLMSD_7b/unet-5000/adapter_model.bin" \
    --bf16 True \
    --tf32 True \
    --output_dir "./checkpoints/SmartEdit_7b_ckpt" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 5000 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --ddp_find_unused_parameters True \
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
