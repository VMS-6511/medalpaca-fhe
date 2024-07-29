conda activate medalpaca

python medalpaca/train.py \
    --model /data/healthy-ml/scratch/vinithms/projects/medalpaca-fhe/llama-7b-hf \
    --data_path medical_meadow_small.json \
    --output_dir 'output_test' \
    --train_in_8bit True \
    --use_lora True \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --global_batch_size 128 \
    --per_device_batch_size 8 \