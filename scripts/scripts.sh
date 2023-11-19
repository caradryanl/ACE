accelerate launch attacks/ita.py --low_vram_mode \
    --instance_data_dir_for_adversarial data/training \
    --output_dir output/ \
    --class_data_dir data/class \
    --instance_prompt "a photo of a sks person, high quality, masterpiece" \
    --class_prompt "a painting, high quality, masterpiece" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --checkpointing_iterations 1

accelerate launch attacks/aspl_lora.py \
    --instance_data_dir_for_train data/training \
    --instance_data_dir_for_adversarial data/training \
    --output_dir output/ \
    --class_data_dir data/class \
    --instance_prompt "a photo of a sks person, high quality, masterpiece" \
    --class_prompt "a painting, high quality, masterpiece" \
    --mixed_precision bf16 \
    --max_train_steps 5 \
    --checkpointing_iterations 1

accelerate launch benchmarks/advdm.py \
    --instance_data_dir_for_adversarial data/training \
    --output_dir output/ \
    --instance_prompt "a photo of a sks person, high quality, masterpiece" \
    --mixed_precision bf16 \
    --checkpointing_iterations 1

accelerate launch benchmarks/photoguard.py \
    --input_dir data/training \
    --output_dir output/ \

