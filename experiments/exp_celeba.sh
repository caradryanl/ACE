accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/121 --output_dir lora_repo/training/121 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1135 --output_dir lora_repo/training/1135 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1422 --output_dir lora_repo/training/1422 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1499 --output_dir lora_repo/training/1499 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/celeba/1657 --output_dir lora_repo/training/1657 --class_data_dir data/class/attack --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1

accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/121 --output_dir lora_repo/outputs/121 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1135 --output_dir lora_repo/outputs/1135 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1422 --output_dir lora_repo/outputs/1422 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1499 --output_dir lora_repo/outputs/1499 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/training/1657 --output_dir lora_repo/outputs/1657 --class_data_dir data/class/training --output_format safe --instance_prompt "a photo of a sks person" --class_prompt "a photo of a person" --mixed_precision bf16

python i2i.py -m t2i -lp lora_repo/outputs/121/lora_weight.safetensors -op outputs/121 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/outputs/1135/lora_weight.safetensors -op outputs/1135 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/outputs/1422/lora_weight.safetensors -op outputs/1422 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/outputs/1499/lora_weight.safetensors -op outputs/1499 -spi 100 -p "a photo of a sks person"
python i2i.py -m t2i -lp lora_repo/outputs/1657/lora_weight.safetensors -op outputs/1657 -spi 100 -p "a photo of a sks person"


