accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/wikiart/Fauvism_henri-matisse --output_dir lora_repo/wikiart/training/Fauvism_henri-matisse --class_data_dir data/class/wikiart/attack --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/wikiart/Impressionism_claude-monet --output_dir lora_repo/wikiart/training/Impressionism_claude-monet --class_data_dir data/class/wikiart/attack --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/wikiart/Pointillism_paul-signac --output_dir lora_repo/wikiart/training/Pointillism_paul-signac --class_data_dir data/class/wikiart/attack --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/wikiart/Post_Impressionism-van-gogh --output_dir lora_repo/wikiart/training/Post_Impressionism-van-gogh --class_data_dir data/class/wikiart/attack --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1
accelerate launch attacks/aspl_lora_m.py --instance_data_dir_for_adversarial data/wikiart/Rococo_canaletto --output_dir lora_repo/wikiart/training/Rococo_canaletto --class_data_dir data/class/wikiart/attack --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16 --max_train_steps 5 --checkpointing_iterations 1

accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/wikiart/training/Fauvism_henri-matisse/noise-ckpt/5 --output_dir lora_repo/wikiart/outputs/Fauvism_henri-matisse --class_data_dir data/class/wikiart/training --output_format safe --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/wikiart/training/Impressionism_claude-monet/noise-ckpt/5 --output_dir lora_repo/wikiart/outputs/Impressionism_claude-monet --class_data_dir data/class/wikiart/training --output_format safe --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/wikiart/training/Pointillism_paul-signac/noise-ckpt/5 --output_dir lora_repo/wikiart/outputs/Pointillism_paul-signac --class_data_dir data/class/wikiart/training --output_format safe --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/wikiart/training/Post_Impressionism-van-gogh/noise-ckpt/5 --output_dir lora_repo/wikiart/outputs/Post_Impressionism-van-gogh --class_data_dir data/class/wikiart/training --output_format safe --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16
accelerate launch train_lora_dreambooth.py --instance_data_dir lora_repo/wikiart/training/Rococo_canaletto/noise-ckpt/5 --output_dir lora_repo/wikiart/outputs/Rococo_canaletto --class_data_dir data/class/wikiart/training --output_format safe --instance_prompt "a photo of a sks painting" --class_prompt "a photo of a painting" --mixed_precision bf16


python i2i.py -m t2i -lp lora_repo/wikiart/outputs/Fauvism_henri-matisse/lora_weight.safetensors -op outputs/wikiart/ace_t2i/Fauvism_henri-matisse -spi 100 -p "a photo of a sks painting"
python i2i.py -m t2i -lp lora_repo/wikiart/outputs/Impressionism_claude-monet/lora_weight.safetensors -op outputs/wikiart/ace_t2i/Impressionism_claude-monet -spi 100 -p "a photo of a sks painting"
python i2i.py -m t2i -lp lora_repo/wikiart/outputs/Pointillism_paul-signac/lora_weight.safetensors -op outputs/wikiart/ace_t2i/Pointillism_paul-signac -spi 100 -p "a photo of a sks painting"
python i2i.py -m t2i -lp lora_repo/wikiart/outputs/Post_Impressionism-van-gogh/lora_weight.safetensors -op outputs/wikiart/ace_t2i/Post_Impressionism-van-gogh -spi 100 -p "a photo of a sks painting"
python i2i.py -m t2i -lp lora_repo/wikiart/outputs/Rococo_canaletto/lora_weight.safetensors -op outputs/wikiart/ace_t2i/Rococo_canaletto -spi 100 -p "a photo of a sks painting"

python i2i.py -m i2i -ip lora_repo/wikiart/training/Fauvism_henri-matisse/noise-ckpt/5 -op outputs/wikiart/ace_i2i/Fauvism_henri-matisse -spi 1 -p "a photo of a sks painting"
python i2i.py -m i2i -ip lora_repo/wikiart/training/Impressionism_claude-monet/noise-ckpt/5 -op outputs/wikiart/ace_i2i/Impressionism_claude-monet -spi 1 -p "a photo of a sks painting"
python i2i.py -m i2i -ip lora_repo/wikiart/training/Pointillism_paul-signac/noise-ckpt/5 -op outputs/wikiart/ace_i2i/Pointillism_paul-signac -spi 1 -p "a photo of a sks painting"
python i2i.py -m i2i -ip lora_repo/wikiart/training/Post_Impressionism-van-gogh/noise-ckpt/5 -op outputs/wikiart/ace_i2i/Post_Impressionism-van-gogh -spi 1 -p "a photo of a sks painting"
python i2i.py -m i2i -ip lora_repo/wikiart/training/Rococo_canaletto/noise-ckpt/5 -op outputs/wikiart/ace_i2i/Rococo_canaletto -spi 1 -p "a photo of a sks painting"

python evaluation.py -m CLIPT2I --path outputs/wikiart/ace_t2i/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python evaluation.py -m CLIPT2I --path outputs/wikiart/ace_t2i/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python evaluation.py -m CLIPT2I --path outputs/wikiart/ace_t2i/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python evaluation.py -m CLIPT2I --path outputs/wikiart/ace_t2i/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python evaluation.py -m CLIPT2I --path outputs/wikiart/ace_t2i/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting

python evaluation.py -m MSSSIM --path outputs/wikiart/ace_i2i/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python evaluation.py -m MSSSIM --path outputs/wikiart/ace_i2i/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python evaluation.py -m MSSSIM --path outputs/wikiart/ace_i2i/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python evaluation.py -m MSSSIM --path outputs/wikiart/ace_i2i/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python evaluation.py -m MSSSIM --path outputs/wikiart/ace_i2i/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting

python evaluation.py -m CLIPI2I --path outputs/wikiart/ace_i2i/Fauvism_henri-matisse --std_path data/wikiart/Fauvism_henri-matisse -c painting
python evaluation.py -m CLIPI2I --path outputs/wikiart/ace_i2i/Impressionism_claude-monet --std_path data/wikiart/Impressionism_claude-monet -c painting
python evaluation.py -m CLIPI2I --path outputs/wikiart/ace_i2i/Pointillism_paul-signac --std_path data/wikiart/Pointillism_paul-signac -c painting
python evaluation.py -m CLIPI2I --path outputs/wikiart/ace_i2i/Post_Impressionism-van-gogh --std_path data/wikiart/Post_Impressionism-van-gogh -c painting
python evaluation.py -m CLIPI2I --path outputs/wikiart/ace_i2i/Rococo_canaletto --std_path data/wikiart/Rococo_canaletto -c painting



