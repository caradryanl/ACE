python vae_attack.py -id dataset/celeba-20-1135/ -od ./abalation/VAE/l2_w_target -m 1 --iter 100
python vae_attack.py -id dataset/celeba-20-1135/ -od ./abalation/VAE/l2_o_target -m 2 --iter 100
python vae_attack.py -id dataset/celeba-20-1135/ -od ./abalation/VAE/ssim_w_target -m 3 --iter 100
python vae_attack.py -id dataset/celeba-20-1135/ -od ./abalation/VAE/ssim_o_target -m 4 --iter 100
python vae_attack.py -id dataset/celeba-20-1135/ -od ./abalation/VAE/l2d_w_target -m 5 --iter 100
python vae_attack.py -id dataset/celeba-20-1135/ -od ./abalation/VAE/l2d_o_target -m 6 --iter 100