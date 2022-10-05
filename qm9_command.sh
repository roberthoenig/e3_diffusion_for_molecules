python main_qm9.py
    --n_epochs 1
    --exp_name edm_qm9
    --n_stability_samples 1000
    --diffusion_noise_schedule polynomial_2
    --diffusion_noise_precision 1e-5 
    --diffusion_steps 2 
    --diffusion_loss_type l2 
    --batch_size 4 
    --nf 16
    --n_layers 2
    --lr 1e-4 
    --normalize_factors [1,4,10] 
    --test_epochs 0 
    --ema_decay 0.9999
    --no_wandb