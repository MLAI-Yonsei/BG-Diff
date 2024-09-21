for target in 1 0.8 0.6 0.4 0.2 0
    do
    CUDA_VISBLE_DEVICES=1 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py \
    --device cuda \
    --visualize \
    --ignore_wandb \
    --train_num_steps 16 \
    --sample_only \
    --target_label $target 
    done