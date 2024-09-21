#!/bin/bash

CUDA_VISIBLE_DEVICES=0 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 1 --reg_model_sel val --target_group 0 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=1 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 1 --reg_model_sel val --target_group 2 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=2 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 1 --reg_model_sel val --target_group 3 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=3 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 1 --reg_model_sel val --target_group 4 --t_scheduling "uniform" &

CUDA_VISIBLE_DEVICES=4 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 2 --reg_model_sel val --target_group 0 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=5 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 2 --reg_model_sel val --target_group 2 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=6 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 2 --reg_model_sel val --target_group 3 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=7 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 2 --reg_model_sel val --target_group 4 --t_scheduling "uniform" &

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림

CUDA_VISIBLE_DEVICES=0 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 3 --reg_model_sel val --target_group 0 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=1 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 3 --reg_model_sel val --target_group 2 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=2 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 3 --reg_model_sel val --target_group 3 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=3 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 3 --reg_model_sel val --target_group 4 --t_scheduling "uniform" &

CUDA_VISIBLE_DEVICES=4 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 4 --reg_model_sel val --target_group 0 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=5 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 4 --reg_model_sel val --target_group 2 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=6 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 4 --reg_model_sel val --target_group 3 --t_scheduling "uniform" &
CUDA_VISIBLE_DEVICES=7 /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python main.py --regressor_epoch 2000 --diffusion_time_steps 2000 --sample_only --train_fold 4 --reg_model_sel val --target_group 4 --t_scheduling "uniform" &
