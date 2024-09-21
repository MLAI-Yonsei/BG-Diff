#!/bin/bash

GPU_IDS=(4 5)  # 사용할 GPU ID 리스트
IDX=0
loss="group_average_loss"
run_group="train_param_sweep_gal"
for train_fold in 0 1 2 3 4
do
  # Set hyperparameters based on train_fold
  if [ $train_fold -eq 0 ]; then
    g_pos="rear"
    t_scheduling="uniform"
    final_layers=4
    g_mlp_layers=2
    concat_label_mlp="--concat_label_mlp"
  elif [ $train_fold -eq 1 ]; then
    g_pos="rear"
    t_scheduling="train-step"
    final_layers=4
    g_mlp_layers=4
    concat_label_mlp="--concat_label_mlp"
  elif [ $train_fold -eq 2 ]; then
    g_pos="rear"
    t_scheduling="train-step"
    final_layers=4
    g_mlp_layers=2
    concat_label_mlp="--concat_label_mlp"
  elif [ $train_fold -eq 3 ]; then
    g_pos="rear"
    t_scheduling="train-step"
    final_layers=12
    g_mlp_layers=2
    concat_label_mlp=""
  elif [ $train_fold -eq 4 ]; then
    g_pos="rear"
    t_scheduling="train-step"
    final_layers=4
    g_mlp_layers=4
    concat_label_mlp="--concat_label_mlp"
  fi

  for train_epochs in 2000
  do
    for diffusion_time_steps in 2000
    do
      for eta_min in 0.001 0.01
      do
        for init_lr in 0.0001 0.00001 0.001  
        do
          for weight_decay in 1e-3 1e-2 1e-4
          do
            # 현재 GPU ID 선택
            CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python reg_resnet.py \
            --run_group ${run_group} \
            --train_epochs ${train_epochs} \
            --diffusion_time_steps ${diffusion_time_steps} \
            --T_max ${train_epochs} \
            --eta_min ${eta_min} \
            --init_lr ${init_lr} \
            --n_block 8 \
            --train_fold ${train_fold} \
            --weight_decay ${weight_decay} \
            --g_pos ${g_pos} \
            --loss ${loss} \
            --final_layers ${final_layers} \
            --g_mlp_layers ${g_mlp_layers} \
            --t_scheduling ${t_scheduling} \
            ${concat_label_mlp} &  # 백그라운드에서 실행

            # GPU ID를 다음 것으로 변경
            IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))

            # 모든 GPU가 사용 중이면 기다림
            if [ $IDX -eq 0 ]; then
              wait
            fi
          done
        done
      done
    done
  done
done

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림
