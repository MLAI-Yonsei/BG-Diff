#!/bin/bash

GPU_IDS=(2 3 4 5 6)  # 사용할 GPU ID 리스트
IDX=0
for train_fold in 4
do
  for train_epochs in 2000
  do
    for diffusion_time_steps in 2000
    do
        for eta_min in 0.001 0.01
        do
          for init_lr in 0.0001 0.00001 0.001 0.01
          do
            # 현재 GPU ID 선택
            CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} /mlainas/bubble3jh/anaconda3/envs/ppg/bin/python reg.py \
            --train_epochs ${train_epochs} \
            --diffusion_time_steps ${diffusion_time_steps} \
            --T_max ${train_epochs} \
            --eta_min ${eta_min} \
            --init_lr ${init_lr} \
            --train_fold ${train_fold} \
            --t_scheduling "uniform" &  # 백그라운드에서 실행

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

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림