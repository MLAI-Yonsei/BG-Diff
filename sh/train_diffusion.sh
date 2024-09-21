#!/bin/bash

GPU_IDS=(2 3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0

for train_fold in 0 1 2 3 4
do
for train_num_steps in 16 # 8 32
do
  for diffusion_time_steps in 2000
  do
        for train_lr in 8e-5 # 8e-6 8e-4 
        do
          # 현재 GPU ID 선택
          CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
          --train_num_steps ${train_num_steps} \
          --diffusion_time_steps ${diffusion_time_steps} \
          --train_lr ${train_lr} \
          --visualize \
          --train_fold ${train_fold} \
          --t_scheduling "uniform" & # 백그라운드에서 실행

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
wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림