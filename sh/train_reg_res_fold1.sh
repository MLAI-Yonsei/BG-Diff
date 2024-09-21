#!/bin/bash
GPU_IDS=(1 2 5 6 7)  # 사용할 GPU ID 리스트
IDX=0
run_group="_fold1_regenerate"
for benchmark in "ppgbp"
do
for train_fold in 1
do
for train_epochs in 1000
do
for diffusion_time_steps in 2000
do
for loss in "group_average_loss"
do
for eta_min in 0.001 #0.01
do
for init_lr in 0.0001
do
for weight_decay in 1e-1
do
for do_rate in 0.7
do
for t_scheduling in "train-step" #"loss-second-moment"
do
for final_layers in 3 # 4 8 12
do
for auxilary_classification in "--auxilary_classification" #""
do
for is_se in "--is_se"
do
# 현재 GPU ID 선택
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} /mlainas/teang1995/anaconda3/envs/PPG/bin/python reg_resnet.py \
--run_group ${run_group} \
--train_epochs ${train_epochs} \
--diffusion_time_steps ${diffusion_time_steps} \
--T_max ${train_epochs} \
--eta_min ${eta_min} \
--init_lr ${init_lr} \
--train_fold ${train_fold} \
--weight_decay ${weight_decay} \
--do_rate ${do_rate} \
--loss ${loss} \
--disable_g \
--final_layers ${final_layers} \
--t_scheduling ${t_scheduling} \
--benchmark ${benchmark} \
--ignore_wandb \
${is_se} \
${auxilary_classification} & # 백그라운드에서 실행

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
done
done
done
done
done
done
done

wait  # 마지막 배치의 모든 작업이 완료될 때까지 기다림
