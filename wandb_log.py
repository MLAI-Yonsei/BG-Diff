import wandb
from datetime import datetime

wandb.login()
# 프로젝트와 엔터티 이름 설정
project_name = "ppg_regressor"
entity_name = "ppg-diffusion"

# WandB API 객체 생성
api = wandb.Api()

# 프로젝트의 모든 실행을 불러옴
runs = api.runs(f"{entity_name}/{project_name}")
timestamp_threshold = datetime.fromisoformat('2023-09-29T00:00:00+00:00').timestamp()
group_name = "train_param_sweep_gal"
best_model_selection = "best_gal_val_loss_tot"
# 각 실행에서 특정 두 값의 합을 계산하고 저장
for target_fold in [0,1,2,3,4]:
    passed = 0
    run_sums = []
    print(f'target train fold : {target_fold}')
    for run in runs:
        if isinstance(run, wandb.apis.public.Run):            
            train_fold = run.config.get("train_fold", None)   
            group = run.group
            diffuse_step = run.config.get("diffusion_time_steps", None)
            created_at = datetime.fromisoformat(run.created_at.replace("Z", "+00:00")).timestamp()  # created_at을 Unix timestamp로 변환
       
            if train_fold == target_fold and created_at > timestamp_threshold and diffuse_step == 2000 and group == group_name:
                if run.state == "running":
                    passed += 1
                    continue
                # metric1 = run.summary.get("best_val_loss_sbp", 0)
                # metric2 = run.summary.get("best_val_loss_dbp", 0)
                # metric1 = run.summary.get("val_best_group_0_mae_sbp", 999)
                # metric2 = run.summary.get("val_best_group_0_mae_dbp", 999)
                metric = run.summary.get(best_model_selection, 999)
                # run_sums.append((run, metric1 + metric2))
                run_sums.append((run, metric))

    # 합이 가장 낮은 순으로 정렬
    sorted_runs = sorted(run_sums, key=lambda x: x[1])

    # 가장 작은 값을 가진 run을 가져옴
    best_run = sorted_runs[0][0]

    # 해당 run의 다른 값을 출력
    if passed != 0:
        print(f"!! some runs did not completed !!")
    print(f"Best Run ID: {best_run.name}")
    print(f"Created At: {best_run.created_at}")
    print(f'Best model selection metric : {best_model_selection}')
    print("Other values:")
    for key, value in best_run.summary.items():
        # if key in ['best_train_loss_sbp','best_train_loss_dbp','best_val_loss_sbp', 'best_val_loss_dbp']:
        #     print(f"{key}: {value:.4f}")
        # print(f"{key}: {value}")
        if 'best' in key:
            print(f"{key}: {value:.4f}")
    print("\n")
    # # 해당 run의 config 값을 테이블 형태로 출력합니다.
    print("\nBest Run Config:")
    print("-" * 30)
    print(f"{'Config Key':<15} | {'Config Value':<15}")
    print("-" * 30)
    for key, value in best_run.config.items():
        print(f"{key:<15} | {value:<15}")