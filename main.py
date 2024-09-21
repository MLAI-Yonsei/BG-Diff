import time
import wandb
import argparse
import pickle
import torch
import os 
import pandas as pd
from denoising_diffusion_pytorch.model import Classifier, Regressor, Unet1DEncoder, ResNet1D
from denoising_diffusion_pytorch.cond_fn import classifier_cond_fn, regressor_cond_fn
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from utils import visualize, sample_sbp_dbp, get_data, get_sample_batch_size, get_reg_modelpath
import paths

def generate_diffusion_sequence(args, data, dataset, device, diffusion, regressor_cond_fn, regressor, sampling_dir, target_group):
    sample_batch_size = args.sample_batch_size if args.sample_batch_size is not None else get_sample_batch_size(data, target_group)
    sample_batch_size = args.gen_size
    if sample_batch_size == 0:
        os.makedirs(sampling_dir, exist_ok=True)
        with open(f'{sampling_dir}/skipped_{target_group}.pkl', 'wb') as f:
            pickle.dump([], f)  
        return
    micro_batch_size = 2048  # TODO: 적절한 micro batch size 찾기 OOM 회피 위함.
    num_loops = (sample_batch_size + micro_batch_size - 1) // micro_batch_size  

    sampled_seq_list = []; ori_y_list = []
    
    for i in range(num_loops):
        current_batch_size = min(micro_batch_size, sample_batch_size - i * micro_batch_size)
        
        ori_y = sample_sbp_dbp(target_group, current_batch_size)
        sbp_label=(ori_y[:,0]-80)//10; dbp_label=(ori_y[:,1]-40)//10
        y = dataset._min_max_normalize(ori_y, dataset.label_max, dataset.label_min).to(device)

        sampled_seq = diffusion.sample(
            batch_size=current_batch_size,
            return_all_timesteps=False,
            cond_fn=regressor_cond_fn,
            guidance_kwargs={
                "regressor": regressor,
                "y": y,
                # "g": torch.fill(torch.zeros(current_batch_size,), target_group).long().to(device),
                "g": torch.stack((sbp_label, dbp_label), dim=1).long().to(device),
                "regressor_scale": args.regressor_scale,
            }
        )
        ori_y_list.append(ori_y)
        sampled_seq_list.append(sampled_seq)
    
    try:
        # torch.cat 호출 부분
        os.makedirs(sampling_dir, exist_ok=True)
        # with open(f'{sampling_dir}/sample_{target_group}.pkl', 'wb') as f:
        #     pickle.dump(torch.cat(sampled_seq_list, dim=0), f)
        result = {
            'sampled_seq': torch.cat(sampled_seq_list, dim=0),
            'y': torch.cat(ori_y_list, dim=0)
        }
        print(f"Sampling completed for target group {target_group}\nat {sampling_dir}")
        
        with open(f'{sampling_dir}/sample_{args.reg_train_loss}_{target_group}.pkl', 'wb') as f:
            pickle.dump(result, f)
            
    except RuntimeError as e:
        # 문제가 발생했음을 알리는 파일 생성
        os.makedirs(sampling_dir, exist_ok=True)
        with open(f'{sampling_dir}/error_{target_group}.txt', 'w') as error_file:
            error_file.write(str(e))
        print(f"An error occurred with target_group {target_group}: {e}")

def main(args):
    if args.regressor_scale == 1e+8 or args.regressor_scale == 5e+8:
        return 0
    if args.device == 'cuda' and torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    device = torch.device(device_name)
    print(f"Device : {device_name}")

    train_set_root = paths.TRAINSET_ROOT
    train_setting = f'fold_{args.train_fold}/seed_{args.seed}_sampling_method_{args.sampling_method}_num_samples_{args.num_samples}-'\
                      f'diffusion_time_steps_{args.diffusion_time_steps}-train_num_steps_{args.train_num_steps}_{args.benchmark}'
    if not args.ignore_wandb:
        wandb.init(project='check_train_time',
                   entity='ppg-diffusion')
        wandb_run_name = train_setting
        wandb.run.name = wandb_run_name

    result_path = os.path.join(paths.WEIGHT_ROOT, train_setting)
    sampling_root = paths.SAMPLING_ROOT
    if not args.disable_guidance:
        train_setting = train_setting + "_guided"
    sampling_name = train_setting + f'_use_group_info_regressor_train_lr_{args.train_lr}_reg_set_{args.reg_selection_dataset}_tr_{args.reg_train_loss}_sel_{args.reg_selection_loss}_gs_{args.regressor_scale}'
    sampling_dir = os.path.join(sampling_root, sampling_name)

    #------------------------------------ Load Data --------------------------------------
    if args.benchmark == "ppgbp":
        args.seq_length=262
    else:
        args.seq_length=625
    data_sampling_start = time.time()
    # ppg, label = get_data(sampling_method=args.sampling_method,
    #                              num_samples=args.num_samples,
    #                              data_root=paths.DATA_ROOT,
    #                              benchmark=args.benchmark)
    data = get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark=args.benchmark,
                                    train_fold=args.train_fold)
    # data = get_data(benchmark=args.benchmark)
    data_sampling_time = time.time() - data_sampling_start
    if not args.ignore_wandb:
        wandb.log({'n_sample': args.num_samples})
        wandb.log({'data_sampling_time': data_sampling_time})
    print(f"data sampling finished, collapsed time: {data_sampling_time:.5f}")
    os.makedirs(train_set_root, exist_ok=True)
    
    # with open(os.path.join(train_set_root, train_set_name), 'wb') as f:
    #     pickle.dump(ppg, f)
    
    tr_dataset = dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'] ,normalize=True)
    val_dataset = Dataset1D(data['valid']['ppg'], label=data['valid']['spdp'], groups=data['valid']['group_label'] ,normalize=True)

    #----------------------------------- Create Model ------------------------------------

    model = Unet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = args.channels,
        seq_length = args.seq_length
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.seq_length,
        timesteps = args.diffusion_time_steps,                           
        objective = 'pred_v',
        normalized = args.min_max,             
        denorm_func = dataset.undo_normalization if args.min_max else None
    )

    # resnet ------
    if not args.disable_guidance:
        model_path, args = get_reg_modelpath(args)
        # model_path = f'/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/train-step_epoch_2000_diffuse_2000_wd_0.0001_eta_0.0_lr_0.0001_3_final_no_group_label_resnet_group_average_loss_erm.pt'
        # regressor = ResNet1D(output_size=2, final_layers=args.final_layers).to(device)
        regressor = ResNet1D(output_size=2, final_layers=args.final_layers, n_block=8, 
                             disable_g=True, is_se=args.is_se, auxilary_classification=args.auxilary_classification,
                             do_rate=args.do_rate, seq_length=args.seq_length).to(device)
        # regressor = ResNet1D(output_size=2, disable_g=True).to(device) #  disable_g=True
        model_state_dict = torch.load(model_path)['model_state_dict']
        regressor.load_state_dict(model_state_dict)
        regressor.eval()
        print("regressor model load complete")
    #------------------------------------- Training --------------------------------------
    
    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = args.train_batch_size,
        train_lr = args.train_lr,
        train_num_steps = args.train_num_steps * len(dataset),          # total training steps
        gradient_accumulate_every = 2,                                  # gradient accumulation steps
        ema_decay = 0.995,                                              # exponential moving average decay
        amp = True,                                                     # turn on mixed precision
        results_folder = result_path
    )
    
    if not args.sample_only and not os.path.exists(f'/mlainas/ETRI_2023/weights/fold_{args.train_fold}/seed_1000_sampling_method_first_k_num_samples_5-diffusion_time_steps_2000-train_num_steps_32_sensors/model-last.pt'):
        trainer.train()
        trainer.save('last')
    else:
        trainer.load('last')

    #------------------------------------- Sampling --------------------------------------
    target_group=args.target_group
    if not args.disable_guidance:
        print("sampling with guidance")
        if args.target_group == -1:
            for target_group in [0,1,2,3]:
                generate_diffusion_sequence(args, data, dataset, device, diffusion, regressor_cond_fn, regressor, sampling_dir, target_group)
        else:
            generate_diffusion_sequence(args, data, dataset, device, diffusion, regressor_cond_fn, regressor, sampling_dir, args.target_group)
    else:
        sampled_seq = diffusion.sample(batch_size = 16) #TODO: hard coding
        os.makedirs(sampling_dir, exist_ok=True)
        with open(f'{sampling_dir}/sample_{target_group}.pkl', 'wb') as f:
            pickle.dump(sampled_seq, f)
    print(f"Data sampled at {sampling_dir}")
    #------------------------------------- Visualize --------------------------------------

    if args.visualize:
        visualize(sampling_dir, args.target_group, min_value=-0.1, max_value=0.5)

if __name__ == '__main__':

    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="generate ppg with regressor guidance")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")
    parser.add_argument("--visualize", action='store_true',
        help = "Visualize results (Default : False)")

    ## DATA ----------------------------------------------------
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=625) #32660
    parser.add_argument("--sampling_method", type=str, default='first_k')
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--min_max", action='store_false',
        help = "Min-Max normalize data (Default : True)")
    parser.add_argument("--benchmark", type=str, default='bcg')
    parser.add_argument("--train_fold", type=int, default=-1)
    parser.add_argument("--channels", type=int, default=1)

    ## Model ---------------------------------------------------
    parser.add_argument("--disable_guidance", action='store_true',
        help = "Stop using guidance (Default : False)")
    parser.add_argument("--reg_train_loss", type=str, default='group_average_loss')
    parser.add_argument("--reg_selection_dataset", type=str, default='val')
    parser.add_argument("--reg_selection_loss", type=str, default='gal',  choices=["erm", "gal", "worst"])
    

    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_num_steps", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=8e-5)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--eval_only", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## Sampling ------------------------------------------------
    parser.add_argument("--sample_only", action='store_true',
        help = "Stop Training (Default : False)")
    parser.add_argument("--sample_batch_size", default=None)
    # parser.add_argument("--target_label", type=float, default=1) # deprecated
    parser.add_argument("--target_group", type=int, default=-1, choices=[-1,0,1,2,3,4], 
                        help="-1(all) 0(hyp0) 1(normal) 2(perhyper) 3(hyper2) 4(crisis) (Default : 1 (normal))")
    parser.add_argument("--t_scheduling", type=str, default="uniform",  choices=["loss-second-moment", "uniform", "train-step"])
    parser.add_argument("--regressor_scale", type=float, default=1.0)
    parser.add_argument("--regressor_epoch", type=int, default=2000)
    parser.add_argument("--gen_size", type=int, default=8096)
    args = parser.parse_args()

    main(args)