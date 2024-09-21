import os
import torch
import pickle
import numpy as np
import pandas as pd
import tabulate
from collections import defaultdict, Counter
import scipy.io
from torch import nn
from matplotlib import pyplot as plt
import wandb
import random

def visualize(root, target_label, min_value, max_value):
    plot_root = os.path.join(root, f'plot_results_target_{target_label}')
    os.makedirs(plot_root, exist_ok=True)
    sample_path = os.path.join(root, f'sample_{target_label}.pkl')
    try:
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)
            samples = sample.squeeze(1)
            samples = samples.detach().cpu().numpy()
            for i, sample in enumerate(samples):
                plt_path = f'{plot_root}/{i}.png'
                plt.plot(sample)
                plt.ylim(min_value, max_value)  # Set the y-axis limits
                plt.savefig(plt_path)
                plt.cla()
    except Exception as e:
        print(e)


def sample_sbp_dbp(target_group, batch_size, mode = "sample_each"):
    size = batch_size if mode == "sample_each" else 1
    if target_group == 0:#"hypo":
        sbp = np.random.uniform(80, 90, size=size)
        dbp = np.random.uniform(40, 60, size=size)
    elif target_group == 1:#"normal":
        sbp = np.random.uniform(90, 120, size=size)
        dbp = np.random.uniform(60, 80, size=size)
    elif target_group == 2:#"prehyper":
        sbp = np.random.uniform(120, 140, size=size)
        dbp = np.random.uniform(80, 90, size=size)
    elif target_group == 3:#"hyper2":
        sbp = np.random.uniform(140, 180, size=size)
        dbp = np.random.uniform(90, 120, size=size)
    elif target_group == 4:#"crisis":
        # sbp = np.random.uniform(180, 200, size=size)
        # dbp = np.random.uniform(120, 130, size=size)
        raise ValueError("[crisis] group is deprecated")
    else:
        raise ValueError("Invalid target group")
    print(f"Target ({target_group}) Batch ({sbp.shape[0]}) : [sbp {sbp.mean().item():.2f}, dbp {dbp.mean().item():.2f}]")
    if mode == "same":
        return torch.tensor([[sbp, dbp]] * batch_size)
    return torch.tensor(np.array([sbp, dbp]).T)
    
def assign_group_label(sbp, dbp, mode, r=10):
    if (sbp <= 80) or (sbp >= 180) or (dbp >= 120) or (dbp <= 40): # add crisis
        return None # eliminate from dataset
    if mode=="etri":
        if (140 <= sbp) or (90 <= dbp):
            return 3 #"hyper2"
        elif (120 <= sbp) or (80 <= dbp):
            return 2 #"prehyper"
        elif (90 <= sbp) or (60 <= dbp):
            return 1 #"normal"
        elif (80 <= sbp) or (40 <= dbp):
            return 0 #"hypo"
    elif mode=="same":
        sbp_label = (sbp - 80) // 10
        dbp_label = (dbp - 40) // 10
        return (sbp_label, dbp_label)
    
def same_to_group(same):   
    new_group=[]
    for a in same:
        sbp=a[0]; dbp=a[1]
        if (6 <= sbp) or (5 <= dbp):
            new_group.append(3) #"hyper2"
        elif (4 <= sbp) or (4 <= dbp):
            new_group.append(2) #"prehyper"
        elif (1 <= sbp) or (2 <= dbp):
            new_group.append(1) #"normal"
        elif (0 <= sbp) or (0 <= dbp):
            new_group.append(0) #"hypo"
    return torch.tensor(new_group)




def print_group_counts(group_labels):
    group_names = {0: "undefined", 1: "hypo", 2: "normal", 3: "prehyper", 4: "hyper2", 5: "crisis"}
    group_counts = defaultdict(int, Counter(group_labels))
    print("Group       | Count")
    print("------------|-------")
    for group_id, group_name in group_names.items():
        count = group_counts[group_id]
        print(f"{group_name.ljust(12)}| {count}")

def get_data(sampling_method='first_k',
             num_samples=5,
             data_root='./',
             benchmark='bcg',
             train_fold = 0,
             group_mode="same",
             d500_idx=0):
    
    if benchmark=='bcg' or benchmark=='ppgbp' or benchmark=='sensors':
        if train_fold == 0:
            fold_nums = [0,1,4]
            val_fold = 3
        elif train_fold == 1:
            fold_nums = [1,2,4]
            val_fold = 0
        elif train_fold == 2:
            fold_nums = [2,3,4]
            val_fold = 1
        elif train_fold == 3:
            fold_nums = [0,2,3]
            val_fold = 4
        elif train_fold == 4:
            fold_nums = [0,1,3]
            val_fold = 2
        elif train_fold == -1:
            fold_nums = [0,1,2,3,4]
            val_fold = 0
        data = {"train": {},
                "valid": {} }
        # train
        ppgs_tensor, spdps_tensor, group_labels = fold_data(fold_nums, group_mode, dataset=benchmark)
        data['train']['ppg'] = ppgs_tensor
        data['train']['spdp'] = spdps_tensor
        data['train']['group_label'] = group_labels
        # valid
        ppgs_tensor, spdps_tensor, group_labels = fold_data([val_fold], group_mode, dataset=benchmark)
        data['valid']['ppg'] = ppgs_tensor
        data['valid']['spdp'] = spdps_tensor
        data['valid']['group_label'] = group_labels
        return data

    #--------------------------------------------------------------------------------------
    
    elif benchmark == 'etri':
        assert sampling_method in ['first_k']

        col_names = ['time', 'PPG', 'abp']
        sample_list = []

        if sampling_method == 'first_k':
            for patient_id in sorted(os.listdir(data_root)):
                patient_dir = os.path.join(data_root, patient_id)

                for signal_name in sorted(os.listdir(patient_dir))[:num_samples]:
                    sample = pd.read_csv(f'{os.path.join(patient_dir, signal_name)}', names=col_names)
                    sample_tensor = torch.tensor(sample['PPG'].values)
                    sample_list.append(sample_tensor)

        len_seq = len(sample_list)
        training_seq = torch.stack(sample_list).unsqueeze(1).half()
        
        return training_seq, len_seq
    
    #--------------------------------------------------------------------------------------
    
    elif benchmark == 'd500':
        return torch.load(f'/data1/bubble3jh/ppg/data/six_ch/d500_sliced_{d500_idx}.pth')
    
def fold_data(fold_nums, group_mode, dataset='bcg'):
    ppgs, spdps, group_labels = [], [], []
    count = 0
    for fold_num in fold_nums:
        ppg, spdp = load_fold_np(fold_num=fold_num, root=f'/mlainas/ETRI_2023/splitted/{dataset}_dataset')
        ppg = torch.permute(ppg, (1,0,2))
        sbp = spdp[0, :, 0].squeeze()
        dbp = spdp[1, :, 0].squeeze()
        # Assign group labels based on sbp and dbp values
        for s, d, p, sp in zip(sbp, dbp, ppg, spdp.squeeze().T):
            group_label = assign_group_label(s, d, group_mode)
            if group_label is not None:
                ppgs.append(p)
                spdps.append(sp)
                group_labels.append(group_label)
            else:
                count = count + 1
    # print_group_counts(group_labels)
    print(f"{count} datas eliminated.\n")
    return torch.cat(ppgs, dim=0).unsqueeze(1).half(), torch.stack(spdps, dim=0).float(), torch.tensor(group_labels)

def load_fold_np(fold_num, root='/mlainas/ETRI_2023/splitted/bcg_dataset'):
    if os.path.exists(f'{root}/signal_fold_{fold_num}_ppg.npy'):
        return torch.from_numpy(np.load(f'{root}/signal_fold_{fold_num}_ppg.npy')), torch.from_numpy(np.load(f'{root}/signal_fold_{fold_num}_spdp.npy'))
    else:
        mat = scipy.io.loadmat(f'{root}/signal_fold_{fold_num}.mat')
        data_dict = {key: mat[key] for key in ['SP', 'DP']}
        spdp_numpy_data = np.array([value for value in data_dict.values()])
        np.save(f'{root}/signal_fold_{fold_num}_spdp.npy', spdp_numpy_data)
        data_dict = {key: mat[key] for key in ['signal']}
        ppg_numpy_data = np.array([value for value in data_dict.values()])
        np.save(f'{root}/signal_fold_{fold_num}_ppg.npy', ppg_numpy_data)   
        return torch.from_numpy(ppg_numpy_data), torch.from_numpy(spdp_numpy_data)

def get_sample_batch_size(data, target_group):
    group_label = same_to_group(data['train']['group_label'])
    unique_elements, counts = torch.unique(group_label, return_counts=True)

    max_count = torch.max(counts).item()
    
    target_count = counts[unique_elements == target_group].item() if target_group in unique_elements else 0
    print(f"max count : {max_count}, target count : {target_count}")
    
    return max_count - target_count

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    
def get_reg_modelpath(args):
    if args.benchmark == "bcg":
        if args.reg_selection_loss == "worst":
            if args.train_fold == 0:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 2
                args.final_layers = 4
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.g_pos = "front"
                args.concat_label_mlp = False
                args.weight_decay = 0.0001

            elif args.train_fold == 1:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 4
                args.final_layers = 4
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.001

            elif args.train_fold == 2:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 2
                args.final_layers = 4
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.0001

            elif args.train_fold == 3:
                args.t_scheduling = "uniform"
                args.g_mlp_layers = 4
                args.final_layers = 4
                args.eta_min = 0.01
                args.init_lr = 0.001
                args.g_pos = "front"
                args.concat_label_mlp = False
                args.weight_decay = 0.0001

            elif args.train_fold == 4:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 4
                args.final_layers = 4
                args.eta_min = 0.01
                args.init_lr = 0.00001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.001


        # best erm val
        if args.reg_selection_loss == "erm":
            if args.train_fold == 0:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 2
                args.final_layers = 12
                args.eta_min = 0.01
                args.init_lr = 0.00001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.001

            elif args.train_fold == 1:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 4
                args.final_layers = 4
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.001

            elif args.train_fold == 2:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 2
                args.final_layers = 4
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.0001

            elif args.train_fold == 3:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 2
                args.final_layers = 12
                args.eta_min = 0.001
                args.init_lr = 0.00001
                args.g_pos = "rear"
                args.concat_label_mlp = False
                args.weight_decay = 0.0001

            elif args.train_fold == 4:
                args.t_scheduling = "train-step"
                args.g_mlp_layers = 2
                args.final_layers = 3
                args.eta_min = 0.01
                args.init_lr = 0.0001
                args.g_pos = "rear"
                args.concat_label_mlp = True
                args.weight_decay = 0.1


        # best gal val

        if args.reg_selection_loss == "gal":
            if args.train_fold == 0:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.1
                args.do_rate = 0.8
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = False

            elif args.train_fold == 1:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.00001
                args.weight_decay = 0.1
                args.do_rate = 0.6
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True

            elif args.train_fold == 2:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.01
                args.do_rate = 0.6
                args.loss = 'ERM'
                args.is_se = False
                args.auxilary_classification = False

            elif args.train_fold == 3:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.00001
                args.weight_decay = 0.01
                args.do_rate = 0.8
                args.loss = 'group_average_loss'
                args.is_se = True
                args.auxilary_classification = True

            elif args.train_fold == 4:
                args.t_scheduling = "train-step"
                args.final_layers = 3
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.weight_decay = 0.1
                args.do_rate = 0.7
                args.loss = 'ERM'
                args.is_se = False
                args.auxilary_classification = True

        # group average loss trained model load
        if args.reg_selection_dataset == "val":
            # model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/{args.t_scheduling}_epoch_{args.regressor_epoch}_diffuse_{args.diffusion_time_steps}_wd_{args.weight_decay}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_g_{args.g_mlp_layers}_layer_g_pos{args.g_pos}_cat_{str(args.concat_label_mlp)}_resnet_{args.reg_train_loss}_{args.reg_selection_loss}.pt" # test best model로 변경
            model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/train-step_epoch_2000_diffuse_2000_wd_{args.weight_decay}_do_rate_{args.do_rate}_loss_{args.loss}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_no_group_label_timelayer_MLP_is_se_{str(args.is_se)}_auxilary_classifcation_{str(args.auxilary_classification)}_last_resnet_{args.reg_selection_loss}.pt"
        elif args.reg_selection_dataset == "last":
            model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/{args.t_scheduling}_epoch_{args.regressor_epoch}_diffuse_{args.diffusion_time_steps}_wd_{args.weight_decay}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_g_{args.g_mlp_layers}_layer_g_pos{args.g_pos}_cat_{str(args.concat_label_mlp)}_last_resnet_{args.reg_selection_loss}.pt" # test best model로 변경
    
# --------------------------------- ppgbp -----------------------------------------------
    elif args.benchmark == "ppgbp":
        if args.reg_selection_loss == "gal":
            if args.train_fold == 0:
                args.t_scheduling = "train-step"
                args.final_layers = 3
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.001
                args.do_rate = 0.7
                args.loss = 'group_average_loss'
                args.is_se = True
                args.auxilary_classification = True

            elif args.train_fold == 1:
                args.t_scheduling = "train-step"
                args.final_layers = 3
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.weight_decay = 0.1
                args.do_rate = 0.7
                args.loss = 'group_average_loss'
                args.is_se = True
                args.auxilary_classification = True

            elif args.train_fold == 2:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.1
                args.do_rate = 0.8
                args.loss = 'group_average_loss'
                args.is_se = True
                args.auxilary_classification = True

            elif args.train_fold == 3:
                args.t_scheduling = "train-step"
                args.final_layers = 3
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.01
                args.do_rate = 0.8
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True

            elif args.train_fold == 4:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 1e-05
                args.weight_decay = 0.001
                args.do_rate = 0.7
                args.loss = 'group_average_loss'
                args.is_se = True
                args.auxilary_classification = True
        else:
            raise(f"model selection must be GAL in {args.benchmark} dataset")
                                                                        #    train-step_diffuse_2000_wd_0.001_do_rate_0.7_loss_group_average_loss_eta_0.001_lr_0.001_3_final_no_group_label_timelayer_MLP_is_se_True_ppgbp_resnet_group_average_loss_gal.pt
        model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/train-step_diffuse_2000_wd_{args.weight_decay}_do_rate_{args.do_rate}_loss_{args.loss}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_no_group_label_timelayer_MLP_is_se_{str(args.is_se)}_{args.benchmark}_resnet_{args.loss}_{args.reg_selection_loss}.pt"

# --------------------------------- sensors -----------------------------------------------
    elif args.benchmark == "sensors":
        if args.reg_selection_loss == "gal":
            if args.train_fold == 0:
                args.t_scheduling = "train-step"
                args.final_layers = 3
                args.eta_min = 0.001
                args.init_lr = 0.0001
                args.weight_decay = 0.01
                args.do_rate = 0.8
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True

            elif args.train_fold == 1:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.01
                args.do_rate = 0.6
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True

            elif args.train_fold == 2:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 0.001
                args.weight_decay = 0.1
                args.do_rate = 0.8
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True

            elif args.train_fold == 3:
                args.t_scheduling = "train-step"
                args.final_layers = 3
                args.eta_min = 0.001
                args.init_lr = 1e-05
                args.weight_decay = 0.01
                args.do_rate = 0.6
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True

            elif args.train_fold == 4:
                args.t_scheduling = "train-step"
                args.final_layers = 2
                args.eta_min = 0.001
                args.init_lr = 1e-05
                args.weight_decay = 0.001
                args.do_rate = 0.7
                args.loss = 'group_average_loss'
                args.is_se = False
                args.auxilary_classification = True
        else:
            raise(f"model selection must be GAL in {args.benchmark} dataset")
        model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/train-step_diffuse_2000_wd_{args.weight_decay}_do_rate_{args.do_rate}_loss_{args.loss}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_no_group_label_timelayer_MLP_is_se_{str(args.is_se)}_{args.benchmark}_resnet_{args.loss}_{args.reg_selection_loss}.pt"
    return model_path, args

# Batch-wise MAE 계산 함수
def calculate_batch_mae(model_output, ground_truth, dataset, group, mae_sbp_lists, mae_dbp_lists, overall_mae_sbp_list, overall_mae_dbp_list):
    model_output = dataset.undo_normalization_label(model_output)
    ground_truth = dataset.undo_normalization_label(ground_truth)
    group = same_to_group(group)

    # Calculate overall MAE for all data (ignoring group labels)
    overall_loss_batch = torch.abs(model_output - ground_truth)
    overall_mae_sbp_list.extend(overall_loss_batch[:, 0].detach().cpu().numpy().tolist())
    overall_mae_dbp_list.extend(overall_loss_batch[:, 1].detach().cpu().numpy().tolist())

    # Loop through each unique group value (0 to 4)
    for g in torch.unique(group):
        # Get the indices for the current group
        indices = torch.where(group == g)[0]
        
        # Extract model outputs and ground truth values for the current group
        model_output_group = model_output[indices]
        ground_truth_group = ground_truth[indices]
        
        # Calculate MAE for the current group
        loss_batch = torch.abs(model_output_group - ground_truth_group)
        mae_sbp = loss_batch[:, 0].detach().cpu().numpy().tolist()
        mae_dbp = loss_batch[:, 1].detach().cpu().numpy().tolist()
        
        # Save the MAE values to the lists
        if g.item() not in mae_sbp_lists:
            mae_sbp_lists[g.item()] = []
            mae_dbp_lists[g.item()] = []

        mae_sbp_lists[g.item()].extend(mae_sbp)
        mae_dbp_lists[g.item()].extend(mae_dbp)
    
    # Save model_output and ground_truth to CSV
    df_output = pd.DataFrame(model_output.detach().cpu().numpy(), columns=['Output_SBP', 'Output_DBP'])
    df_output.to_csv('./check_outs/model_output.csv', index=False)
    
    df_truth = pd.DataFrame(ground_truth.detach().cpu().numpy(), columns=['True_SBP', 'True_DBP'])
    df_truth.to_csv('./check_outs/ground_truth.csv', index=False)
    return overall_mae_sbp_list , overall_mae_dbp_list , mae_sbp_lists , mae_dbp_lists 

# Global Metrics Logging 함수
def log_global_metrics(args, overall_mae_sbp_list, overall_mae_dbp_list, mae_sbp_lists, mae_dbp_lists, phase):
    # Calculate and log overall MAE for SBP and DBP
    overall_mae_sbp = sum(overall_mae_sbp_list) / len(overall_mae_sbp_list)
    overall_mae_dbp = sum(overall_mae_dbp_list) / len(overall_mae_dbp_list)
    
    # print(f"{phase} Overall Mean MAE_SBP: {overall_mae_sbp}")
    # print(f"{phase} Overall Mean MAE_DBP: {overall_mae_dbp}")

    # Calculate and log MAE for each group
    for group, values in mae_sbp_lists.items():
        group_mae_sbp = sum(values) / len(values)
        # print(f"{phase} Mean MAE_SBP for group {group}: {group_mae_sbp}")

    for group, values in mae_dbp_lists.items():
        group_mae_dbp = sum(values) / len(values)
        # print(f"{phase} Mean MAE_DBP for group {group}: {group_mae_dbp}")

    # Logging to Weights and Biases (wandb) if it's not ignored
    if not args.ignore_wandb:
        wandb_metrics = {
            f"{phase}_overall_mae_sbp": overall_mae_sbp,
            f"{phase}_overall_mae_dbp": overall_mae_dbp
        }
        
        for group in mae_sbp_lists.keys():
            wandb_metrics[f"{phase}_group_{group}_mae_sbp"] = sum(mae_sbp_lists[group]) / len(mae_sbp_lists[group])
            wandb_metrics[f"{phase}_group_{group}_mae_dbp"] = sum(mae_dbp_lists[group]) / len(mae_dbp_lists[group])
            
        wandb.log(wandb_metrics)
    return overall_mae_sbp, overall_mae_dbp, mae_sbp_lists, mae_dbp_lists

def set_seed(random_seed=1000):
    '''
    Set Seed for Reproduction
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    