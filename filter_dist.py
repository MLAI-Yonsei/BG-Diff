import torch
from denoising_diffusion_pytorch.model import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import paths
from utils import *
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
import pickle
from prettytable import PrettyTable
from sklearn.decomposition import PCA

def main(args):
    # 주어진 데이터에 대해 위의 함수를 사용하여 처리
    boundary = 3
    tar_group = 1
    n_components=10
    pca = PCA(n_components=n_components)
    data = get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark='bcg',
                                    train_fold=args.train_fold)
    path=f'/mlainas/ETRI_2023/sampling_results/fold_0/seed_1000_sampling_method_first_k_num_samples_5-diffusion_time_steps_2000-train_num_steps_32_bcg_guided_train_lr_8e-05_reg_set_val_tr_group_average_loss_sel_erm/sample_group_average_loss_{tar_group}.pkl'
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    gen_ppg = sample['sampled_seq']
    gen_y = sample['y']
    gen_group = torch.full((gen_ppg.shape[0],),tar_group)
    gen_dataset = Dataset1D(gen_ppg, label=gen_y, groups=gen_group, normalize=True)

    train_dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'], normalize=True)
    train_statistics = extract_statistics(train_dataset, boundary=boundary)
    train_pca_statistics = extract_pca_statistics(train_dataset, pca, boundary=boundary)

    print("for validation set\n")
    val_dataset = Dataset1D(data['valid']['ppg'], label=data['valid']['spdp'], groups=data['valid']['group_label'], normalize=True)
    filtered_val_data = filter_dataset_based_on_statistics(val_dataset, train_statistics)
    filtered_val_data_pca = filter_dataset_based_on_statistics(val_dataset, train_pca_statistics, pca=pca, is_pca=True)

    print("for generated set")
    filtered_val_data = filter_dataset_based_on_statistics(gen_dataset, train_statistics)
    filtered_val_data_pca = filter_dataset_based_on_statistics(gen_dataset, train_pca_statistics, pca=pca, is_pca=True)

def extract_statistics(dataset, boundary=2, group_labels=[0, 1, 2, 3], device='cuda'):
    """
    주어진 데이터셋에서 그룹별 평균과 분산을 추출합니다.
    """
    all_data = torch.cat([data for data, _, _ in dataset]).to(device)
    all_groups = torch.stack([g for _, _, g in dataset]).to(device)
    grouped_data = get_grouped_data(all_data, all_groups, group_labels)

    statistics = {}
    for group_label in group_labels:
        statistics[group_label] = get_statistics(grouped_data[group_label], boundary)

    return statistics

def extract_pca_statistics(dataset, pca, boundary=2, group_labels=[0, 1, 2, 3], device='cuda'):
    """
    주어진 데이터셋의 PCA 변환 버전에서 그룹별 평균과 분산을 추출합니다.
    """
    all_data = torch.cat([data for data, _, _ in dataset]).to(device)
    pca.fit(all_data.cpu().numpy())
    all_data = pca.transform(all_data.cpu().numpy())
    all_data = torch.tensor(all_data).to(device)
    
    all_groups = torch.stack([g for _, _, g in dataset]).to(device)
    grouped_data = get_grouped_data(all_data, all_groups, group_labels)

    statistics = {}
    for group_label in group_labels:
        statistics[group_label] = get_statistics(grouped_data[group_label], boundary)

    return statistics

def get_statistics(data, boundary=2):
    """
    주어진 데이터로 그룹별 평균과 분산을 계산합니다.
    """
    mean = data.mean(dim=0)
    var = data.var(dim=0)
    lower_bound = mean - boundary * torch.sqrt(var)
    upper_bound = mean + boundary * torch.sqrt(var)
    return mean, var, lower_bound, upper_bound

def filter_dataset_based_on_statistics(dataset, statistics, pca=None, group_labels=[0, 1, 2, 3], is_pca=False, device='cuda'):
    """
    주어진 데이터셋을 주어진 통계치를 기준으로 필터링합니다.
    """
    all_data = torch.cat([data for data, _, _ in dataset]).to(device)
    all_groups = torch.stack([g for _, _, g in dataset]).to(device)

    if is_pca:
        print('pca transformed')
        all_data = pca.fit_transform(all_data.cpu().numpy())
        all_data = torch.tensor(all_data).to(device)
    else:
        print('raw data')
    grouped_data = get_grouped_data(all_data, all_groups, group_labels)

    filtered_data = {}
    for group_label in group_labels:
        mean, var, lower_bound, upper_bound = statistics[group_label]
        current_data = grouped_data[group_label]
        filtered_data[group_label] = filter_data(current_data, lower_bound, upper_bound)
        print(f"Group {group_label}: {current_data.size(0)} to {filtered_data[group_label].size(0)}")

    return filtered_data

def filter_data(data, lower_bound, upper_bound):
    """
    주어진 데이터와 통계치로 필터링된 데이터를 반환합니다.
    """
    is_within_bounds = (data >= lower_bound) & (data <= upper_bound)
    rows_to_keep = is_within_bounds.all(dim=1)
    return data[rows_to_keep]

def get_grouped_data(data, groups, group_labels=[0, 1, 2, 3]):
    """
    주어진 데이터와 그룹 레이블로 그룹별 데이터를 반환합니다.
    """
    if len(groups.shape) == 2:
        groups = same_to_group(groups)
    grouped_data = {}
    for group_label in group_labels:
        mask = (groups == group_label)
        grouped_data[group_label] = data[mask]
    return grouped_data

if __name__ == '__main__':

    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="generate ppg with regressor guidance")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--run_group", type=str, default='default')
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

    ## DATA ----------------------------------------------------
    parser.add_argument("--seq_length", type=int, default=625)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--min_max", action='store_false',
        help = "Min-Max normalize data (Default : True)")
    parser.add_argument("--benchmark", type=str, default='bcg')
    parser.add_argument("--train_fold", type=int, default=0)

    ## Model ---------------------------------------------------
    parser.add_argument("--load_checkpoint", action='store_true',
        help = "Resume model training (Default : False)")
    parser.add_argument("--concat_label_mlp", action='store_true',
        help = "concat label mlp (Default : False)")
    parser.add_argument("--final_layers", type=int, default=3)
    parser.add_argument("--g_mlp_layers", type=int, default=3)
    parser.add_argument("--n_block", type=int, default=8)
    parser.add_argument("--g_pos", type=str, default='rear',  choices=["rear", "front"])
    
    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--init_bias", type=float, default=0.2)
    parser.add_argument("--final_bias", type=float, default=1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--loss", type=str, default='group_average_loss',  choices=["ERM", "group_average_loss"])
    parser.add_argument("--t_scheduling", type=str, default="uniform",  choices=["loss-second-moment", "uniform", "train-step"])
    parser.add_argument("--T_max", type=int, default=2000)  
    parser.add_argument("--eta_min", type=float, default=0)  

    ## Sampling ------------------------------------------------
    parser.add_argument("--target_group", type=int, default=1, choices=[0,1,2,3,4], 
                        help="0(hyp0) 1(normal) 2(perhyper) 3(hyper2) 4(crisis) (Default : 1 (normal))")
    args = parser.parse_args()

    table = PrettyTable()
    table.field_names = ["Parameter", "Value"]
    for arg in vars(args):
        table.add_row([arg.replace('_', ' ').title(), getattr(args, arg)])

    print("Hyperparameters Configuration:")
    print(table)
    main(args)

