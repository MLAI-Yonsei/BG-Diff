import time
import torch
from denoising_diffusion_pytorch.model import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import paths
from utils import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from denoising_diffusion_pytorch.resample import create_named_schedule_sampler
import wandb
import numpy as np
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
from prettytable import PrettyTable


def cycle(dl):
    while True:
        for data in dl:
            yield data

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda")
    batch_size = args.train_batch_size
    diffuse_time_step = args.diffusion_time_steps 
    epochs=args.train_epochs
    time_layer = 'Linear' if args.time_linear  else 'MLP'
    if not args.ignore_wandb:
        run_group = args.benchmark + args.run_group
        # import pdb; pdb.set_trace()
        wandb.init(entity="ppg-diffusion" ,project="ppg_regressor", config=args, group=run_group)
        wandb.run.name=f"fold_{args.train_fold}_{args.t_scheduling}_epoch_{epochs}_diffuse_{diffuse_time_step}_wd_{args.weight_decay}_eta_{args.eta_min}_lr_{args.init_lr}_nblock_{args.n_block}_{args.final_layers}-layer-clf_timelayer_{time_layer}-auxilary_classifcation_{args.auxilary_classification}"
    data = get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark=args.benchmark,
                                    train_fold=args.train_fold)
    
    schedule_sampler = create_named_schedule_sampler(
        args.t_scheduling, diffuse_time_step=diffuse_time_step,total_epochs=epochs, init_bias=args.init_bias, final_bias=args.final_bias
    ) 
    
    if args.disable_g :
        model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/{args.t_scheduling}_diffuse_{diffuse_time_step}_wd_{args.weight_decay}_do_rate_{args.do_rate}_loss_{args.loss}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_no_group_label_timelayer_{time_layer}_is_se_{args.is_se}"
    else:
        model_path = f"/mlainas/ETRI_2023/reg_model/fold_{args.train_fold}/{args.t_scheduling}_diffuse_{diffuse_time_step}_wd_{args.weight_decay}_eta_{args.eta_min}_lr_{args.init_lr}_{args.final_layers}_final_g_{args.g_mlp_layers}_layer_g_pos{args.g_pos}_cat_{args.concat_label_mlp}_timelayer_{time_layer}"

    tr_dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'] ,normalize=True)
    val_dataset = Dataset1D(data['valid']['ppg'], label=data['valid']['spdp'], groups=data['valid']['group_label'] ,normalize=True)
    val_every = int(len(tr_dataset) / args.train_batch_size); print(f'eval every {val_every} epochs')
    tr_dl = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)
    tr_dl = cycle(tr_dl)
    
    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 0)
    # regressor = Unet1DEncoder(
    #         dim = args.seq_length,
    #         dim_mults = (1, 2, 4, 8),
    #         channels = 1
    #     ).to(device)
    if args.benchmark == "ppgbp":
        seq_length=262
    else:
        seq_length=625
    regressor = ResNet1D(output_size=2, final_layers=args.final_layers, n_block=args.n_block, is_se=args.is_se, use_bn=True, use_do=True,
                         concat_label_mlp=args.concat_label_mlp, g_pos=args.g_pos, g_mlp_layers=args.g_mlp_layers, disable_g=args.disable_g, 
                         time_linear=args.time_linear, auxilary_classification=args.auxilary_classification, do_rate=args.do_rate, seq_length=seq_length).to(device)
    optimizer = optim.AdamW(regressor.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    # 전체 validation set 크기만큼 val_t_all을 미리 샘플링
    total_val_size = len(val_dl.dataset)
    val_t_all, _ = schedule_sampler.sample(total_val_size, device)

    if args.load_checkpoint:
        checkpoint = torch.load(model_path+"_resnet.pt")
        model_state_dict = checkpoint['model_state_dict']
        optim_state_dict = checkpoint['optimizer_state_dict']

        regressor.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)

    # regressor.load_state_dict(weight)
    diffusion = GaussianDiffusion1D(
        model = regressor,
        seq_length = args.seq_length,
        timesteps = diffuse_time_step,
        objective = 'pred_v'
    ).to(device)
    
    # Initialize loss variables
    best_train_loss = float('inf')
    best_val_train_loss = float('inf')
    val_loss_sbp = 0
    val_loss_dbp = 0

    # For standard validation
    best_val_loss_sbp = float('inf')
    best_val_loss_dbp = float('inf')

    # For group average loss validation
    best_gal_val_loss_sbp = float('inf')
    best_gal_val_loss_dbp = float('inf')

    # For worst group validation
    best_worst_val_loss_sbp = float('inf')
    best_worst_val_loss_dbp = float('inf')
    
    cls_criterion = nn.CrossEntropyLoss()
    with tqdm(initial = 0, total = epochs) as pbar:
        for i in range(epochs):
            mae_sbp_lists={}; mae_dbp_lists={}; overall_mae_sbp_list=[]; overall_mae_dbp_list=[]
            # Train step
            batch, spdp, g = next(tr_dl)
            batch = batch.to(device); spdp=spdp.to(device); g=g.to(device)
            t, _ = schedule_sampler.sample(batch.size(0), device)
            batch = diffusion.q_sample(batch, t)
            optimizer.zero_grad()
            
            if args.auxilary_classification:
                out, sbp_prob, dbp_prob = regressor(batch, t, g) # cls_result.shape = batch_size * 2 * 10
                loss = F.mse_loss(out, spdp, reduction="none")
                sbp_cls_loss = cls_criterion(sbp_prob, g.long()[:, 0])
                dbp_cls_loss = cls_criterion(dbp_prob, g.long()[:, 1])
            else:
                out = regressor(batch, t, g)
                # for normal loss
                loss = F.mse_loss(out, spdp, reduction="none")
            # for group loss
            group = same_to_group(g)
            group_losses={}
            for g_i in torch.unique(group):
                indices = torch.where(group == g_i)[0]
                model_output_group = out[indices]
                ground_truth_group = spdp[indices]
                g_loss = F.mse_loss(model_output_group, ground_truth_group, reduction="mean")
                if g_i.item() not in group_losses:
                    group_losses[g_i.item()] = []
                group_losses[g_i.item()].append(g_loss)
            total_loss = 0
            for _, value in group_losses.items(): 
                total_loss += value[0]
            group_avg_loss = total_loss/len(group_losses.keys())

            overall_mae_sbp_list , overall_mae_dbp_list , mae_sbp_lists , mae_dbp_lists  = calculate_batch_mae(out, spdp, tr_dataset, g, mae_sbp_lists, mae_dbp_lists, overall_mae_sbp_list, overall_mae_dbp_list) 
            overall_mae_sbp, overall_mae_dbp, _, _ = log_global_metrics(args, overall_mae_sbp_list , overall_mae_dbp_list , mae_sbp_lists , mae_dbp_lists, "train")

            if args.t_scheduling == "loss-second-moment":
                schedule_sampler.update_with_local_losses(t,loss)
            elif args.t_scheduling == "train-step":
                schedule_sampler.set_epoch(i+1)
            loss = loss.mean()
            t_mean = t.sum().item()/len(t)
            if not args.ignore_wandb :
                wandb.log({"train_loss": loss.item(), "t_mean": t_mean})
            if args.loss == "ERM":
                if args.auxilary_classification:
                    loss = loss + sbp_cls_loss + dbp_cls_loss
                loss.backward()
            elif args.loss == "group_average_loss":
                if args.auxilary_classification:
                    group_avg_loss = group_avg_loss + sbp_cls_loss + dbp_cls_loss
                group_avg_loss.backward()
            optimizer.step()
            scheduler.step()
            # Best Train Loss Update
            # if loss < best_train_loss:
            #     print(f"best train loss updated: {loss:.4f}")
            #     best_train_loss = loss
                # torch.save({
                #     'model_state_dict': regressor.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict()
                # }, model_path+".pt")
            # Validation Step
            if (i + 1) % val_every == 0:
                val_mae_sbp_lists={}; val_mae_dbp_lists={}; val_overall_mae_sbp_list=[]; val_overall_mae_dbp_list=[]
                with torch.no_grad():
                    val_loss_sbp = 0
                    val_loss_dbp = 0
                    start_idx = 0
                    for data in val_dl:   
                        val_batch, val_spdp, val_g = data
                        val_batch = val_batch.to(device); val_spdp = val_spdp.to(device); val_g = val_g.to(device)
                   
                        val_t = val_t_all[start_idx:start_idx + val_batch.size(0)].to(device)
                        val_batch = diffusion.q_sample(val_batch, val_t)
                        if args.auxilary_classification:
                            val_out, _, _ = regressor(val_batch, val_t, val_g)
                        else:
                            val_out = regressor(val_batch, val_t, val_g)
                        # val_out = regressor(val_batch, val_t, val_g)

                        val_overall_mae_sbp_list , val_overall_mae_dbp_list , val_mae_sbp_lists , val_mae_dbp_lists  = calculate_batch_mae(val_out, val_spdp, val_dataset, val_g, val_mae_sbp_lists, val_mae_dbp_lists, val_overall_mae_sbp_list, val_overall_mae_dbp_list)
                        
                        start_idx += val_batch.size(0)
                    val_loss_sbp, val_loss_dbp, val_mae_sbp_lists, val_mae_dbp_lists = log_global_metrics(args, val_overall_mae_sbp_list, val_overall_mae_dbp_list, val_mae_sbp_lists, val_mae_dbp_lists, "valid")

                    # Group Losses Calculate
                    val_sbp_group_losses = {}
                    val_dbp_group_losses = {}
                    for group in val_mae_sbp_lists.keys():
                        if g_i.item() not in group_losses:
                            val_sbp_group_losses[g_i.item()] = []
                            val_dbp_group_losses[g_i.item()] = []
                        val_sbp_group_losses[group] = sum(val_mae_sbp_lists[group]) / len(val_mae_sbp_lists[group])
                        val_dbp_group_losses[group] = sum(val_mae_dbp_lists[group]) / len(val_mae_dbp_lists[group])
                    gal_val_loss_sbp = sum(val_sbp_group_losses.values())/len(val_sbp_group_losses.values())
                    gal_val_loss_dbp = sum(val_dbp_group_losses.values())/len(val_dbp_group_losses.values())
                    worst_val_loss_sbp = max(val_sbp_group_losses.values())
                    worst_val_loss_dbp = max(val_dbp_group_losses.values())
                    # Best Validation Loss Update
                    if val_loss_sbp+val_loss_dbp < best_val_loss_sbp+best_val_loss_dbp:
                        print(f"best erm val loss updated: sbp {val_loss_sbp:.4f} dbp {val_loss_dbp:.4f}")
                        best_val_loss_sbp = val_loss_sbp
                        best_val_loss_dbp = val_loss_dbp
                        best_erm_val_loss_tot = val_loss_dbp + val_loss_sbp
                        if not args.ignore_wandb:
                            for group in val_mae_sbp_lists.keys():
                                wandb.run.summary[f"val_erm_best_group_{group}_mae_sbp"] = val_sbp_group_losses[group]
                                wandb.run.summary[f"val_erm_best_group_{group}_mae_dbp"] = val_dbp_group_losses[group]
                                wandb.run.summary[f"val_erm_best_group_{group}_mae_tot"] = val_sbp_group_losses[group] + val_dbp_group_losses[group]
                                wandb.run.summary["best_val_epoch"] = i
                        best_val_train_loss = loss
                        # torch.save({
                        #     'model_state_dict': regressor.state_dict(),
                        #     'optimizer_state_dict': optimizer.state_dict()
                        # }, model_path+f"_{args.benchmark}_resnet_{args.loss}_erm.pt")
                        
                    if gal_val_loss_sbp+gal_val_loss_dbp < best_gal_val_loss_sbp+best_gal_val_loss_dbp:
                        print(f"best gal val loss updated: sbp {gal_val_loss_sbp:.4f} dbp {gal_val_loss_dbp:.4f}")
                        best_gal_val_loss_sbp = gal_val_loss_sbp
                        best_gal_val_loss_dbp = gal_val_loss_dbp
                        best_gal_val_loss_tot = gal_val_loss_dbp + gal_val_loss_sbp
                        if not args.ignore_wandb:
                            for group in val_mae_sbp_lists.keys():
                                wandb.run.summary[f"val_gal_best_group_{group}_mae_sbp"] = val_sbp_group_losses[group]
                                wandb.run.summary[f"val_gal_best_group_{group}_mae_dbp"] = val_dbp_group_losses[group]
                                wandb.run.summary[f"val_gal_best_group_{group}_mae_tot"] = val_sbp_group_losses[group] + val_dbp_group_losses[group]
                        torch.save({
                            'model_state_dict': regressor.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }, model_path+f"_{args.benchmark}_resnet_{args.loss}_gal.pt")
                        
                    if worst_val_loss_sbp + worst_val_loss_dbp < best_worst_val_loss_sbp + best_worst_val_loss_dbp:
                        print(f"best worst val loss updated: sbp {worst_val_loss_sbp:.4f} dbp {worst_val_loss_dbp:.4f}")
                        best_worst_val_loss_sbp = worst_val_loss_sbp
                        best_worst_val_loss_dbp = worst_val_loss_dbp
                        best_worst_val_loss_tot = worst_val_loss_dbp + worst_val_loss_sbp
                        if not args.ignore_wandb:
                            for group in val_mae_sbp_lists.keys():
                                wandb.run.summary[f"val_worst_best_group_{group}_mae_sbp"] = val_sbp_group_losses[group]
                                wandb.run.summary[f"val_worst_best_group_{group}_mae_dbp"] = val_dbp_group_losses[group]
                                wandb.run.summary[f"val_worst_best_group_{group}_mae_tot"] = val_sbp_group_losses[group] + val_dbp_group_losses[group]
                        # torch.save({
                        #     'model_state_dict': regressor.state_dict(),
                        #     'optimizer_state_dict': optimizer.state_dict()
                        # }, model_path+f"_{args.benchmark}_resnet_{args.loss}_worst.pt")
                    
            pbar.set_description(f'tr_loss: {loss:.4f} | tr_sbp_mae: {overall_mae_sbp:.4f} | tr_dbp_mae: {overall_mae_dbp:.4f} | val_sbp_mae: {val_loss_sbp:.4f} | val_dbp_mae: {val_loss_dbp:.4f} | t: {t_mean:.1f}')
            pbar.update(1)
    # torch.save({
    #     'model_state_dict': regressor.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, model_path+f"_{args.benchmark}_last_resnet_gal.pt")
    if not args.ignore_wandb:
        
        wandb.run.summary["best_train_loss"] = loss
        wandb.run.summary["best_val_train_loss"] = best_val_train_loss
        wandb.run.summary["best_val_loss_sbp"] = best_val_loss_sbp
        wandb.run.summary["best_val_loss_dbp"] = best_val_loss_dbp
        wandb.run.summary["best_erm_val_loss_tot"] = best_erm_val_loss_tot
        wandb.run.summary["best_gal_val_loss_tot"] = best_gal_val_loss_tot
        wandb.run.summary["best_worst_val_loss_tot"] = best_worst_val_loss_tot
        wandb.run.summary["best_train_loss_sbp"] = overall_mae_sbp
        wandb.run.summary["best_train_loss_dbp"] = overall_mae_dbp
    print(f'best_val_loss_sbp: {best_val_loss_sbp:.4f} | best_val_loss_dbp: {best_val_loss_dbp:.4f} | tr_mae_sbp: {overall_mae_sbp:.4f} | tr_mae_dbp: {overall_mae_dbp:.4f}')
            
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
    parser.add_argument("--disable_g", action='store_true',
        help = "disable_group_label (Default : False)")
    parser.add_argument("--final_layers", type=int, default=3)
    parser.add_argument("--g_mlp_layers", type=int, default=3)
    parser.add_argument("--n_block", type=int, default=8)
    parser.add_argument("--g_pos", type=str, default='rear',  choices=["rear", "front"])
    parser.add_argument("--time_linear", action='store_true',
        help = "use linear layer instead MLP for time embedding (Default : False)")
    parser.add_argument("--auxilary_classification", action='store_true',
        help = "using classification as auxilary task (Default : False)")
    parser.add_argument("--is_se", action='store_true',
        help = "using se architecture (Default : False)")
    parser.add_argument("--do_rate", type=float, default=0.5)

    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--init_bias", type=float, default=0.2)
    parser.add_argument("--final_bias", type=float, default=1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--loss", type=str, default='group_average_loss',  choices=["ERM", "group_average_loss"])
    parser.add_argument("--t_scheduling", type=str, default="train-step",  choices=["loss-second-moment", "uniform", "train-step"])
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

