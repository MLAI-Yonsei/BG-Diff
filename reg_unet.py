import time
import torch
from denoising_diffusion_pytorch.model import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import paths
from utils import get_data
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


def cycle(dl):
    while True:
        for data in dl:
            yield data

def main(args):
    device = torch.device("cuda")
    batch_size = args.train_batch_size
    diffuse_time_step = args.diffusion_time_steps 
    epochs=args.train_epochs
    data = get_data(sampling_method='first_k',
                                    num_samples=5,
                                    data_root=paths.DATA_ROOT,
                                    benchmark='bcg',
                                    train_fold=args.train_fold,
                                    group_mode="same")
    
    schedule_sampler = create_named_schedule_sampler(
        args.t_scheduling, diffuse_time_step=diffuse_time_step,total_epochs=epochs, init_bias=args.init_bias, final_bias=args.final_bias
    ) 
    
    model_path = f"your_path_here/reg_model/fold_{args.train_fold}/epoch_{epochs}_diffuse_{diffuse_time_step}_eta_{args.eta_min}_lr_{args.init_lr}"

    tr_dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'] ,normalize=True)
    val_dataset = Dataset1D(data['valid']['ppg'], label=data['valid']['spdp'], groups=data['valid']['group_label'] ,normalize=True)
    val_every = int(len(tr_dataset) / args.train_batch_size); print(f'eval every {val_every} epochs')
    tr_dl = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 0)
    tr_dl = cycle(tr_dl)

    val_dl = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 0)
    regressor = Unet1DEncoder(
            dim = args.seq_length,
            dim_mults = (1, 2, 4, 8),
            channels = 1
        ).to(device)

    optimizer = optim.Adam(regressor.parameters(), lr=args.init_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    if args.load_checkpoint:
        checkpoint = torch.load(model_path+".pt")
        model_state_dict = checkpoint['model_state_dict']
        optim_state_dict = checkpoint['optimizer_state_dict']

        regressor.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)

    # regressor.load_state_dict(weight)
    diffusion = GaussianDiffusion1D(
        model = regressor,
        seq_length = 625,
        timesteps = diffuse_time_step,
        objective = 'pred_v'
    ).to(device)
    best_train_loss = float('inf');best_val_train_loss=float('inf');best_val_loss = float('inf');val_loss=0
    with tqdm(initial = 0, total = epochs) as pbar:
        for i in range(epochs):
            # Train step
            batch, spdp, g = next(tr_dl)
            batch = batch.to(device); spdp=spdp.to(device); g=g.to(device)
            t, _ = schedule_sampler.sample(batch.size(0), device)
            batch = diffusion.q_sample(batch, t)
            optimizer.zero_grad()
            out, emb = regressor(batch, t, g)
            loss = F.mse_loss(out, spdp, reduction="none")
            if args.t_scheduling == "loss-second-moment":
                schedule_sampler.update_with_local_losses(t,loss)
            elif args.t_scheduling == "train-step":
                schedule_sampler.set_epoch(i+1)
            loss = loss.mean()
            t_mean = t.sum().item()/len(t)
            if not args.ignore_wandb :
                wandb.log({"loss": loss.item(), "t_mean": t_mean})
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Best Train Loss Update
            if loss < best_train_loss:
                print(f"best train loss updated: {loss}")
                best_train_loss = loss
                # torch.save({
                #     'model_state_dict': regressor.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict()
                # }, model_path+".pt")
            # Validation Step
            if (i + 1) % val_every == 0:
                with torch.no_grad():
                    val_loss = 0
                    val_count = 0
                    for data in val_dl:  
                        val_batch, val_spdp, val_g = data
                        val_batch = val_batch.to(device); val_spdp = val_spdp.to(device); val_g = val_g.to(device)
                        val_t, _ = schedule_sampler.sample(val_batch.size(0), device)
                        val_batch = diffusion.q_sample(val_batch, val_t)
                        val_out, val_emb = regressor(val_batch, val_t, val_g)
                        # val_loss_batch = F.mse_loss(val_out, val_spdp, reduction="none").mean()
                        val_out = val_dataset.undo_normalization_label(val_out)
                        val_spdp = val_dataset.undo_normalization_label(val_spdp)
                        
                        val_loss_batch = torch.abs(val_out - val_spdp).mean()
                        val_loss += val_loss_batch.item()
                        val_count += 1

                    val_loss /= val_count  
                    if not args.ignore_wandb:
                        wandb.log({"val_loss": val_loss})  
                    # Best Validation Loss Update
                    if val_loss < best_val_loss:
                        print(f"best val loss updated: {val_loss}")
                        best_val_loss = val_loss
                        best_val_train_loss = loss
                        torch.save({
                            'model_state_dict': regressor.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        }, model_path+"_val.pt")
            pbar.set_description(f'tr_loss: {loss:.4f} / val_loss: {val_loss:.4f} / t: {t_mean:.1f}')
            pbar.update(1)
    torch.save({
        'model_state_dict': regressor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path+"_last.pt")
    if not args.ignore_wandb:
        wandb.run.summary["best_train_loss"] = best_train_loss
        wandb.run.summary["best_val_loss"] = best_val_loss
        wandb.run.summary["best_val_train_loss"] = best_val_loss

if __name__ == '__main__':

    ## COMMON --------------------------------------------------
    parser = argparse.ArgumentParser(description="generate ppg with regressor guidance")
    parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
    parser.add_argument("--device", type=str, default='cuda')
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
    
    ## Training ------------------------------------------------
    parser.add_argument("--diffusion_time_steps", type=int, default=2000)
    parser.add_argument("--train_epochs", type=int, default=2000)
    parser.add_argument("--init_lr", type=float, default=0.0001)
    parser.add_argument("--init_bias", type=float, default=0.2)
    parser.add_argument("--final_bias", type=float, default=1)
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--t_scheduling", type=str, default="uniform",  choices=["loss-second-moment", "uniform", "train-step"])
    parser.add_argument("--T_max", type=int, default=2000)  
    parser.add_argument("--eta_min", type=float, default=0)  

    ## Sampling ------------------------------------------------
    parser.add_argument("--target_group", type=int, default=1, choices=[0,1,2,3,4], 
                        help="0(hyp0) 1(normal) 2(perhyper) 3(hyper2) 4(crisis) (Default : 1 (normal))")
    args = parser.parse_args()

    main(args)