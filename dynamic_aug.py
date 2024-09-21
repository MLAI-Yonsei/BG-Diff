import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from denoising_diffusion_pytorch.model import *
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import argparse
import paths
import pickle
from torch.utils.data import ConcatDataset, DataLoader, Dataset

def main(args):
    device = torch.device("cuda")

    # initial Group augmented data samples count
    group_augmented_samples = [100, 100, 100, 100]
    
    data = get_data(sampling_method='first_k',
                                        num_samples=5,
                                        data_root=paths.DATA_ROOT,
                                        benchmark='bcg',
                                        train_fold=args.train_fold)
    train_dataset = Dataset1D(data['train']['ppg'], label=data['train']['spdp'], groups=data['train']['group_label'], normalize=True)
    model = ResNet1D(output_size=2, final_layers=args.final_layers, n_block=args.n_block, 
                         concat_label_mlp=args.concat_label_mlp, g_pos=args.g_pos, g_mlp_layers=args.g_mlp_layers).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    # Function to perform training and return group-wise losses

    previous_group_losses = {group: float('inf') for group in range(4)}  # Initialize with infinity

    for epoch in range(1, 101):
        # Get the combined dataset for the current epoch
        combined_dataset = get_combined_dataset(group_augmented_samples, train_dataset)
        train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
        # Train the model and get the current losses
        import pdb;pdb.set_trace()
        current_group_losses = train_and_evaluate(model, train_loader, optimizer, criterion)

        if epoch > 50:
            # Adjust the number of augmented samples based on the comparison of losses
            for group in range(4):
                loss_improvement = previous_group_losses[group] - current_group_losses[group]
                if loss_improvement < 0:
                    # If loss has not decreased, revert the increase
                    group_augmented_samples[group] = max(10, group_augmented_samples[group] - 20) # decrease by 10 from the original increment
                previous_group_losses[group] = current_group_losses[group]
        else:
            # For epochs <= 50, just update the previous losses without changing the samples
            previous_group_losses = current_group_losses

        # Print out losses and augmented samples for monitoring
        print(f'Epoch {epoch}: Group Losses: {current_group_losses}')
        print(f'Augmented Samples Per Group: {group_augmented_samples}')

# Function to train the model and get the loss for each group
def train_and_evaluate(model, train_loader, optimizer, criterion):
    model.train()
    group_losses = {group: 0.0 for group in range(4)}  # Assuming 4 groups
    group_counts = {group: 0 for group in range(4)}  # To count samples per group

    for inputs, labels, groups in train_loader:
        import pdb;pdb.set_trace()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        for group in range(4):
            group_mask = (groups == group)
            group_losses[group] += loss.item() * group_mask.sum().item()
            group_counts[group] += group_mask.sum().item()

    # Calculate the average loss per group
    for group in group_losses:
        group_losses[group] /= group_counts[group]

    return group_losses

def load_augmented_data_for_group(tar_group, num_samples):
    path = f'/mlainas/ETRI_2023/sampling_results/fold_0/seed_1000_sampling_method_first_k_num_samples_5-diffusion_time_steps_2000-train_num_steps_32_bcg_guided_train_lr_8e-05_reg_set_val_tr_group_average_loss_sel_erm/sample_group_average_loss_{tar_group}.pkl'
    
    with open(path, 'rb') as f:
        sample = pickle.load(f)
    
    # Here we ensure that only num_samples samples are loaded
    sampled_seq = sample['sampled_seq'].squeeze()
    gen_ppg = sampled_seq[:num_samples].cuda()
    gen_y = sample['y'][:num_samples].cuda()
    gen_group = torch.full((num_samples,), tar_group).cuda()
    
    return Dataset1D(gen_ppg, label=gen_y, groups=gen_group, normalize=True)

def get_combined_dataset(group_augmented_samples, real_dataset):
    # Load augmented datasets with the specified number of samples for each group
    augmented_datasets = [load_augmented_data_for_group(tar_group, num_samples)
                          for tar_group, num_samples in enumerate(group_augmented_samples)]
    
    # Combine real and augmented datasets
    real_dataset = ReshapeDataset(real_dataset)
    combined_dataset = ConcatDataset([real_dataset] + augmented_datasets)
    return combined_dataset

class ReshapeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        # Get the sample from the original dataset
        x, y, z = self.dataset[index]
        x, y, z = x.cuda(), y.cuda(), z.cuda()
        # Reshape or squeeze the sample here
        x = x.squeeze()  # This removes the singleton dimension
        return x, y, z

    def __len__(self):
        return len(self.dataset)

# Don't forget to include validation, checkpoint saving, early stopping as needed.
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
    main(args)
