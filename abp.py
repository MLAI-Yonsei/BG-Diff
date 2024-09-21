import time
import wandb
import argparse
import pickle
import torch
import os 
import pandas as pd
from denoising_diffusion_pytorch.model import Classifier, Regressor
from denoising_diffusion_pytorch.cond_fn import classifier_cond_fn, regressor_cond_fn
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_guided import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from vis import visualize
import data
import paths
import tqdm

meta_data_path = paths.DATA_ROOT
seed = 1000

for meta_data in tqdm(os.listdir(meta_data_path)):
        with open(f'{meta_data_path}/{meta_data}', "r") as meta:
            df_meta = pd.read_csv(meta, index_col=0)
            if df_meta.shape[0] == 0:
                continue
            patientID = df_meta.iloc[0, 0]

            # Sampling max 100 segment per patient
            count = 0
            df_meta = df_meta.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffling and reset index
            for row in range(df_meta.shape[0]):
                segmentID = df_meta.iloc[row, 2]
                file = f'{args.data_path}{patientID}/{segmentID}.csv'

                sbp = df_meta.iloc[row, 5]
                dbp = df_meta.iloc[row, 6]
                with open(file, "r") as f:
                    content = pd.read_csv(f, header=None)
                    time = list(content.iloc[:, 0])
                    ppg = list(content.iloc[:, 1])