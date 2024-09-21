# Few-Shot PPG Signal Generation via Guided Diffusion Models

This is the official pytorch implementation of the paper ["Few-Shot PPG Signal Generation via Guided Diffusion Models"](https://ieeexplore.ieee.org/document/10680298), Kang et al.

## Quick Summary

![image](https://github.com/user-attachments/assets/a5f33682-04f3-4c07-a1b5-a2d593cd1914)
Can guide biased generation via diffusion models in unbalanced datasets to force generation for minority groups.

![OevrallFramework](https://github.com/user-attachments/assets/dac35b7c-00ba-4050-b8cf-3a501ae2b85f)
However, minority groups cannot create validity easily because of the group attribute. Therefore, Bi-guidance, continuous and deterministic guidance, guiding to the target group, and applying data enhancement generate plausible data.

![image](https://github.com/user-attachments/assets/a754732f-798b-4cc6-9248-4fa5bb720b84)
It can generate data covering the distribution of unseen real data, even though it was trained and generated in a few-shot setting.


## Abstract
Recent advancements in deep learning for predicting Arterial Blood Pressure (ABP) have prominently featured photoplethysmography (PPG) signals. Notably, PPG signals exhibit significant variability due to differences in measurement environments, alongside stark disparities in the distribution of collected signal data among different labels. To address these challenges, this study introduces a Bi-Guided Diffusion Model designed to generate PPG signals with expected features of ABP within a few-shot setting for each label group. We propose a guided diffusion model architecture that simultaneously considers both the determinant group condition and the continuous label condition for each group in a few-shot setting. To our knowledge, this is the first study to use a diffusion model for generating PPG signals with a limited dataset. Initially, we categorized them into four groups based on SBP and DBP values: Hypo, Normal, Prehyper, and Hyper2. In each group, we sample an equal number of data points according to the few-shot setting and then generate appropriate PPG signals for each group through guidance.
Additionally, Our study proposes a post-processing technique to address the limitations of generative models in few-shot settings, consistently boosting performance across various methods such as training from scratch, transfer learning, and linear probing. When benchmarked, our methodology demonstrated performance improvements across all datasets, including BCG, PPGBP, and SENSORS. We confirmed data quality by comparing training, generated, and actual data. We analyzed error cases, morphology features, and t-SNE distribution to highlight the role of synthetic data in enhancing performance.


## Model and Data Directory

- Model code example
    - code\train\core\load_model.py

    ```
    import os
    root = '/path/to/model'
    join = os.path.join
    model_fold = {"bcg": {0: {0: join(root, 'bcg-resnet1d/fold0.ckpt'),
                        {1: join(root, 'bcg-resnet1d/fold1.ckpt'),
                        {2: join(root, 'bcg-resnet1d/fold2.ckpt'),
                        {3: join(root, 'bcg-resnet1d/fold3.ckpt'),
                        {4: join(root, 'bcg-resnet1d/fold4.ckpt')}}}

    ```
- Data

    ```
    ├── bp-algorithm
    ├── datasets
    │   ├── splits
    │   │   ├── bcg_dataset
    │   │   │   ├── feat_fold_0.csv
    │   │   │   ├── feat_fold_1.csv
    │   │   │   ├── feat_fold_2.csv
    │   │   │   ├── feat_fold_3.csv
    │   │   │   ├── feat_fold_4.csv

    ```


## Explanation of the arguments and Example
### Key Arguments:

- `--seed`: Random seed for reproducibility (default: `1000`).
- `--num_samples`: Number of samples to process (default: `5`).
- `--seq_length`: Sequence length for the data (default: `625`).
- `--train_batch_size`: Batch size during training (default: `32`).
- `--benchmark`: Dataset or benchmark to use (e.g., `bcg`) (default: `'bcg'`).

### Training Options:

- `--diffusion_time_steps`: Number of diffusion time steps (default: `2000`).
- `--train_num_steps`: Number of training steps (default: `32`).
- `--train_lr`: Learning rate for training (default: `8e-5`).

### Sampling Options:

- `--sample_only`: Skip training and only run sampling (default: `False`).
- `--sample_batch_size`: Batch size for sampling (if needed).
- `--target_group`: Target group for sampling (`-1`: all, `0`: hyp0, `1`: normal, `2`: perhyper, `3`: hyper2, `4`: crisis) (default: `-1`).
- `--regressor_scale`: Scaling factor for regressor (default: `1.0`).

### Code Example
```
conda env create -f environment.yml

# Train and Generate Synthetic PPG for all default options
python main.py

```
To evaluate generated PPG
please use this [repository](https://github.com/inventec-ai-center/bp-benchmark).



