# ptv3-vertebrae-segmentation

Adaptation of **Point Transformer v3 (PTv3)** for vertebrae segmentation on 3D point cloud data.

This repository contains only the components required to reproduce the vertebrae segmentation experiments.
 
The core PTv3 implementation is adapted from the official Pointcept repository.

👉 **Official PTv3 repository:**  
https://github.com/Pointcept/PointTransformerV3

## Repository Scope

This repository includes:
- Adapted PTv3 model wrappers
- Custom dataloaders for the SpineDepth dataset
- Training scripts for semantic segmentation
- Minimal utilities required for reproduction

It **does not** include:
- Full Pointcept framework
- Raw datasets


## Installation Set Up
```
conda create -n ptv3 python=3.8 -y
conda activate ptv3

conda install ninja -y

# Install PyTorch 2.1.0, Torchvision 0.16.0, and Torchaudio 2.1.0 with CUDA 11.8 support
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cuda-nvrtc-dev=11.8 -c pytorch -c nvidia -y
conda install pytorch-scatter pytorch-sparse pytorch-cluster -c pyg -y
conda install cuda-toolkit=11.8 -c nvidia -y

conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y

pip install -r requirements.txt 
```

## Dataset

This project uses the SpineDepth dataset, a publicly available cadaveric RGB-D spinal dataset.

Dataset paper: Liebmann et al., 2021

Data is assumed to be preprocessed and annotated

Each frame is stored as a NumPy array with shape (N, 10):

```
[x, y, z, r, g, b, h, s, v, label]
```

By default, symbolic links are used:

```
/store/SpineDepth               # original dataset
/store/SpineDepth_labeled_data  # labeled point clouds
```

Create symbolic link and change the root in the code to the right path



# Training 

## Quick Start
A lightweight training script using a subset of the dataset is provided for fast experimentation:

```
python train_sem_seg_ptv3_num.py 
```
During training:

- A new folder will be created under log/

- Model checkpoints and logs will be saved automatically

## Visualization
After training, predictions can be visualized using:

```
train_visualize_SpineDepth.ipynb
```

Before running:

Update log_train_dir and log_best_dir to point to your training outputs

## Configuration
Training hyperparameters are defined in parse_args inside each training script.
Initial settings are in the origin code


# Semi-Synthetic Data Creation

We provide a Python script designed to be executed within **Blender** to generate semi-synthetic point clouds sampled from 3D surface meshes with tagged points.

- Script: `tag_points.py`, `main_process.py`
- Purpose: create tag points in json file -> generate point clouds -> saved as .ply files

The generated point clouds can be further processed using the provided preprocessing scripts.
---

## Data Preparation for Semi-Synthetic Data (Antonin’s Dataset)

Before running the preprocessing pipeline, ensure that the dataset follows the directory structure described below.

After verification, run:

```bash
preprocess_o3file.ipynb
```
This notebook will generate an SJxxxxxxx.json metadata file inside each acquisition-date folder, which is required for subsequent processing steps.

## Dataset Directory Structure

Each subject (SJxxxxx) may contain multiple acquisition dates.
Each acquisition-date folder includes 2D radiographs and 3D geometry files.
```
<path_to_Antonins_Dataset>/
├── SJ0000270/
│   ├── 2006-09-05/
│   │   ├── SJ0000270_lat.jpg      # Lateral radiograph
│   │   ├── SJ0000270_pa_0.jpg     # Posteroanterior (PA) radiograph
│   │   ├── SJ0000270.o2           # 3D data file (format-specific)
│   │   ├── SJ0000270.o3           # 3D data file (format-specific)
│   │   └── SJ0000270.wrl          # 3D surface model (VRML)
│   ├── 2006-12-12/
│   │   ├── SJ0000270_lat.jpg
│   │   ├── SJ0000270_pa_0.jpg
│   │   ├── SJ0000270.o2
│   │   ├── SJ0000270.o3
│   │   └── SJ0000270.wrl
│
├── SJ0000285/
├── SJ0000321/
├── ...
```


# Citation
```
@inproceedings{wu2024point,
  title={Point transformer v3: Simpler faster stronger},
  author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4840--4851},
  year={2024}
}

@article{liebmann2021spinedepth,
  title={Spinedepth: a multi-modal data collection approach for automatic labelling and intraoperative spinal shape reconstruction based on rgb-d data},
  author={Liebmann, Florentin and St{\"u}tz, Dominik and Suter, Daniel and Jecklin, Sascha and Snedeker, Jess G and Farshad, Mazda and F{\"u}rnstahl, Philipp and Esfandiari, Hooman},
  journal={Journal of Imaging},
  volume={7},
  number={9},
  pages={164},
  year={2021},
  publisher={MDPI}
}
```