# ptv3-vertebrae-segmentation
Adapted Point Transformer v3 for vertebrae segmentation on 3D point cloud data

For more information, please refer to (https://github.com/Pointcept/PointTransformerV3). The code will be updated in Pointcept v1.5. 



## Installation Set Up
```
conda create -n ptv3 python=3.8 -y
conda activate ptv3
<<<<<<< HEAD
=======
conda install ninja -y
>>>>>>> 02a7287 (added conda install cuda-nvrtc-dev)

# Install PyTorch 2.1.0, Torchvision 0.16.0, and Torchaudio 2.1.0 with CUDA 11.8 support
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cuda-nvrtc-dev=11.8 -c pytorch -c nvidia -y
conda install pytorch-scatter pytorch-sparse pytorch-cluster -c pyg -y
conda install cuda-toolkit=11.8 -c nvidia -y

conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y

pip install -r requirements.txt 
```

