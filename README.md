# ContrastiveGaussian

Official implementation of [ContrastiveGaussian: High-Fidelity 3D Generation with Contrastive Learning and Gaussian Splatting](https://arxiv.org/abs/2309.16653). 

Our work is accepted by IEEE International Conference on Multimedia & Expo 2025 (ICME 2025).

### [Paper](https://arxiv.org/abs/2504.08100)

## Install

```bash
conda create -n contrastivegaussian python=3.8
conda activate contrastivegaussian

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Real-ESRGAN
# Please follow the instruction of the Real-ESRGAN's repository to implement the Real-ESRGAN
git clone https://github.com/xinntao/Real-ESRGAN.git

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install kiui

pip install -r requirements.txt

```

Tested on:

- Ubuntu 20.04 with Python 3.8 & Torch 2.1 & CUDA 11.8 on a RTX 4090 (24G).

## Usage

```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py data/name.jpg

# process all jpg images under a dir
python process.py data

### stage 1 training
# train 500 iters and export ckpt & coarse_mesh to logs 
python stage1.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# export glb instead of obj
python stage1.py --config configs/image.yaml input=data/name_rgba.png save_path=name mesh_format=glb

# gui mode 
python stage1.py --config configs/image.yaml input=data/name_rgba.png save_path=name gui=True

# use an estimated elevation angle if image is not front-view (e.g., common looking-down image can use -30)
python stage1.py --config configs/image.yaml input=data/name_rgba.png save_path=name elevation=-30

### training mesh stage
# auto load coarse_mesh and refine 50 iters (~1min), export fine_mesh to logs
python stage2.py --config configs/image.yaml input=data/name_rgba.png save_path=name

# gui mode
python stage2.py --config configs/image.yaml input=data/name_rgba.png save_path=name gui=True

# export glb instead of obj
python stage2.py --config configs/image.yaml input=data/name_rgba.png save_path=name mesh_format=glb
```

## Tips
* The world & camera coordinate system is the same as OpenGL:
```
    World            Camera        
  
     +y              up  target                                              
     |               |  /                                            
     |               | /                                                
     |______+x       |/______right                                      
    /                /         
   /                /          
  /                /           
 +z               forward           

elevation: in (-90, 90), from +y to -y is (-90, 90)
azimuth: in (-180, 180), from +z to +x is (0, 90)
```

## Acknowledgement

This work is built on numerous impressive research works and open-source projects. We sincere thanks to the authors for sharing their code.

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## Citation

```
@misc{liu2025contrastivegaussian,
      title={ContrastiveGaussian: High-Fidelity 3D Generation with Contrastive Learning and Gaussian Splatting}, 
      author={Junbang Liu and Enpei Huang and Dongxing Mao and Hui Zhang and Xinyuan Song and Yongxin Ni},
      year={2025},
      eprint={2504.08100},
      archivePrefix={arXiv},
      primaryClass={cs.CV}, 
}
```
