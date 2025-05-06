# [NeurIPS 2024] DACER

This is the official repository of

**Diffusion Actor-Critic with Entropy Regulator**,

<p align="left">
<a href='https://arxiv.org/abs/2405.15177' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

## Installation

```bash
# Create environemnt
conda create -n relax python=3.11 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate relax

# One of: Install jax WITH CUDA 
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install package
pip install -r requirements.txt
pip install -e .
```

## Run
```bash
# Run one experiment
XLA_FLAGS='--xla_gpu_deterministic_ops=true' CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.1 python scripts/train_mujoco.py --alg dacer --seed 100
```

```bash
# Run multiple experiments
chmod +x run_experiments.sh
./run_experiments.sh "0 1 2"
```

## To Do

The code is under cleaning and will be released gradually.

- [ ] improve docs
- [x] more diffusion-based code [DIPO, QSM. QVPO(other branch)]
- [x] training code


## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@article{wang2024diffusion,
  title={Diffusion actor-critic with entropy regulator},
  author={Wang, Yinuo and Wang, Likun and Jiang, Yuxuan and Zou, Wenjun and Liu, Tong and Song, Xujie and Wang, Wenxuan and Xiao, Liming and Wu, Jiang and Duan, Jingliang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={54183--54204},
  year={2024}
}
```

## Tips
1. Search "other envs" in the code to find the parameters for other mujoco environments. The existing parameters in the code are used for Humanoid-v3 and HalfCheetah-v3.
2. The version of MuJoCo is mujoco210.
