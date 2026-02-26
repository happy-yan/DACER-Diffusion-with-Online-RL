# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

DACER (Diffusion Actor-Critic with Entropy Regulator) is a JAX-based reinforcement learning framework for continuous control tasks in MuJoCo environments. The Python package is called `relax`. It has no web UI, databases, or network services — it is a pure compute/research workload.

### Environment setup

The conda environment `relax` (Python 3.11) is pre-configured and auto-activated via `~/.bashrc`. MuJoCo 210 binaries are installed at `~/.mujoco/mujoco210` with `LD_LIBRARY_PATH` set accordingly.

### Key gotchas

- **JAX version**: The codebase requires JAX 0.4.27. Newer versions (0.5+) break due to removed APIs (`jax.core.ClosedJaxpr`). Do not upgrade JAX.
- **Cython version**: Must use `Cython<3` (0.29.x). Cython 3.x breaks `mujoco-py` compilation due to `noexcept` callback issues.
- **`setup.py` requires `$CONDA_PREFIX`**: The C extensions build reads `$CONDA_PREFIX` for include/lib paths. Always have the conda `relax` env activated before running `pip install -e .`.
- **No GPU in Cloud VM**: JAX runs on CPU. Training is slow but functional. Use small `--total_step` values for testing.
- **First `mujoco-py` import compiles Cython**: The first import of any MuJoCo environment triggers a ~30s Cython compilation. Subsequent imports are cached.

### Running

See `README.md` for training commands. Quick test:

```bash
python scripts/train_mujoco.py --alg sac --env HalfCheetah-v3 --num_vec_envs 0 --total_step 200 --start_step 20 --seed 42
```

Use `--num_vec_envs 0` for single-env (no multiprocessing) mode in the cloud VM.

### Lint

```bash
black --check .
```

### No automated tests

The repository does not include a test suite. Validation is done by running training scripts and checking that they produce logs and checkpoints in `logs/`.
