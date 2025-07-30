# Reasoning360-MTL Environment Usage

## 1. Login / CPU work (interactive)
```bash
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
source ~/venv_reasoning360mtl/bin/activate
```

The virtual environment `~/venv_reasoning360mtl` contains all CPU-only dependencies.

## 2. GPU jobs (ARC compute nodes)
Inside your job script, after the two commands above, add:
```bash
module load PyTorch/2.1.2-foss-2023a
```
This overrides the CPU wheels with GPU-enabled PyTorch/CUDA libraries supplied by the cluster toolchain.

## 3. Rebuilding the environment
Run the installer any time you need a fresh venv:
```bash
bash setup_arc_env.sh
```

---
Happy hacking!
