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

## 4. Ray cluster usage

### Interactive (Remote-Desktop) debugging
```bash
# Inside your GPU node
source ~/venv_reasoning360mtl/bin/activate
ray stop -f                        # clean up any prior cluster
PORT=6379
ray start --head --port=$PORT \
          --num-cpus 4 --num-gpus 1 --block &
# Tell training scripts where the head is
export RAY_ADDRESS="$(hostname --ip-address):$PORT"
```
All training scripts automatically:
* Skip starting a new Ray cluster when `RAY_ADDRESS` is set.
* Connect to the existing cluster via that address.

### Batch jobs (Slurm)
`scripts/train/example_multinode_mtl_qwen2_7b.sh` starts Ray on the first node
and propagates the address to all workers. No manual steps required. If you
prefer to supply your own cluster, simply export `RAY_ADDRESS` **before**
invoking the script and it will join instead of starting a new one.

---
Happy hacking!
