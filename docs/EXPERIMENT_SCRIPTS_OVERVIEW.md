# 📋 Experiment Scripts Overview

## 🎯 Complete MTL vs Baseline Comparison Pipeline

This document provides a comprehensive overview of all experiment scripts and their relationships for the MTL vs Baseline comparison experiments.

---

## 📁 **Main Workflow Architecture**

```
run_mtl_comparison.sh (LAUNCHER)
├── run_guru_baseline.sh (Baseline Experiment)
│   └── example_multinode_mtl_qwen2_7b.sh (Core Training)
└── run_mtl_pcgrad.sh (MTL Experiment)  
    └── example_multinode_mtl_qwen2_7b.sh (Core Training)

After Training Complete:
├── run_evaluation_comparison.sh (Evaluation)
│   └── example_multinode_eval_guru7b_mtl.sh (Evaluation Engine)
├── run_checkpoint_evolution_analysis.sh (Evolution Analysis)
│   └── example_multinode_eval_guru7b_mtl.sh (Evaluation Engine)
└── analyze_results.py (Results Analysis)
```

---

## 📊 **Detailed Script Breakdown**

### **1. 🚀 `run_mtl_comparison.sh` - MAIN LAUNCHER**

**Purpose**: Single command to start both experiments  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/run_mtl_comparison.sh`

**What it does**:
- Submits baseline experiment to SLURM
- Submits MTL experiment to SLURM  
- Provides monitoring instructions
- Shows expected completion time

**Usage**:
```bash
cd /home/jinming/Reasoning360-MTL/scripts/train
bash run_mtl_comparison.sh
```

**Output**:
- Job ID for baseline experiment
- Job ID for MTL experiment
- Monitoring commands
- WandB dashboard link

---

### **2. 📊 `run_guru_baseline.sh` - BASELINE EXPERIMENT**

**Purpose**: SLURM job script for baseline (no MTL)  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/run_guru_baseline.sh`

**SLURM Configuration**:
```bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
```

**Environment Variables**:
```bash
export MTL_ENABLED=false
export WANDB_EXPERIMENT_NAME="guru-baseline-$(date +%Y%m%d-%H%M%S)"
```

**Calls**: `example_multinode_mtl_qwen2_7b.sh`

---

### **3. 📊 `run_mtl_pcgrad.sh` - MTL EXPERIMENT**

**Purpose**: SLURM job script for MTL with PCGrad  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/run_mtl_pcgrad.sh`

**SLURM Configuration**:
```bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
```

**Environment Variables**:
```bash
export MTL_ENABLED=true
export MTL_METHOD=pcgrad
export RAY_PORT=6380  # Avoid conflicts with baseline
export WANDB_EXPERIMENT_NAME="mtl-pcgrad-$(date +%Y%m%d-%H%M%S)"
```

**Calls**: `example_multinode_mtl_qwen2_7b.sh`

---

### **4. ⚙️ `example_multinode_mtl_qwen2_7b.sh` - CORE TRAINING ENGINE**

**Purpose**: The actual PPO training script (used by both experiments)  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/example_multinode_mtl_qwen2_7b.sh`

**Key Configuration**:
```bash
# Training Parameters
trainer.total_epochs=30
trainer.val_before_train=true  
trainer.test_freq=3

# Early Stopping (Following GURU Paper)
+trainer.early_stopping_min_epochs=10
+trainer.early_stopping_patience=3
+trainer.early_stopping_min_delta=0.001
+trainer.early_stopping_metric='validation_reward_mean'
+trainer.early_stopping_mode='max'

# Checkpointing
trainer.save_freq=5  # Save every 5 epochs
trainer.max_actor_ckpt_to_keep=10  # Keep last 10 checkpoints

# Data & Models
TRAIN_FILES="[/home/jinming/Reasoning360-MTL/data/online_eval/math__math_500.parquet]"
BASE_MODEL=/home/jinming/llm_models/Qwen2.5-7B

# MTL Configuration (controlled by environment variables)
mtl.enabled=${MTL_ENABLED:-true}
mtl.method=${MTL_METHOD:-pcgrad}
```

**Features**:
- Single-node Ray cluster setup (SLURM-friendly)
- Early stopping following GURU paper methodology
- Automatic GPU detection
- WandB logging integration
- Improved reward functions integration

---

### **5. 📈 `run_evaluation_comparison.sh` - POST-TRAINING EVALUATION**

**Purpose**: Evaluate and compare final models  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/run_evaluation_comparison.sh`

**What it does**:
- Finds latest baseline and MTL checkpoints
- Submits evaluation jobs for both models
- Evaluates across all domains (Math, Logic, Code, Science, Simulation, Tabular)
- Organizes results for comparison

**Usage**:
```bash
# After training completes
bash run_evaluation_comparison.sh
```

**Output**:
- Evaluation job IDs for both models
- Results organized in directories
- Ready for analysis script

---

### **6. 📊 `example_multinode_eval_guru7b_mtl.sh` - EVALUATION ENGINE**

**Purpose**: Domain-specific model evaluation  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/example_multinode_eval_guru7b_mtl.sh`

**Evaluation Coverage**:
- **Math**: AIME (240), MATH500 (500), AMC (332)
- **Logic**: Zebra puzzles (200), Ordering puzzles (100)  
- **Code**: HumanEval (164), MBPP (200), LiveCodeBench (279)
- **Science**: SuperGPQA (200)
- **Simulation**: CodeI/O (200), ARC-AGI (200)
- **Tabular**: HiTab (200), MultiHiertt (200)

**Total**: 19 tasks across 6 domains + OOD

**Used by**: Both evaluation and evolution analysis scripts

---

### **7. 📈 `run_checkpoint_evolution_analysis.sh` - EVOLUTION ANALYSIS**

**Purpose**: Analyze performance evolution across training epochs  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/run_checkpoint_evolution_analysis.sh`

**What it does**:
- Finds ALL checkpoints (epoch_5, epoch_10, epoch_15, epoch_20, epoch_25, epoch_30)
- Evaluates each checkpoint across all domains
- Tracks learning curves per domain
- Identifies optimal stopping points
- Detects overfitting patterns

**Usage**:
```bash
# After training completes
bash run_checkpoint_evolution_analysis.sh
```

**Benefits**:
- Performance evolution tracking
- Domain-specific learning curves
- Optimal stopping point identification
- Overfitting detection
- Comparative analysis between MTL and baseline

---

### **8. 🔍 `analyze_results.py` - RESULTS ANALYSIS**

**Purpose**: Parse evaluation logs and generate comparison reports  
**Location**: `/home/jinming/Reasoning360-MTL/scripts/train/analyze_results.py`

**Features**:
- Loads evaluation results from both models
- Compares domain-wise performance
- Generates comprehensive markdown reports
- Identifies improvements and regressions
- Provides actionable recommendations

**Usage**:
```bash
# After evaluations complete
python analyze_results.py
```

**Output**:
- Detailed comparison reports
- Performance improvement analysis
- Domain-specific insights
- Recommendations for future work

---

## 🔄 **Script Dependencies & Execution Flow**

### **Phase 1: Training**
```
1. run_mtl_comparison.sh
   ├── Submits → run_guru_baseline.sh
   │   └── Executes → example_multinode_mtl_qwen2_7b.sh (MTL_ENABLED=false)
   └── Submits → run_mtl_pcgrad.sh  
       └── Executes → example_multinode_mtl_qwen2_7b.sh (MTL_ENABLED=true)
```

### **Phase 2: Evaluation**
```
2. run_evaluation_comparison.sh
   ├── Finds latest checkpoints
   ├── Submits evaluation jobs
   └── Uses → example_multinode_eval_guru7b_mtl.sh

3. run_checkpoint_evolution_analysis.sh  
   ├── Finds ALL checkpoints
   ├── Submits evaluation jobs for each epoch
   └── Uses → example_multinode_eval_guru7b_mtl.sh
```

### **Phase 3: Analysis**
```
4. analyze_results.py
   ├── Reads evaluation outputs
   ├── Compares baseline vs MTL
   └── Generates comprehensive reports
```

---

## 📊 **Configuration Summary**

### **Training Configuration**:
- **Model**: Qwen2.5-7B (local path: `/home/jinming/llm_models/Qwen2.5-7B`)
- **Data**: Math domain (500 samples)
- **Resources**: 2 GPUs, 32 CPUs per job
- **Duration**: 30 epochs maximum
- **Early Stopping**: Minimum 10 epochs, patience 3
- **Validation**: Every 3 epochs
- **Checkpoints**: Every 5 epochs, keep last 10

### **MTL Configuration**:
- **Method**: PCGrad (Project Conflicting Gradients)
- **Baseline**: MTL disabled
- **Comparison**: Direct A/B testing

### **Evaluation Configuration**:
- **Domains**: 6 reasoning domains + OOD
- **Tasks**: 19 total evaluation tasks
- **Alignment**: Matches GURU paper evaluation suite
- **Coverage**: ~3,500 evaluation samples

---

## 🎯 **Current Experiment Status**

### **Active Jobs**:
- **Baseline**: Job ID 3270070 (pending)
- **MTL**: Job ID 3270071 (pending)
- **Started**: 2025-08-03 19:39:13 EDT
- **Expected Duration**: 4-8 hours with early stopping

### **Monitoring**:
```bash
# Check job status
squeue -u jinming

# Monitor training progress
tail -f guru_baseline_3270070.out
tail -f mtl_pcgrad_3270071.out

# WandB dashboard
https://wandb.ai/jin-ming-vt/Reasoning360-MTL
```

### **Next Steps**:
1. **Wait for training completion** (4-8 hours)
2. **Run evaluation comparison**: `bash run_evaluation_comparison.sh`
3. **Run evolution analysis**: `bash run_checkpoint_evolution_analysis.sh`
4. **Generate analysis report**: `python analyze_results.py`

---

## 🔧 **Key Features & Improvements**

### **SLURM Integration**:
- Fixed nested srun issues
- Single-node Ray cluster setup
- Proper resource allocation
- Queue-friendly configuration

### **Early Stopping**:
- Follows GURU paper methodology
- Minimum 10 epochs before stopping
- Patience of 3 epochs
- Validation-based stopping criterion

### **Comprehensive Evaluation**:
- 19 tasks across 6 domains
- Matches paper evaluation suite
- Evolution analysis across epochs
- Automated comparison reports

### **Improved Reward Functions**:
- Integrated fallback reward functions
- Better format compliance
- Reduced zero reward rates
- Enhanced training stability

---

## 📚 **Documentation & References**

### **Related Files**:
- **Configuration**: `/home/jinming/Reasoning360-MTL/verl/trainer/config/ppo_trainer.yaml`
- **Data**: `/home/jinming/Reasoning360-MTL/data/online_eval/`
- **Checkpoints**: `/home/jinming/Reasoning360-MTL/checkpoints/Reasoning360-MTL/`
- **Results**: Generated in experiment-specific directories

### **Key Papers**:
- GURU: Universal Reasoning via Multi-Task Learning
- PCGrad: Gradient Surgery for Multi-Task Learning
- PPO: Proximal Policy Optimization

---

**Last Updated**: 2025-08-03 19:55 EDT  
**Status**: Experiments running, documentation complete  
**Next Review**: After training completion (~4-8 hours)
