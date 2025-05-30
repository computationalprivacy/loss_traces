# Code for the paper "Free Record-Level Privacy Risk Evaluation through Artifact-Based Methods"
## Abstract
Membership inference attacks (MIAs) are widely used to empirically assess privacy risks in machine learning models, both
providing model-level vulnerability metrics and identifying
the most vulnerable training samples. State-of-the-art methods, however, require training hundreds of shadow models
with the same architecture as the target model. This makes
the computational cost of assessing the privacy of models
prohibitive for many practical applications, particularly when
used iteratively as part of the model development process and
for large models. We propose a novel approach for identifying
the training samples most vulnerable to membership inference attacks by analyzing artifacts naturally available during
the training process. Our method, Loss Trace Interquartile
Range (LT-IQR), analyzes per-sample loss trajectories collected during model training to identify high-risk samples
without requiring any additional model training. Through
experiments on standard benchmarks, we demonstrate that
LT-IQR achieves 92% precision@k= 1% in identifying the
samples most vulnerable to state-of-the-art MIAs. This result
holds across datasets and model architectures with LT-IQR
outperforming both traditional vulnerability metrics, such as
loss, and lightweight MIAs using few shadow models. We
also show LT-IQR to accurately identify points vulnerable to
multiple MIA methods and perform ablation studies. We believe LT-IQR enables model developers to identify vulnerable
training samples, for free, as part of the model development
process. Our results emphasize the potential of artifact-based
methods to efficiently evaluate privacy risks.

## Setup

To install dependencies, run:
```

pip install -r requirements.txt

```
Next create a config.py file with the following and fill in the relevant paths:

```

LOCAL_DIR = # path to this folder
# paths to store stuff...
STORAGE_DIR = 
MY_STORAGE_DIR = 
MODEL_DIR = 
DATA_DIR = 

```

## Quick Start

### 1. Run the Complete Pipeline

```bash
cd loss_traces
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --full
```

This will:
1. Train a WideResNet28-2 target model on CIFAR10 (100 epochs)
2. Train 256 shadow models with the same architecture  
3. Run the LiRA membership inference attack

### 2. Check Status

```bash
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --status
```

### 3. Run Individual Stages

```bash
# Train only the target model
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --target-only

# Train only the shadow models
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --shadows-only

# Run only the attack (requires target and shadow models to exist)
python run_attack_pipeline.py --exp_id wrn28-2_CIFAR10_demo --attack-only
```

## Configuration Options

### Basic Options

```bash
python run_attack_pipeline.py \
    --exp_id my_experiment \
    --arch wrn28-2 \
    --dataset CIFAR10 \
    --n_shadows 256 \
    --gpu :0 \
    --seed 2546 \
    --full
```

### Advanced Options

- `--force`: Force retrain/rerun even if outputs already exist
- `--target-only`: Train target model only
- `--shadows-only`: Train shadow models only  
- `--attack-only`: Run attack only
- `--status`: Show current experiment status

## Architecture Support

The pipeline supports the following architectures (via `--arch`):

- `wrn28-2`: WideResNet28-2 (default)
- `wrn28-10`: WideResNet28-10
- `wrn40-4`: WideResNet40-4


## Dataset Support

Currently supported datasets (via `--dataset`):
- `CIFAR10` (default)
- `CIFAR100`
- `CINIC10`

## Training Configuration

The pipeline uses optimized hyperparameters for WideResNet28-2 on CIFAR10:

- **Batch size**: 256
- **Learning rate**: 0.1  
- **Epochs**: 100
- **Weight decay**: 5e-4
- **Momentum**: 0.9
- **Data augmentation**: Enabled
- **Optimizer**: SGD with Cosine Annealing

## Output Structure

The pipeline creates the following directory structure:

```
{storage_dir}/trained_models/{exp_id}/
├── target                    # Target model
├── shadow_0                  # Shadow model 0
├── shadow_1                  # Shadow model 1
├── ...
└── shadow_256                 # Shadow model 256

{storage_dir}/lira_scores/
└── {exp_id}_target          # LiRA attack results (CSV)

{storage_dir}/losses/
└── {exp_id}_target          # Computed loss traces

{storage_dir}/scaled_logits_intermediate/
└── {exp_id}.pt              # Intermediate statistics for attack
```

## Examples

### Example 1: Quick Demo with Fewer Shadow Models

```bash
python run_attack_pipeline.py \
    --exp_id quick_demo \
    --n_shadows 16 \
    --full
```

### Example 2: Different Architecture

```bash
python run_attack_pipeline.py \
    --exp_id resnet_experiment \
    --arch rn-18 \
    --n_shadows 32 \
    --full
```

### Example 3: GPU Training

```bash
python run_attack_pipeline.py \
    --exp_id gpu_experiment \
    --gpu :0 \
    --full
```

### Example 4: Resume Interrupted Training

```bash
# Check what's already completed
python run_attack_pipeline.py --exp_id my_experiment --status

# Continue from where it left off
python run_attack_pipeline.py --exp_id my_experiment --full
```

### Example 5: Force Complete Retrain

```bash
python run_attack_pipeline.py \
    --exp_id my_experiment \
    --full \
    --force
```

## Results Interpretation

After completion, the attack results are saved as a CSV file containing:

- `lira_score`: The LiRA attack score for each sample
- `target_trained_on`: Boolean indicating if sample was in target training set
- `og_idx`: Original dataset index

Higher LiRA scores indicate higher likelihood of membership in the training set.


### Resume After Interruption

The pipeline automatically skips completed components:

```bash
# This will skip any existing models and continue from where it left off
python run_attack_pipeline.py --exp_id my_experiment --full
```

### Clean Restart

```bash
# Force complete retrain
python run_attack_pipeline.py --exp_id my_experiment --full --force
```

## Prerequisites

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Required packages (see `requirements.txt`)
- Properly configured `config.py` with data paths

## Files Generated

The pipeline generates several types of files:

1. **Model files**: Trained PyTorch models with weights and metadata
2. **Attack results**: CSV files with membership inference scores  
3. **Intermediate statistics**: Cached statistics for faster attack reruns
4. **Logs**: Training progress and attack execution logs



## References

[^1]: N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramer,
“Membership inference attacks from first principles,” in 2022 IEEE
Symposium on Security and Privacy (SP). IEEE, 2022, pp. 1897–
1914

[^2]: J. Ye, A. Maddi, S. K. Murakonda, V. Bindschaedler, and R. Shokri,
“Enhanced membership inference attacks against machine learning
models,” in Proceedings of the 2022 ACM SIGSAC Conference on
Computer and Communications Security, 2022, pp. 3093–3106.
