# Code for the paper "Free Record-Level Privacy Risk Evaluation through Artifact-Based Methods"
## Abstract
Membership inference attacks (MIAs) are widely
used to empirically assess the privacy risks of samples
used to train a target machine learning model. State-of-the-
art methods however require training hundreds of shadow
models, with the same size and architecture of the target
model, solely to evaluate the privacy risk. While one might
be able to afford this for small models, the cost often becomes
prohibitive for medium and large models.
We here instead propose a novel approach to identify
the at-risk samples using only artifacts available during
training, with little to no additional computational overhead.
Our method analyzes individual per-sample loss traces and
uses them to identify the vulnerable data samples. We
demonstrate the effectiveness of our artifact-based approach
through experiments on the CIFAR10 dataset, showing high
precision in identifying vulnerable samples as determined by
SOTA shadow model-based MIA(LiRA [^1]). Impressively,
our method reaches the same precision than SOTA MIAs
with one another despite despite it being orders of magnitude
cheaper. We then show LT-IQR to outperform alternative
loss aggregation methods, perform ablation studies on hy-
perparameters, and validate the robustness of our method
to the target metric. Finally, we study the evolution of the
vulnerability score distribution across the training process
throughout training as a metric for model-level risk assess-

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

## References

[^1]: N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramer,
“Membership inference attacks from first principles,” in 2022 IEEE
Symposium on Security and Privacy (SP). IEEE, 2022, pp. 1897–
1914

[^2]: J. Ye, A. Maddi, S. K. Murakonda, V. Bindschaedler, and R. Shokri,
“Enhanced membership inference attacks against machine learning
models,” in Proceedings of the 2022 ACM SIGSAC Conference on
Computer and Communications Security, 2022, pp. 3093–3106.




