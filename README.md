
# DSA4212 Project – Character-Level Language Modeling

## Overview  
This repository contains a transformer-based character-level language model that predicts the next character given a context of length (L):

$$P(x_{t+1} \mid x_{t-L+1}, \dots, x_t)$$

The project explores how different model and training hyperparameters, such as context window size, model depth and width, positional encoding methods, and loss functions, affect overall performance.

## Repo Structure  
```
/
├── data/                                 # text8 dataset files
│   ├── text8_train.txt                   # training corpus (~90M chars)
│   └── text8_test.txt                    # test corpus (~5M chars)
│
├── models/                               # model implementations
│   ├── models_with_pe.py                 # transformer with positional encoding
│   └── __pycache__/                      # compiled Python cache
│
├── util/
│   ├── generation.py                     # text generation utilities
│   └── __pycache__/                      # compiled Python cache
│
├── transformer_final.py                  # final model (best architecture + hyperparameters)
├── transformer_attentionheads.py         # attention head ablation experiments
├── transformer_depth_width.py            # depth vs width experiments
├── transformer_hyperparameter_tuning.py  # Optuna hyperparameter tuning
├── transformer_loss_function.py          # CE vs LS vs Focal loss comparison
├── transformer_pe_test.py                # positional encoding tests
├── transformer_seqlen.py                 # sequence length ablation tests
│
└── README.md                             # project documentation (this file)
```

## Final Model Configuration  
| Parameter              | Value                          |
|------------------------|--------------------------------|
| Vocabulary Size        | `len(char_set)`                |
| Model Dimension (d_model) | 256                        |
| Number of Heads (n_heads) | 8                          |
| Number of Layers (n_layers) | 4                       |
| Maximum Sequence Length (max_len) | 128               |
| Positional Encoding     | Learned                        |
| Loss Function           | Cross Entropy                  |
| Learning Rate           | 0.00092545                     |
| Batch Size              | 128                            |
| Weight Decay            | 0.0252649                      |
| Dropout Rate            | 7.8672e-06                     |

## Final Training Results  
- Final Iterations: 213,970  
- Training Time: ~7,171 seconds  
- Train Loss: 1.2308  
- Test Loss: 1.2351  
- Training Accuracy: 61.5%  
- Test Accuracy: 61.1%  
- Training Last-Character Prediction Accuracy: 67.2%  
- Test Last-Character Prediction Accuracy: 66.4%  

## Key Findings

* Learned positional embeddings outperformed sinusoidal and rotary alternatives.
* Depth proved more beneficial than width for this task (4 layers worked better than wider models).
* Very long contexts (e.g., max_len = 512) resulted in degraded performance within the fixed time budget.
* Cross-entropy loss was more effective than label-smoothing or focal loss under these conditions.

## Group Members  
- Sim Wei
- Yu Zifan
- RaeAnne Choo Chenxi
- Ng Junlin Benjamin
