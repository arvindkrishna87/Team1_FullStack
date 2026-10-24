# Presentation 1 · Mathletes write-up

**Author** Junho Hong (Team1_FullStack)
**Scope** The base code `Presentation1_base_code` written as mathematics, then evaluated. Base code is cited as `file.py Lx–y`.
**My code** `Code1_PatchSizeFilter_22Oct_Junho.py` (patch filter count), `Code2_MathChecks_22Oct_Junho.py` (parameter counts, class-weight check)

---

## 1. Data preprocessing

**Labels** (`config.py VALID_CLASSES`, `data_utils.py L114–116`). Classes 0 and 2 are dropped.

```math
y_c = \begin{cases} 0 & \mathrm{Class}_c = 1 \ \text{(benign)} \\ 1 & \mathrm{Class}_c \in \lbrace 3,4\rbrace \ \text{(high-grade)} \end{cases}
```

Of the cases in the label CSV, 93 have patches, 35 benign and 58 high-grade. This set is $`\mathcal{C}`$.

**Hierarchy as sets.** With stains $`\mathcal{S} = \lbrace \text{H\&E}, \text{Melan-A}, \text{SOX10}\rbrace`$, slices $`\mathcal{T}_{c,s}`$ of case $`c`$ in stain $`s`$, and patches $`\mathcal{P}_{c,s,t}`$ of slice $`t`$,

```math
c = \big\lbrace (s, t, \mathcal{P}_{c,s,t}) \ :\ s \in \mathcal{S},\ t \in \mathcal{T}_{c,s} \big\rbrace .
```

The caps keep the first $`T_{\max} = 5`$ slices per stain (sorted by slice id) and at most $`K = 500`$ patches per slice (`dataset.py L77–78, L112–113`, values from the sbatch files).

**Patch filter** (`precompute_pooled_features.py L66–68`). For patch $`x`$ of height $`h_x`$ and width $`w_x`$ in pixels,

```math
\mathcal{P}_{c,s,t} \leftarrow \lbrace x \in \mathcal{P}_{c,s,t} \ :\ \min(h_x, w_x) \ge 32 \rbrace .
```

Measured with Code1 on `/projects/e32998/patches_benign_split`, **2,003 of 83,630 patches (2.40%)** are removed.

**Standardisation** (`precompute_pooled_features.py L215–221`). For channel $`k \in \lbrace R,G,B\rbrace`$ at pixel $`(u,v)`$, with ImageNet constants,

```math
\tilde x_k(u,v) = \frac{x_k(u,v)/255 - \mu_k}{\sigma_k}, \qquad \mu = (0.485, 0.456, 0.406), \quad \sigma = (0.229, 0.224, 0.225).
```

**Features.** The frozen KimiaNet encoder $`\Phi`$ (DenseNet-121) gives a $`1024 \times h' \times w'`$ map, pooled to $`2\times2`$ and flattened, and is computed once offline.

```math
f(x) = \mathrm{vec}\big(\mathrm{AvgPool}_{2\times2}(\Phi(\tilde x))\big) \in \mathbb{R}^{4096}
```

**Group function and fold constraint** (`config.py GROUPED_CASES`, `make_splits.py L149–153`). Four benign patients were split into pseudo-cases.

```math
g(c) = \begin{cases} 22 & c \in \lbrace 22,107,108\rbrace \\ 26 & c \in \lbrace 26,109,110,111\rbrace \\ 24 & c \in \lbrace 24,118\rbrace \\ 25 & c \in \lbrace 25,119\rbrace \\ c & \text{otherwise} \end{cases} \qquad\quad g(c) = g(c') \Rightarrow \mathrm{fold}(c) = \mathrm{fold}(c')
```

`StratifiedGroupKFold` enforces this, so one patient never appears in both training and test.

---

## 2. Splits

The 60/20/20 target becomes 3 : 1 : 1 over 5 folds $`F_1,\dots,F_5`$ of 18–19 cases, each with 7 benign. Each split trains on three folds, validates on one and tests on one, and the folds rotate so every case is tested once.

| Split | Train (B / H) | Val (B / H) | Test (B / H) |
|---|---|---|---|
| 01 | 57 (21 / 36) | 18 (7 / 11) | 18 (7 / 11) |
| 02 | 56 (21 / 35) | 18 (7 / 11) | 19 (7 / 12) |
| 03 | 55 (21 / 34) | 19 (7 / 12) | 19 (7 / 12) |
| 04 | 55 (21 / 34) | 19 (7 / 12) | 19 (7 / 12) |
| 05 | 56 (21 / 35) | 19 (7 / 12) | 18 (7 / 11) |

The benign share is 0.37–0.39 in every part. Sizes were obtained by running `make_splits.py` on Quest and match the Code walkthrough (slide 9).

---

## 3. Tensor transformations

One case per step (batch size 1). For a slice with $`P \le K`$ patches, the model computes the following (`models.py`).

```math
X_{s,t} \in \mathbb{R}^{P\times4096} \ \xrightarrow{\ \text{projection}\ }\ H_{s,t} = \mathrm{Dropout}\big(\mathrm{ReLU}(X_{s,t}W_p^\top + b_p)\big) \in \mathbb{R}^{P\times512}
```

**Gated attention** (Ilse et al. 2018, Eq. 9; `models.py L8–48`). For instances $`h_1,\dots,h_M \in \mathbb{R}^{512}`$,

```math
e_i = w^\top\big[\tanh(Vh_i + b_V) \odot \mathrm{sigm}(Uh_i + b_U)\big] + b_w, \qquad a_i = \frac{e^{e_i}}{\sum_j e^{e_j}}, \qquad \mathrm{Attn}(h_{1:M}) = \sum_i a_i h_i ,
```

with $`V, U \in \mathbb{R}^{128\times512}`$, $`b_V, b_U, w \in \mathbb{R}^{128}`$, $`b_w \in \mathbb{R}`$. The weights $`a_i`$ are non-negative and sum to 1, so the output is a weighted average. ($`b_w`$ shifts every score equally and cancels in the softmax.)

**Three levels**, each with its own attention module.

```math
\underbrace{H_{s,t}}_{P\times512} \xrightarrow{\mathrm{Attn}_{\text{patch}}} \underbrace{z_{s,t}}_{512} \quad\Big|\quad \underbrace{[z_{s,1};\dots;z_{s,T}]}_{T\times512} \xrightarrow{\mathrm{Attn}_{\text{slice}}} \underbrace{z_s}_{512} \quad\Big|\quad \underbrace{[z_{s_1};\dots;z_{s_S}]}_{S\times512} \xrightarrow{\mathrm{Attn}_{\text{stain}}} \underbrace{z_c}_{512}
```

```math
\ell = W_c\,\mathrm{Dropout}(z_c) + b_c \in \mathbb{R}^2, \qquad p = \mathrm{softmax}(\ell), \qquad \hat y = \arg\max_k p_k
```

**Train vs. validation vs. prediction**

| | Training | Validation | Prediction |
|---|---|---|---|
| Case order | shuffled | fixed | fixed |
| Patches per slice | random $`K`$, redrawn each epoch | first $`K`$ by file name | first $`K`$ by file name |
| Dropout | on | off | off |
| Gradients | on, update after every case | off | off |
| Output | loss $`\mathcal{L}_c`$ (Section 5) | CE, accuracy | $`p`$, $`\hat y`$, accuracy |
| Weights | current | current | last epoch (Issue 4) |

---

## 4. Parameters

| Module | Formula | Count |
|---|---|---|
| Patch projector | $`4096\cdot512 + 512`$ | 2,097,664 |
| Gated attention (×3) | $`3\,[\,2(512\cdot128+128) + 129\,]`$ | 394,371 |
| Classifier | $`512\cdot2 + 2`$ | 1,026 |
| **Total, all trained** | | **2,493,061** |

KimiaNet adds 6,953,856 frozen parameters that are not estimated. Both numbers were reproduced with Code2. With about 56 training cases per split, this is about 44,500 trained parameters per training case.

**Tuned by Optuna** learning rate $`\eta = 2.08\times10^{-4}`$, weight decay $`\lambda_{wd} = 3.05\times10^{-4}`$, dropout $`0.2599`$, benign class weight $`w(0) = 2.5285`$ (`config.py` "from OPTUNA study"). The entropy weight $`\lambda`$ is in the search space, but the config holds the round value $`10^{-3}`$.

**Hard-coded** embedding size 512, attention size 128, caps $`K = 500`$ and $`T_{\max} = 5`$, 30 epochs, batch size 1, LR factor 0.2139 with patience 4 and floor $`10^{-6}`$, early-stopping patience 8, min delta $`10^{-3}`$, min 10 epochs, normalised max-entropy mode, 32 px filter, $`2\times2`$ pooling, seed 42.

---

## 5. Loss function

PyTorch's weighted cross-entropy with `reduction='mean'` (`trainer.py L69`) divides by the batch's total weight. The training loss adds a patch-attention entropy bonus (`trainer.py L159–205, L259–268`).

```math
\mathcal{L}_c = \frac{w(y_c)\,[-\log p(y_c\mid c)]}{w(y_c)} \;-\; \lambda\,\bar H_c, \qquad \bar H_c = \mathrm{mean}_{(s,t)} \frac{-\sum_{i=1}^{P} a_i \log a_i}{\log P}, \qquad \lambda = 10^{-3}
```

$`\bar H_c \in [0,1]`$ is the entropy of the patch attention, divided by its maximum $`\log P`$ and averaged over the case's slices. Subtracting it rewards attention spread over many patches. Weight decay is applied by AdamW in the update step, so it does not appear in $`\mathcal{L}_c`$.

---

## 6. Regularisers

| Regulariser | Rule |
|---|---|
| Dropout ($`p = 0.2599`$) | $`\tilde h = m \odot h/(1-p)`$, $`m_j \sim \mathrm{Bernoulli}(1-p)`$, on patch embeddings and on $`z_c`$, training only |
| Weight decay (AdamW) | $`\theta_t = \theta_{t-1} - \eta\big(\hat m_t/(\sqrt{\hat v_t}+\epsilon) + \lambda_{wd}\,\theta_{t-1}\big)`$ |
| Entropy bonus | $`-\lambda \bar H_c`$ in the loss (Section 5) |
| Early stopping | stop at epoch $`e \ge 10`$ once validation loss has not dropped by more than $`10^{-3}`$ for 8 epochs, else at 30 |
| LR on plateau | $`\eta \leftarrow \max(0.2139\,\eta,\ 10^{-6})`$ after more than 4 epochs without improvement |
| Patch subsampling | a new random set of $`K`$ patches per slice each epoch |
| Frozen encoder | KimiaNet's 6.95 M parameters are not trained |

---

## 7. Critical evaluation

### Issue 1. Hyperparameters were tuned on the test cases
The Optuna study (`optuna_training.py L381–420, L550–560`) minimises the mean validation loss over all five splits. Their validation folds are $`F_1,\dots,F_5`$, whose union is $`\mathcal{C}`$, so every test case was also used to choose the hyperparameters in `config.py`. This biases the reported accuracy upward (Varma and Simon 2006, Cawley and Talbot 2010). The size of the bias cannot be computed from the code.

**Fix** Nested cross-validation, tuning only on each split's training folds.

### Issue 2. The class weight cancels at batch size 1
In Section 5, $`w(y_c)`$ appears in both the numerator and the denominator, so $`\mathcal{L}_c = -\log p(y_c\mid c) - \lambda\bar H_c`$. The weight 2.5285 changes neither the loss nor its gradient, and Optuna tuned a parameter with no effect. Code2 confirms this (0.474077 with and without weights). A larger `batch_size` alone does not help because `trainer.py L259` uses only `batch[0]`.

**Fix** `reduction='sum'`, or divide by a fixed constant, or oversample benign cases (Buda et al. 2018).

### Issue 3. Two regularisers are too small to act
*Entropy.* Normalisation gives $`0 \le \lambda\bar H_c \le 10^{-3}`$. AEM (Zhang et al.), the method this term follows, uses the unnormalised entropy (up to $`\log 500 \approx 6.2`$) with $`\lambda \in [0.005, 0.1]`$, a maximum bonus of $`0.03`$ to $`0.6`$. That is 30–600 times stronger.

*Weight decay.* Each step multiplies a weight by $`1 - \eta\lambda_{wd}`$, with $`\eta\lambda_{wd} = 2.08\times10^{-4}\times3.05\times10^{-4} \approx 6.3\times10^{-8}`$. Over at most $`56\times30 \approx 1{,}700`$ steps the total shrinkage is about $`1.1\times10^{-4}`$, or 0.01%. In AdamW the decay is scaled by $`\eta`$ (Loshchilov and Hutter 2019), so the Optuna range $`[10^{-6}, 2\times10^{-3}]`$ cannot reach an effective value.

With about 44,500 parameters per training case, these safeguards matter.

**Fix** Use the unnormalised entropy or a larger $`\lambda`$, raise the weight-decay range by one to two orders of magnitude, and re-tune.

### Issue 4. The test set is scored with last-epoch weights
`trainer.py L370–378` saves `best.pth` at the lowest validation loss, but `main.py L289–305` calls `evaluate()` right after `train()` without loading it. Whenever early stopping fires, the weights used are 8 epochs past the best. Early stopping is meant to return the best weights (Prechelt).

**Fix** `trainer.load_checkpoint(best.pth)` before `evaluate()`.

### Measuring the 95% target
One test fold has 18–19 cases, so one error moves accuracy by $`1/19 \approx 5.3`$ points and a fold reaches 95% only with zero errors. Pooled over all 93 cases, the current overall accuracy of 89% (project description slides) corresponds to $`83/93`$, whose 95% Wilson interval is **[81.3%, 94.1%]**. This agrees with Varoquaux (2018), who reports errors of roughly ±10% for cross-validated accuracy on about 100 samples. With 62% high-grade cases, accuracy alone also hides benign performance.

**Recommendation** Report pooled accuracy with a Wilson interval, plus sensitivity and specificity.

---

## References
1. Ilse, Tomczak, Welling (2018). Attention-based Deep Multiple Instance Learning. ICML.
2. Riasatian et al. (2021). KimiaNet. Medical Image Analysis 70, 102032.
3. Varma, Simon (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7, 91.
4. Cawley, Talbot (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. JMLR 11.
5. Buda, Maki, Mazurowski (2018). A systematic study of the class imbalance problem in CNNs. Neural Networks 106.
6. Zhang et al. AEM: Attention Entropy Maximization for MIL-based WSI classification. arXiv:2406.15303.
7. Loshchilov, Hutter (2019). Decoupled Weight Decay Regularization. ICLR.
8. Prechelt. Early Stopping, But When? Neural Networks: Tricks of the Trade.
9. Varoquaux (2018). Cross-validation failure. NeuroImage 180.
10. Wilson (1927). Probable inference. JASA 22.
11. STAT390 instructional team. `Presentation1_base_code` and Code walkthrough slides (Fall 2026).
12. AI use. Claude (Anthropic) helped read the code and formulas, and find reference documents.
