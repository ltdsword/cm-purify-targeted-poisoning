# Consistency Model for Purifying Data Poisoning

This research proposes a fast, poison-aware data purification framework based on a one-step pixel-space Consistency Model (CM). The goal is to sanitize an untrusted training dataset before downstream model training, while preserving the semantic content and clean labels of benign samples.

The method is designed for targeted clean-label data poisoning attacks, especially:

- **Witches' Brew / Gradient Matching**: a clean-label targeted poisoning attack for training-from-scratch settings.
- **Bullseye Polytope**: a clean-label feature-space poisoning attack for transfer-learning settings.

The central idea is to train a purifier that maps a noised and potentially poisoned image back to its clean counterpart in one neural function evaluation. This keeps the generative purification spirit of diffusion defenses, but avoids long iterative denoising chains.

---

## Motivation

Diffusion-based purification can remove small adversarial or poisoning perturbations by adding noise and then denoising the image toward the natural image manifold. However, standard diffusion purification usually requires many denoising steps per image:

$$
x_t \rightarrow x_{t-1} \rightarrow \cdots \rightarrow x_0.
$$

This becomes expensive when the defender must sanitize an entire training dataset before training the victim model.

Consistency Models are attractive because they learn a direct mapping from a noised state to the clean data point:

$$
f_\theta(x_t, t) \approx x_0.
$$

After training, purification can be performed with one neural function evaluation:

$$
\hat{x}_0 = f_\theta(x_{t^*}, t^*).
$$

Therefore, the intended trade-off is:

> Diffusion purification: many denoising steps  
> Consistency purification: one denoising step

---

## Threat Model

We focus on targeted clean-label poisoning attacks. The attacker injects a small fraction of imperceptibly perturbed training samples into the training set. The labels remain correct, but the poisoned samples cause a specific target test image to be misclassified after victim training.

### Attacker objective

Given a clean target image $x^t$, the attacker wants the victim model trained on the poisoned dataset to classify $x^t$ as an adversarial class $y^{adv}$, while maintaining normal clean test accuracy.

### Attacker capability

The attacker can inject or modify a small number of training samples under a bounded perturbation constraint, for example:

$$
\|\Delta_i\|_\infty \le \epsilon.
$$

The attack is clean-label, so the visible image content and label remain consistent.

### Defender capability

The defender controls the training pipeline and can preprocess the dataset before training the victim model. The defender does not know which samples are poisoned, which target was selected, or which attack algorithm was used. Therefore, purification is applied blindly to all training images.

---

## Attacks Considered

### 1. Witches' Brew / Gradient Matching

Witches' Brew is a targeted clean-label poisoning attack that works even when the victim model is trained from scratch. The key mechanism is gradient matching.

The attack tries to make the average gradient produced by poisoned samples align with the gradient of the target image under the adversarial label:

$$
\nabla_\theta \mathcal{L}(F(x^t, \theta), y^{adv})
\approx
\frac{1}{P}\sum_{i=1}^{P}
\nabla_\theta \mathcal{L}(F(x_i + \Delta_i, \theta), y_i).
$$

Instead of matching exact gradient magnitudes, Witches' Brew maximizes cosine similarity between the target gradient and the poison gradient direction:

$$
B(\Delta, \theta)
=
1 -
\frac{
\left\langle
\nabla_\theta \mathcal{L}(F(x^t, \theta), y^{adv}),
\sum_{i=1}^{P} \nabla_\theta \mathcal{L}(F(x_i + \Delta_i, \theta), y_i)
\right\rangle
}{
\left\|\nabla_\theta \mathcal{L}(F(x^t, \theta), y^{adv})\right\|
\cdot
\left\|\sum_{i=1}^{P} \nabla_\theta \mathcal{L}(F(x_i + \Delta_i, \theta), y_i)\right\|
}.
$$

Differentiable data augmentation is used during poison crafting so that the perturbations survive random crops, flips, translations, and standard training pipelines.

### 2. Bullseye Polytope

Bullseye Polytope is designed mainly for transfer learning. Instead of matching gradients, it manipulates feature representations.

Given substitute feature extractors $\phi^{(i)}$, a target image $x^t$, and $k$ clean base images $x_b^{(j)}$, Bullseye Polytope crafts poisoned samples $x_p^{(j)}$ so that the target representation is close to the mean poison representation:

$$
\min_{\{x_p^{(j)}\}}
\frac{1}{2m}
\sum_{i=1}^{m}
\frac{
\left\|
\phi^{(i)}(x^t) - \frac{1}{k}\sum_{j=1}^{k}\phi^{(i)}(x_p^{(j)})
\right\|_2^2
}{
\left\|\phi^{(i)}(x^t)\right\|_2^2
}
$$

subject to:

$$
\|x_p^{(j)} - x_b^{(j)}\|_\infty \le \epsilon.
$$

By centering the target inside the poison feature cluster, the attack improves transferability and robustness to feature-space shifts.

---

## Proposed Method: Pixel-Space Consistency Purification

We train a pixel-space purifier $P_\theta$ that maps an untrusted image to a purified image:

$$
\hat{x} = P_\theta(x_{untrusted}).
$$

The purifier is trained using an offline paired dataset of clean images and their corresponding poisoned versions. The key design choice is to treat poison perturbations as structured noise inside a consistency-model forward process.

The pipeline has three components:

1. **Offline poison-pair dataset** containing clean-poison pairs.
2. **Poison-aware forward process** that injects poison residuals as structured corruption.
3. **One-step consistency purifier** trained to recover the clean image directly.

---

## Offline Poison-Pair Dataset

Let the clean CIFAR-10 training set be:

$$
D_{clean} = \{(x_{clean}^{(i)}, y^{(i)})\}_{i=1}^{N}.
$$

We generate a paired purifier-training dataset:

$$
D_{pair}
=
\left\{
\left(x_{clean}^{(i)}, x_{poison}^{(i)}, y^{(i)}, m^{(i)}\right)
\right\}_{i=1}^{M},
$$

where:

- $x_{clean}^{(i)}$ is the original clean image,
- $x_{poison}^{(i)}$ is the poisoned counterpart,
- $y^{(i)}$ is the clean label,
- $m^{(i)}$ stores metadata such as attack type, target class, poison class, random seed, and perturbation bound.

The poison residual is:

$$
\delta^{(i)} = x_{poison}^{(i)} - x_{clean}^{(i)}.
$$

The paired dataset contains:

- Witches' Brew poison-clean pairs,
- Bullseye Polytope poison-clean pairs,
- clean identity pairs where $x_{poison} = x_{clean}$.

The clean identity pairs are important because the purifier will be applied blindly to every image. It should learn to preserve benign images instead of unnecessarily modifying them.

Target training size:

```text
10,000 Witches' Brew pairs
10,000 Bullseye Polytope pairs
10,000 clean identity pairs
--------------------------------
30,000 total purifier-training pairs
```

This poison bank is not meant to represent one single victim dataset. Instead, it is a controlled training source for learning general poison-removal behavior.

---

## CIFAR-10 Manifest Allocator

CIFAR-10 contains 5,000 training images per class. For each class $c$, all indices are shuffled with a fixed random seed and allocated into non-overlapping blocks:

| Index range after shuffle | Usage |
|---|---|
| `[0, 499]` | Witches' Brew training case 1 |
| `[500, 999]` | Witches' Brew training case 2 |
| `[1000, 1499]` | Witches' Brew evaluation case |
| `[1500, 2499]` | Bullseye Polytope training pool |
| `[2500, 2519]` | Bullseye Polytope evaluation pool |
| `[2520, 3519]` | Clean identity training pool |
| `[3520, 4999]` | Reserve and target-selection pool |

This split prevents leakage between purifier-training pairs and held-out poison evaluation cases.

Pair counts:

```text
Witches' Brew:       10 classes x 2 blocks x 500 images = 10,000 pairs
Bullseye Polytope:   10 classes x 100 groups x 10 images = 10,000 pairs
Clean identity:      10 classes x 1,000 images = 10,000 pairs
```

---

## Poison-Aware Forward Process

A standard Consistency Model satisfies the boundary condition:

$$
f_\theta(x, 0) = x.
$$

If we directly use $x_{poison}$ as the root of the noising process, then near $t = 0$, the model is encouraged to reconstruct the poisoned image. However, the desired output is $x_{clean}$.

To avoid this conflict, the proposed method uses $x_{clean}$ as the root and treats the poison residual as a structured noise component.

Let:

$$
\delta = x_{poison} - x_{clean}.
$$

The poison-aware forward process is:

$$
x_t^*
=
\sqrt{\bar{\alpha}_t}x_{clean}
+
\sqrt{1 - \bar{\alpha}_t}(\epsilon + \gamma_a\delta),
\quad
\epsilon \sim \mathcal{N}(0, I),
$$

where:

- $a \in \{WB, BP, clean\}$ is the example type,
- $\gamma_a$ controls the strength of the poison residual.

As $t \rightarrow 0$:

$$
\sqrt{1 - \bar{\alpha}_t} \rightarrow 0,
$$

so both Gaussian noise and poison residual vanish:

$$
x_t^* \rightarrow x_{clean}.
$$

This naturally satisfies:

$$
f_\theta(x_{clean}, 0) = x_{clean}.
$$

The model is trained to predict:

$$
\hat{x}_0 = f_\theta(x_t^*, t) \approx x_{clean}.
$$

---

## Training Objectives

The purifier uses a combination of consistency, reconstruction, semantic, and identity-preservation losses.

### 1. Consistency distillation loss

The student and EMA teacher should map different points on the same trajectory to the same clean origin:

$$
\mathcal{L}_{distill}
=
d\left(
 f_\theta(x^*_{t_{n+1}}, t_{n+1}),
 \operatorname{sg}\left[f_{\theta^-}(\hat{x}^{\Psi}_{t_n}, t_n)\right]
\right),
$$

where $\operatorname{sg}[\cdot]$ stops gradients and $\Psi$ is an ODE solver used to estimate a shallower timestep.

### 2. Clean reconstruction loss

The model directly learns to reconstruct the clean image:

$$
\mathcal{L}_{rec} = \|\hat{x}_0 - x_{clean}\|_1.
$$

### 3. Label preservation loss

A frozen clean classifier $C$ can be used to reduce semantic drift:

$$
\mathcal{L}_{cls} = CE(C(\hat{x}_0), y).
$$

### 4. Identity loss

For clean identity pairs, $\delta = 0$. The purifier should preserve clean samples:

$$
\mathcal{L}_{id}
=
\|f_\theta(x_t^{clean}, t) - x_{clean}\|_1,
$$

where:

$$
x_t^{clean}
=
\sqrt{\bar{\alpha}_t}x_{clean}
+
\sqrt{1 - \bar{\alpha}_t}\epsilon.
$$

### Full objective

$$
\mathcal{L}_{total}
=
\lambda_1\mathcal{L}_{distill}
+
\lambda_2\mathcal{L}_{rec}
+
\lambda_3\mathcal{L}_{cls}
+
\lambda_4\mathcal{L}_{id}.
$$

---

## Algorithm 1: Offline Poison-Pair Dataset Generation

```text
Input:
  Clean CIFAR-10 dataset D_clean
  Witches' Brew generator A_WB
  Bullseye Polytope generator A_BP
  Perturbation bound epsilon
  Fixed random seed

Output:
  Paired purifier-training dataset D_pair
  Held-out poison evaluation cases

1. Initialize D_pair = empty set.
2. For each class c in {0, ..., 9}:
   a. Collect all training indices with label c.
   b. Shuffle the class indices using the fixed random seed.
   c. Allocate non-overlapping blocks for WB, BP, clean identity, evaluation, and reserve.

3. For each Witches' Brew training block of 500 base images:
   a. Select a target image from a different class.
   b. Set adversarial label to the poison/base class.
   c. Generate poisoned images using A_WB.
   d. Store (x_clean, x_poison, y, metadata) in D_pair.

4. Split the Bullseye Polytope training pool into groups of 10 base images.
5. For each Bullseye Polytope group:
   a. Select a target image from a different class.
   b. Generate poisoned images using A_BP.
   c. Store (x_clean, x_poison, y, metadata) in D_pair.

6. For each clean identity image:
   a. Store (x_clean, x_clean, y, clean_metadata) in D_pair.

7. Save held-out WB and BP evaluation cases separately.
8. Return D_pair.
```

---

## Algorithm 2: Poison-Aware Consistency Distillation

```text
Input:
  Offline paired dataset D_pair
  Student purifier f_theta
  EMA teacher f_theta_minus
  Frozen classifier C
  ODE solver Psi
  Timestep schedule {t_n}

Output:
  Trained purifier P_theta

1. While not converged:
   a. Sample minibatch {(x_clean, x_poison, y, a)} from D_pair.
   b. Compute poison residual delta = x_poison - x_clean.
   c. Sample timestep t = t_{n+1}.
   d. Sample Gaussian noise epsilon ~ N(0, I).
   e. Construct poison-aware noised input x_t^*.
   f. Predict clean image x_hat_0 = f_theta(x_t^*, t).
   g. Estimate shallower teacher input using solver Psi.
   h. Compute consistency distillation loss.
   i. Compute reconstruction loss.
   j. Compute label preservation loss.
   k. Compute identity loss for clean identity samples.
   l. Combine losses into L_total.
   m. Update student parameters.
   n. Update EMA teacher parameters.

2. Return trained purifier P_theta.
```

---

## Algorithm 3: Inference-Time Dataset Sanitization

```text
Input:
  Untrusted dataset D_untrusted = {(x_untrusted, y)}
  Trained purifier f_theta
  Target timestep t^*

Output:
  Sanitized dataset D_san

1. Initialize D_san = empty set.
2. For each minibatch from D_untrusted:
   a. Sample Gaussian noise epsilon ~ N(0, I).
   b. Add forward diffusion noise up to t^*:

      x_{t^*} = sqrt(alpha_bar_{t^*}) x_untrusted
                + sqrt(1 - alpha_bar_{t^*}) epsilon

   c. Perform one-step purification:

      x_hat = f_theta(x_{t^*}, t^*)

   d. Store (x_hat, y) in D_san.

3. Train downstream victim model on D_san.
4. Return D_san.
```

---

## Evaluation Protocol

The evaluation cases are generated offline but kept disjoint from the purifier-training pairs. This is important because data poisoning is a train-time attack: the attacker poisons training images, not the official test images.

### Witches' Brew evaluation

For each class, one held-out block of 500 base images is used to generate a fresh Witches' Brew poisoning case. The victim model is trained from scratch on the resulting dataset.

Compare:

1. Training on the clean dataset.
2. Training on the poisoned dataset.
3. Training on the purified poisoned dataset.

The same victim initialization seed and training schedule should be used across all three conditions.

### Bullseye Polytope evaluation

For each class, two held-out groups of 10 base images are used to generate fresh Bullseye Polytope poisons. The primary evaluation should use a transfer-learning setting because Bullseye Polytope is designed around feature-space poisoning.

Compare:

1. Training on the clean dataset.
2. Training on the poisoned dataset.
3. Training on the purified poisoned dataset.

### Test set usage

The official CIFAR-10 test set remains untouched. It is used only after downstream victim training to measure:

- clean test accuracy,
- target misclassification rate,
- adversarial-label confidence.

---

## Metrics

Recommended metrics:

- Attack Success Rate (ASR) before purification.
- Attack Success Rate (ASR) after purification.
- Clean test accuracy.
- Target confidence for the adversarial label.
- Average purification time per image.
- Total dataset sanitization time.
- Image distortion metrics, such as PSNR, SSIM, LPIPS, or $\ell_p$ distance.
- Optional: feature-space distance between clean and purified images under a frozen classifier.

---

## Expected Advantages

- **Fast inference**: one neural function evaluation instead of many diffusion denoising steps.
- **Dataset-level scalability**: purification can be batched on GPU.
- **Attack diversity**: purifier training uses both Witches' Brew and Bullseye Polytope poisons.
- **Clean-image preservation**: identity pairs reduce unnecessary modification of benign samples.
- **Attack-agnostic deployment**: the trained purifier is applied blindly to all images.

---

## Key Risks and Limitations

This method is promising but not guaranteed to work without careful validation. The main risks are:

1. **Distribution mismatch**  
   The purifier is trained on generated WB/BP poison distributions. It may not generalize to stronger adaptive poisons or unseen poison mechanisms.

2. **Clean accuracy degradation**  
   If the timestep $t^*$ or residual strength is too large, the purifier may remove useful semantic details and reduce downstream accuracy.

3. **Residual scaling mismatch**  
   During training, the model sees the exact residual $\delta = x_{poison} - x_{clean}$. At inference, it receives only an untrusted image and Gaussian noise. The deployment distribution may differ from the training corruption distribution.

4. **Adaptive attack risk**  
   If the attacker knows the purifier, they may craft poisons that survive the CM transformation.

5. **CIFAR-10 resolution**  
   CIFAR-10 images are small, so aggressive purification may visibly alter semantic content. Timestep and loss weights must be tuned carefully.

6. **Paired poison generation cost**  
   Training the purifier requires expensive offline generation of many poison-clean pairs.

---

## Recommended Experimental Roadmap

### Stage 1: Sanity check on synthetic residuals

Before generating expensive WB/BP poisons, train the CM purifier on clean images with synthetic bounded perturbations. Verify that:

- clean images are preserved,
- small perturbations are removed,
- classifier accuracy does not collapse.

### Stage 2: Small WB/BP pilot

Generate a small poison bank:

```text
1,000 WB pairs
1,000 BP pairs
1,000 clean identity pairs
```

Train a small U-Net CM and evaluate reconstruction quality and clean-image preservation.

### Stage 3: Full poison bank

Scale to the planned 30,000-pair dataset:

```text
10,000 WB pairs
10,000 BP pairs
10,000 clean identity pairs
```

Tune:

- $t^*$,
- $\gamma_{WB}$,
- $\gamma_{BP}$,
- $\lambda_1, \lambda_2, \lambda_3, \lambda_4$,
- EMA decay,
- timestep schedule.

### Stage 4: Defense evaluation

Evaluate on held-out poison cases not used for purifier training:

- WB, training from scratch.
- BP, transfer learning.
- Optional: black-box victim architecture.
- Optional: adaptive attacker aware of the purifier.

---

## Suggested Repository Structure

```text
.
├── README.md
├── configs/
│   ├── cifar10_manifest.yaml
│   ├── cm_train.yaml
│   ├── wb_attack.yaml
│   └── bp_attack.yaml
├── data/
│   ├── manifests/
│   ├── poison_pairs/
│   └── eval_cases/
├── attacks/
│   ├── witches_brew/
│   └── bullseye_polytope/
├── purifier/
│   ├── model.py
│   ├── forward_process.py
│   ├── losses.py
│   ├── train.py
│   └── infer.py
├── evaluation/
│   ├── train_victim.py
│   ├── eval_asr.py
│   └── eval_clean_accuracy.py
└── scripts/
    ├── generate_manifest.py
    ├── generate_poison_pairs.py
    ├── train_purifier.sh
    ├── purify_dataset.sh
    └── evaluate_defense.sh
```

---

## Minimal Usage Plan

### 1. Generate manifest

```bash
python scripts/generate_manifest.py \
  --dataset cifar10 \
  --seed 2026 \
  --out data/manifests/cifar10_manifest.json
```

### 2. Generate poison-clean pairs

```bash
python scripts/generate_poison_pairs.py \
  --manifest data/manifests/cifar10_manifest.json \
  --attack wb,bp \
  --out data/poison_pairs/
```

### 3. Train purifier

```bash
python purifier/train.py \
  --config configs/cm_train.yaml \
  --pair-dir data/poison_pairs/ \
  --out checkpoints/cm_purifier.pt
```

### 4. Purify untrusted training set

```bash
python purifier/infer.py \
  --checkpoint checkpoints/cm_purifier.pt \
  --input data/untrusted_train/ \
  --output data/sanitized_train/ \
  --t-star 0.25
```

### 5. Train and evaluate victim

```bash
python evaluation/train_victim.py \
  --train data/sanitized_train/ \
  --arch resnet18 \
  --out checkpoints/victim_sanitized.pt

python evaluation/eval_asr.py \
  --model checkpoints/victim_sanitized.pt \
  --eval-cases data/eval_cases/
```

---

## Research Claim

The intended research contribution is:

> A one-step pixel-space Consistency Model trained on offline poison-clean pairs to purify targeted clean-label poisoning perturbations before victim training.

More specifically, the method differs from standard diffusion denoising and prior generative poison defenses by combining:

1. train-time data poison purification,
2. one-step CM inference,
3. explicit paired WB/BP poison-clean training,
4. poison residuals modeled as structured noise in the forward process,
5. clean identity preservation for blind dataset-wide sanitization.

---

## References

- Jonas Geiping et al. **Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching.** ICLR 2021.
- Hojjat Aghakhani et al. **Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability.** IEEE EuroS&P 2021.
- Yang Song et al. **Consistency Models.** ICML 2023.
- Chun Tong Lei et al. **Instant Adversarial Purification with Adversarial Consistency Distillation.** CVPR 2025.
- Sunay Bhat et al. **PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics.** NeurIPS 2024.
- Sanghyun Hong et al. **Certified Robustness to Clean-Label Poisoning Using Diffusion Denoising.** 2024.
- Omead Pooladzandi et al. **PureEBM: Universal Poison Purification via Mid-Run Dynamics of Energy-Based Models.** 2024.
