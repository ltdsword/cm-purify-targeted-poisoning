# Narcissus Experiment Specification for CM Purification

## PureGen-Aligned CIFAR-10 Baseline

**Status:** implementation specification  
**Primary attack:** Narcissus (NS)  
**Dataset:** CIFAR-10  
**Victim model:** ResNet-18, trained from scratch  
**Poison setting:** 1% of the full CIFAR-10 training set  
**Trigger bound:** full-image 32x32 trigger, \(\epsilon = 8/255\)  
**Number of primary evaluation cases:** 10, one target class per CIFAR-10 class

---

## 1. Purpose

This document defines how to add Narcissus to the current Consistency Model (CM) poison-purification project while keeping the **evaluation protocol aligned as closely as practical with the PureGen Narcissus baseline**.

There are two separate datasets/protocols and they MUST NOT be confused:

1. **CM purifier-training poison bank**
   - Purpose: teach the CM to remove Narcissus-like trigger residuals.
   - Size: 10,000 poison-clean pairs.
   - This is an internal supervised training construction for the purifier.
   - It does not need to itself form a valid victim-model Narcissus attack dataset.

2. **Held-out Narcissus attack evaluation cases**
   - Purpose: test whether the trained CM actually neutralizes a valid Narcissus attack after downstream victim retraining.
   - Number: 10 cases, one target class per CIFAR-10 class.
   - Each case is a real clean-label Narcissus training attack: 500 target-class training images are patched, labels are unchanged, and the total victim training dataset remains 50,000 images.

The main defense claim is based on the second protocol, not merely on image reconstruction quality.

---

## 2. What is taken directly from the PureGen baseline

PureGen Appendix D.1.3 defines its CIFAR-10 from-scratch Narcissus setting as follows:

- synthesize one Narcissus trigger **per class**;
- use a full 32x32 trigger;
- constrain the trigger with \(\epsilon = 8/255\);
- apply the trigger to 500 CIFAR-10 training images;
- 500 / 50,000 = 1% total-dataset poisoning;
- evaluate on a patched test dataset;
- do **not** use the test-time trigger multiplier used by the original Narcissus paper.

PureGen reports Narcissus using:

- average poison success / ASR across classes;
- maximum poison success / ASR across classes;
- average natural accuracy.

PureGen's defense pipeline is dataset preprocessing: the poisoned training dataset is purified first, then an otherwise standard classifier is trained. The classifier inference pipeline itself is not changed by the purifier.

### Important interpretation for test ASR

The PureGen appendix says it tests on the patched dataset, but does not state an exact test-subset count in that sentence. The original Narcissus protocol defines all-to-one ASR over **all non-target-class test examples**. Therefore, the most baseline-faithful CIFAR-10 implementation is:

- official CIFAR-10 test set: 10,000 images;
- for target class \(c\), exclude its 1,000 test images;
- patch the remaining 9,000 non-target test images using the same trigger \(\delta_c\);
- compute ASR over those 9,000 images.

This document therefore uses **9,000 triggered test queries per case** for the primary experiment.

A 500-image random non-target subset may be implemented as a fast smoke-test mode, but it should not be reported as the primary PureGen-aligned result.

---

## 3. Corrections to the initial proposed design

The proposed design is mostly correct, with the following changes.

### 3.1 Correct: 10 evaluation cases

Use exactly 10 primary Narcissus cases:

| Case | Target class |
|---|---|
| NS-00 | airplane |
| NS-01 | automobile |
| NS-02 | bird |
| NS-03 | cat |
| NS-04 | deer |
| NS-05 | dog |
| NS-06 | frog |
| NS-07 | horse |
| NS-08 | ship |
| NS-09 | truck |

Each case uses an independently synthesized evaluation trigger for its class.

### 3.2 Correct: 500 poisoned training images + 49,500 normal images

For target class \(c\):

- select 500 images from the **5,000 CIFAR-10 training images whose label is \(c\)**;
- apply the class-specific trigger \(\delta_c\);
- keep the original class label \(c\);
- replace those 500 clean images with their poisoned counterparts.

The resulting victim training dataset contains exactly:

```text
500 poisoned target-class images
+ 49,500 unchanged training images
= 50,000 total training images
```

Do **not** append the poison images to the original 50,000; the poisoned images replace their clean versions.

Because CIFAR-10 has 5,000 training images per class, this is:

- 1% of the full training dataset;
- 10% of the target-class training samples.

### 3.3 Change: do not use 100 or 500 test queries as the primary result

For the main PureGen-aligned experiment, use all **9,000 non-target official CIFAR-10 test images** for each target class.

Optional development mode:

```text
--eval-mode smoke
--num-triggered-test 500
```

Primary reporting mode:

```text
--eval-mode full
# automatically uses all 9,000 non-target test images
```

### 3.4 Terminology change

For Narcissus, do not call the patched test images "target images" in code or documentation. That wording is natural for WB/BP but misleading for an all-to-one backdoor attack.

Use:

- `target_class`: the adversarial class the trigger points to;
- `triggered_test_images` or `attack_queries`: non-target test images patched with the trigger.

---

## 4. Core Narcissus notation

For CIFAR-10 target class \(c\), synthesize a full-image trigger:

\[
\delta_c, \qquad \|\delta_c\|_\infty \le 8/255.
\]

Apply it to an image using:

\[
A_c(x) = \operatorname{clip}(x + \delta_c, 0, 1).
\]

No test-time multiplier is used in the PureGen-aligned protocol:

\[
A_c^{test}(x) = \operatorname{clip}(x + \delta_c, 0, 1).
\]

The same evaluation trigger \(\delta_c\) must be used for:

1. poisoning the 500 target-class training images of case \(c\); and
2. patching the non-target test queries used to measure ASR for case \(c\).

---

# Part A — 10,000 Narcissus Pairs for CM Training

## 5. Goal of the CM Narcissus training bank

The CM is trained as an image purifier. For each pair it learns approximately:

\[
P_\theta(x_{poison}) \rightarrow x_{clean}.
\]

For Narcissus, the purifier should ideally learn to remove a class-oriented trigger regardless of the semantic content underneath it.

Therefore, the 10,000-pair CM training bank deliberately uses class-specific Narcissus triggers across diverse image classes.

This is a **purifier-training augmentation**, not a claim that every resulting pair is a valid clean-label Narcissus victim-training poison.

---

## 6. Training-trigger generation

Generate 10 training triggers:

```text
train_trigger_0  -> target airplane
train_trigger_1  -> target automobile
...
train_trigger_9  -> target truck
```

Each trigger must be synthesized using the Narcissus generation method with:

```text
epsilon: 8/255
shape:   3 x 32 x 32
scope:   full image
```

### Critical leakage rule

The evaluation trigger for class \(c\) must NOT be the same tensor as the CM-training trigger for class \(c\).

Use conceptually:

```text
delta_train[c] != delta_eval[c]
```

The evaluation trigger should be regenerated independently using a separate trigger-generation seed and preferably a disjoint target-class synthesis subset.

Reason: if the CM is trained on the exact trigger tensor later used for evaluation, the experiment may test trigger memorization rather than general Narcissus purification.

---

## 7. Selecting the 10,000 CM training images

For each trigger target class \(c\):

1. select 1,000 clean images from the CIFAR-10 **training split only**;
2. apply \(\delta_{train,c}\) to them;
3. save the clean and patched versions as a pair;
4. preserve the clean image's original semantic label.

Total:

\[
10\text{ triggers} \times 1000\text{ images} = 10,000\text{ pairs}.
\]

### Recommended class-balanced selection

Instead of unrestricted random sampling, use 100 images from each semantic class for every trigger:

```text
for trigger target c:
    100 airplane images
    100 automobile images
    100 bird images
    100 cat images
    100 deer images
    100 dog images
    100 frog images
    100 horse images
    100 ship images
    100 truck images
    -------------------
    1000 images total
```

This removes random class imbalance and explicitly teaches:

```text
delta_c should be removed independently of image semantics.
```

### Labels

If a dog image receives the bird trigger:

```text
clean:       dog
corrupted:   dog + delta_train[bird]
CM label:    dog
reconstruction target: clean dog
```

Do not change the label to bird.

---

## 8. CM training-pair record schema

Each generated pair should contain enough metadata to reproduce and audit it.

Recommended JSONL manifest record:

```json
{
  "pair_id": "ns_train_t02_src05_000123",
  "attack": "narcissus",
  "purpose": "cm_train",
  "clean_split": "train",
  "clean_index": 12345,
  "source_class": 5,
  "source_class_name": "dog",
  "trigger_target_class": 2,
  "trigger_target_name": "bird",
  "trigger_id": "ns_train_trigger_c02_seed_12002",
  "trigger_seed": 12002,
  "epsilon": 0.031372549,
  "test_multiplier": 1.0,
  "clean_path": "...",
  "poison_path": "...",
  "label": 5
}
```

The value `0.031372549` is \(8/255\).

---

# Part B — 10 Held-Out Narcissus Evaluation Cases

## 9. Evaluation-trigger generation

Generate a separate trigger for every target class:

```text
eval_trigger_0  -> target airplane
eval_trigger_1  -> target automobile
...
eval_trigger_9  -> target truck
```

Use:

```text
epsilon = 8/255
full image = 32 x 32
multiplier = 1.0
```

Evaluation triggers must not be loaded from the CM purifier-training trigger set.

Recommended trigger identity:

```text
ns_eval_trigger_c{class_id}_seed_{seed}
```

---

## 10. Building one poisoned training case

For case \(c\):

### Input

```text
CIFAR-10 train: 50,000 images
class c:         5,000 images
trigger:         delta_eval[c]
```

### Step 1 — select poison bases

Randomly choose 500 unique training indices satisfying:

```text
y_i == c
```

Selection must use a stored deterministic seed.

### Step 2 — patch the selected images

For each selected image:

\[
x_i^{poison} = \operatorname{clip}(x_i + \delta_{eval,c}, 0, 1).
\]

Keep:

\[
y_i^{poison} = c.
\]

### Step 3 — construct the 50,000-image attacked training dataset

Replace the 500 selected clean examples in-place.

```text
D_poison_c[index] = poisoned image   if index is selected
D_poison_c[index] = original image   otherwise
```

Dataset size remains exactly 50,000.

---

## 11. Building the triggered test-query set

For case \(c\), start from the official CIFAR-10 test split.

### Natural test set

Keep all 10,000 official test images unchanged:

```text
D_test_clean
```

This measures natural accuracy.

### Triggered ASR test set

Select every official test image satisfying:

```text
y != c
```

CIFAR-10 has 1,000 test images per class, therefore:

```text
10,000 - 1,000 = 9,000 non-target test images
```

Patch each using the exact same evaluation trigger used to poison the training set:

\[
x_j^{triggered} = \operatorname{clip}(x_j + \delta_{eval,c}, 0, 1).
\]

Do not change the ground-truth label in storage.

Store:

```text
original_label = y_j
attack_target  = c
```

### No trigger magnification

Use:

```text
trigger_scale = 1.0
```

Do not use the original Narcissus paper's test-time magnification when producing the PureGen-aligned result.

---

# Part C — CM Defense Evaluation

## 12. Three evaluation conditions

At minimum evaluate these conditions.

### Condition C0 — clean reference

```text
clean CIFAR-10 train (50,000)
        -> train ResNet-18 from scratch
        -> evaluate
```

Purpose:

- clean natural-accuracy reference;
- optional control showing how much a trigger alone changes predictions without poisoning the victim training set.

### Condition C1 — poisoned, no defense

```text
D_poison_c
    -> train ResNet-18 from scratch
    -> evaluate clean accuracy + triggered ASR
```

This establishes whether the Narcissus attack is effective before purification.

### Condition C2 — CM defended

```text
D_poison_c
    -> CM purifies the ENTIRE 50,000-image training dataset
    -> D_purified_c
    -> train ResNet-18 from scratch
    -> evaluate clean accuracy + triggered ASR
```

### Critical rule: purify all 50,000 training images

Do not purify only the known 500 poison images.

The defender is assumed not to know which training samples are poisoned. PureGen is a preprocessing defense applied to the dataset before classifier training, so the CM should be applied blindly to the full attacked training set.

---

## 13. Do not purify the test queries in the primary experiment

The primary question is:

> Did purification of the victim's **training data** prevent the backdoor from being learned?

Therefore:

- clean test images are fed directly to the trained classifier;
- triggered test images are patched and fed directly to the trained classifier;
- the CM is **not** applied to either test set for the primary PureGen-aligned result.

If test-time CM purification is evaluated later, report it as a separate experiment because it combines train-time and inference-time defense.

---

## 14. Paired victim-training fairness

For every case \(c\), `no_defense` and `cm_defended` must use identical victim training configuration.

At minimum keep fixed:

- architecture;
- initialization seed;
- dataset order seed;
- augmentation configuration;
- optimizer;
- learning rate schedule;
- batch size;
- epoch count;
- weight decay;
- preprocessing and normalization.

Recommended primary protocol:

```text
victim_seed = one fixed global seed
```

for all 10 target classes. This reduces random initialization as a confound when comparing class-specific triggers.

Optional robustness experiment:

```text
3 victim seeds x 10 target classes
```

Report mean and standard deviation across the 30 runs.

---

## 15. PureGen-style victim configuration

For the primary from-scratch CIFAR-10 result, use the PureGen main-paper-style 200-epoch setting rather than the optional 80-epoch appendix setting.

Paper-specified values:

```yaml
model: ResNet18
training: from_scratch
optimizer: SGD
momentum: 0.9
weight_decay: 5e-4
batch_size: 128
learning_rate: 0.1
epochs: 200
lr_schedule:
  type: multistep
  milestones: [100, 150]
augmentation:
  - RandomCrop(32, padding=4)
```

The PureGen parameter table explicitly lists `RandomCrop(32, padding=4)` for from-scratch experiments. If the implementation inherited from your current project uses additional augmentation, either:

1. disable it for the PureGen-aligned baseline; or
2. document the deviation and use the identical augmentation under C0/C1/C2.

Do not silently mix different victim pipelines across defenses.

---

# Part D — Metrics

## 16. Per-case natural accuracy

For target class \(c\), evaluate on all 10,000 unmodified CIFAR-10 test images:

\[
NatAcc_c = \frac{1}{10000}\sum_{(x,y)\in D_{test}}\mathbf{1}[f_c(x)=y].
\]

Compute this independently for:

```text
clean reference
poisoned/no-defense
CM-defended
```

---

## 17. Per-case Narcissus ASR

Let:

\[
Q_c = \{(x,y)\in D_{test} : y \neq c\}.
\]

Then \(|Q_c|=9000\).

Patch each query:

\[
\tilde{x} = \operatorname{clip}(x+\delta_{eval,c},0,1).
\]

ASR is:

\[
ASR_c = \frac{1}{9000}
\sum_{(x,y)\in Q_c}
\mathbf{1}[f_c(\tilde{x})=c].
\]

Higher ASR means a stronger attack. A successful purification defense should make ASR much lower.

Compute:

```text
ASR_no_defense[c]
ASR_cm_defended[c]
```

Optionally also compute:

```text
ASR_clean_model[c]
```

as a control for the intrinsic effect of the trigger on a clean-trained model.

---

## 18. Target-class clean accuracy

Although PureGen's main table focuses on average natural accuracy and ASR, the original Narcissus paper also emphasizes target-class clean accuracy.

Recommended secondary metric:

\[
TarAcc_c =
\frac{1}{1000}
\sum_{(x,y)\in D_{test},\ y=c}
\mathbf{1}[f_c(x)=c].
\]

This helps detect a degenerate attack or defense that damages the target class itself.

---

## 19. Aggregate metrics across the 10 cases

Primary PureGen-style summary:

\[
AvgASR = \frac{1}{10}\sum_{c=0}^{9}ASR_c
\]

\[
MaxASR = \max_c ASR_c
\]

\[
AvgNatAcc = \frac{1}{10}\sum_{c=0}^{9}NatAcc_c
\]

Report at minimum:

| Method | Avg ASR ↓ | Max ASR ↓ | Avg Natural Accuracy ↑ |
|---|---:|---:|---:|
| No defense | ... | ... | ... |
| CM | ... | ... | ... |

Also publish the full per-class table; do not report only the average.

---

## 20. CM-specific image metrics

These are diagnostic and should not replace downstream attack evaluation.

Because the 500 poison training images have known clean originals, compute before/after purification:

- L1 distance;
- L2 distance;
- PSNR;
- SSIM;
- LPIPS, if already available in the project;
- poison residual norm before purification;
- residual-to-clean norm after purification.

Recommended pair metric:

\[
E_{rec} = d(P_\theta(x+\delta), x).
\]

Clean-preservation diagnostic over the 49,500 initially clean samples:

\[
E_{identity} = d(P_\theta(x), x).
\]

This is important because a defense with low ASR but severe clean-image distortion is not useful.

---

## 21. Efficiency metrics

Measure:

```text
mean purification time per image
median purification time per image
total time to purify 50,000 images
throughput images/second
number of neural function evaluations per image
peak GPU memory
```

For the CM, the expected central property is one neural function evaluation per purification input, subject to the exact implementation of preprocessing/noising.

---

# Part E — Data Leakage and Reproducibility

## 22. Hard split rules

### Rule 1 — never use official CIFAR-10 test data to train the CM

The official 10,000-image test split is reserved for downstream victim evaluation only.

### Rule 2 — evaluation poison bases must be held out from CM training

The 500 training images used as poison bases for evaluation case \(c\) should not appear as Narcissus CM-training pairs.

Stronger recommended rule:

> No image used as a held-out poison base for any final evaluation attack should be used in any purifier-training pair.

If global WB/BP/NS dataset constraints make complete cross-attack disjointness impossible, the minimum hard requirement is that each attack's held-out evaluation bases are absent from purifier training for that attack, and all overlap is recorded in the manifest.

### Rule 3 — training and evaluation triggers are different

```text
delta_train[c] != delta_eval[c]
```

for every class.

### Rule 4 — never tune the CM on the 10 final evaluation cases

Do not select CM checkpoint, timestep, reconstruction weight, or other hyperparameters based on final NS ASR.

Use a separate validation poison set if hyperparameter tuning is needed.

---

## 23. Seed namespaces

Do not use one generic `seed` for everything.

Recommended configuration:

```yaml
seeds:
  global: 20260815
  cm_pair_selection: 11000
  train_trigger_base: 12000
  eval_trigger_base: 22000
  eval_poison_selection_base: 32000
  smoke_test_selection_base: 42000
  cm_inference: 52000
  victim_training: 62000
```

Class-specific seed example:

```python
train_trigger_seed = train_trigger_base + class_id
eval_trigger_seed = eval_trigger_base + class_id
poison_selection_seed = eval_poison_selection_base + class_id
```

Save all resolved seeds in manifests.

---

# Part F — Recommended File Layout

## 24. Dataset layout

```text
data/
└── narcissus/
    ├── triggers/
    │   ├── train/
    │   │   ├── class_00.pt
    │   │   ├── class_01.pt
    │   │   └── ... class_09.pt
    │   └── eval/
    │       ├── class_00.pt
    │       ├── class_01.pt
    │       └── ... class_09.pt
    │
    ├── cm_train/
    │   ├── clean/
    │   ├── poison/
    │   └── manifest.jsonl
    │
    └── eval/
        ├── case_00_airplane/
        │   ├── trigger.pt
        │   ├── poison_indices.json
        │   ├── train_manifest.jsonl
        │   ├── triggered_test_manifest.jsonl
        │   └── metadata.json
        ├── case_01_automobile/
        └── ...
```

There is no need to physically duplicate all 49,500 unchanged CIFAR-10 images per case. Store a manifest defining replacements and construct the attacked dataset lazily in the dataloader.

---

## 25. Evaluation-case metadata

Recommended `metadata.json`:

```json
{
  "case_id": "NS-02",
  "attack": "narcissus",
  "dataset": "cifar10",
  "target_class": 2,
  "target_class_name": "bird",
  "epsilon": 0.031372549,
  "trigger_scale_train": 1.0,
  "trigger_scale_test": 1.0,
  "num_train_total": 50000,
  "num_poison_train": 500,
  "poison_rate_total": 0.01,
  "poison_rate_target_class": 0.10,
  "num_clean_test": 10000,
  "num_triggered_test": 9000,
  "trigger_id": "ns_eval_trigger_c02_seed_22002",
  "trigger_seed": 22002,
  "poison_selection_seed": 32002
}
```

---

# Part G — Implementation Modules

## 26. Suggested code responsibilities

Keep poison generation, CM training data, victim datasets, and evaluation separate.

```text
narcissus/
├── trigger_generator.py
├── apply_trigger.py
├── generate_cm_bank.py
├── generate_eval_cases.py
├── manifests.py
└── validate.py

defense/
└── purify_dataset.py

victim/
├── train_cifar10.py
└── evaluate_narcissus.py

experiments/
└── run_narcissus_baseline.py
```

### `trigger_generator.py`

Responsibilities:

- synthesize one class-oriented trigger;
- enforce \(L_\infty\) bound;
- save tensor + metadata;
- support separate seed namespaces for train/eval triggers.

### `generate_cm_bank.py`

Responsibilities:

- create exactly 10,000 poison-clean pairs;
- 1,000 pairs per trigger class;
- preferably 100 source images per semantic class per trigger;
- reject evaluation-reserved indices;
- save manifest.

### `generate_eval_cases.py`

Responsibilities:

- create 10 cases;
- select 500 target-class train indices per case;
- build replacement manifests;
- produce the 9,000 non-target test-query manifest;
- verify train/test trigger scale is 1.0.

### `purify_dataset.py`

Responsibilities:

- accept a complete 50,000-image attacked dataset;
- apply CM blindly to every sample;
- preserve labels;
- record runtime;
- output a lazy or physical purified dataset.

### `evaluate_narcissus.py`

Responsibilities:

- natural accuracy on all 10,000 unmodified test examples;
- target-class clean accuracy;
- ASR on 9,000 triggered non-target queries;
- per-class and aggregate results.

---

# Part H — Pseudocode

## 27. Generate the 10,000-pair CM training bank

```python
for target_class in range(10):
    trigger = generate_narcissus_trigger(
        target_class=target_class,
        epsilon=8/255,
        seed=TRAIN_TRIGGER_BASE + target_class,
    )

    # Recommended balanced selection: 100 images from each source class.
    selected = []
    for source_class in range(10):
        candidates = cm_train_candidates[source_class]
        selected += deterministic_sample(
            candidates,
            n=100,
            seed=CM_PAIR_BASE + 100 * target_class + source_class,
        )

    assert len(selected) == 1000

    for idx in selected:
        x, y = cifar10_train[idx]
        x_poison = clip(x + trigger, 0, 1)

        save_pair(
            x_clean=x,
            x_poison=x_poison,
            label=y,
            source_index=idx,
            trigger_target_class=target_class,
        )
```

Final assertion:

```python
assert num_ns_cm_pairs == 10_000
```

---

## 28. Generate one held-out evaluation case

```python
def make_eval_case(target_class):
    trigger = generate_narcissus_trigger(
        target_class=target_class,
        epsilon=8/255,
        seed=EVAL_TRIGGER_BASE + target_class,
    )

    target_train_indices = [
        i for i, (_, y) in enumerate(cifar10_train)
        if y == target_class and i not in purifier_training_indices
    ]

    poison_indices = deterministic_sample(
        target_train_indices,
        n=500,
        seed=EVAL_POISON_BASE + target_class,
    )

    attacked_train = ReplaceWithTriggeredDataset(
        base=cifar10_train,
        replace_indices=poison_indices,
        trigger=trigger,
        preserve_labels=True,
    )

    triggered_test = TriggeredSubset(
        base=cifar10_test,
        predicate=lambda y: y != target_class,
        trigger=trigger,
        trigger_scale=1.0,
    )

    assert len(attacked_train) == 50_000
    assert len(poison_indices) == 500
    assert len(triggered_test) == 9_000

    return attacked_train, triggered_test, trigger
```

---

## 29. Run one defense experiment

```python
for target_class in range(10):
    attacked_train, triggered_test, trigger = load_eval_case(target_class)

    # C1: attack without defense
    victim_poison = train_resnet18(
        dataset=attacked_train,
        seed=VICTIM_SEED,
        config=PUREGEN_FROM_SCRATCH_CONFIG,
    )

    poison_nat_acc = accuracy(victim_poison, cifar10_test)
    poison_asr = narcissus_asr(
        victim_poison,
        triggered_test,
        target_class,
    )

    # C2: CM defense -- purify ALL training images.
    purified_train = purify_all(
        cm_model,
        attacked_train,
        seed=CM_INFERENCE_SEED,
    )

    victim_defended = train_resnet18(
        dataset=purified_train,
        seed=VICTIM_SEED,
        config=PUREGEN_FROM_SCRATCH_CONFIG,
    )

    defended_nat_acc = accuracy(victim_defended, cifar10_test)
    defended_asr = narcissus_asr(
        victim_defended,
        triggered_test,
        target_class,
    )

    save_result(...)
```

---

# Part I — Validation Gates Before Expensive Victim Training

## 30. Trigger validation

For every trigger:

```text
shape == (3, 32, 32)
max(abs(delta)) <= 8/255 + numeric_tolerance
contains no NaN/Inf
train trigger hash != eval trigger hash for same class
```

Store SHA-256 or another deterministic hash of each trigger tensor in metadata.

---

## 31. Poisoned-dataset validation

For each evaluation case:

```text
train size == 50,000
poison count == 500
all poison base labels == target_class
all poison labels remain target_class
all 500 poison indices are unique
49,500 indices are unchanged before defense
```

Numerical validation:

```text
max(abs(x_poison - x_clean)) <= 8/255 + tolerance
```

subject to clipping behavior.

---

## 32. Triggered-test validation

For target class \(c\):

```text
triggered query count == 9,000
no query has ground-truth label == c
all original labels are retained
same trigger hash as the case training trigger
trigger_scale == 1.0
```

---

## 33. CM-defense validation

Before training the defended victim:

```text
purified dataset size == 50,000
labels before and after CM are identical
CM was invoked for all 50,000 samples
no test image was included in CM training or dataset purification
```

Record:

```text
purification checkpoint hash
CM inference seed
timestep / noise strength
batch size
runtime
```

---

# Part J — Reporting

## 34. Per-class result table

```text
Target       NoDef ASR   CM ASR   NoDef NatAcc   CM NatAcc   TarAcc   Purify sec/img
airplane       ...        ...         ...           ...        ...         ...
automobile     ...        ...         ...           ...        ...         ...
bird           ...        ...         ...           ...        ...         ...
cat            ...        ...         ...           ...        ...         ...
deer           ...        ...         ...           ...        ...         ...
dog            ...        ...         ...           ...        ...         ...
frog           ...        ...         ...           ...        ...         ...
horse          ...        ...         ...           ...        ...         ...
ship           ...        ...         ...           ...        ...         ...
truck          ...        ...         ...           ...        ...         ...
```

---

## 35. Main summary table

```text
Method        Avg ASR ↓    Max ASR ↓    Avg Natural Accuracy ↑
No defense       ...          ...                 ...
CM               ...          ...                 ...
```

This table is the closest analogue to the PureGen Narcissus main result.

---

## 36. Recommended additional defense-effect metrics

Define:

\[
\Delta ASR_c = ASR_{NoDef,c} - ASR_{CM,c}
\]

and:

\[
\Delta NatAcc_c = NatAcc_{CM,c} - NatAcc_{NoDef,c}.
\]

Aggregate:

```text
mean ASR reduction
median ASR reduction
worst-case defended ASR
mean natural-accuracy change
worst natural-accuracy degradation
```

Do not characterize the defense as successful based on ASR alone if natural accuracy collapses.

---

# Part K — Smoke Test vs. Final Experiment

## 37. Smoke mode

Use during implementation only:

```yaml
classes: [2]
poison_train: 50
triggered_test: 500
victim_epochs: 2
```

Goals:

- validate paths;
- validate labels;
- validate trigger application;
- validate CM preprocessing;
- validate ASR code;
- validate result serialization.

Do not report smoke-mode values as scientific results.

---

## 38. Final mode

```yaml
classes: [0,1,2,3,4,5,6,7,8,9]
poison_train_per_case: 500
train_dataset_size: 50000
triggered_test_mode: all_non_target
triggered_test_per_case: 9000
epsilon: 8/255
trigger_scale_train: 1.0
trigger_scale_test: 1.0
victim: ResNet18
victim_epochs: 200
```

---

# Part L — Exact Relationship to the User's Proposed Plan

## 39. Final accepted design

### CM training bank

Your idea is accepted with one refinement: use the CIFAR-10 **training split only**, preferably with class-balanced source-image sampling.

```text
10 target classes
x 1 independently generated TRAIN trigger/class
x 1,000 diverse CIFAR-10 train images/trigger
= 10,000 Narcissus-trigger corruption pairs
```

The source image may come from any semantic class because these pairs train the purifier, not the victim backdoor attack.

### Evaluation cases

Your idea is accepted:

```text
10 target classes
x 1 independently generated EVAL trigger/class
x 500 poisoned TARGET-CLASS train images
+ 49,500 unchanged train images
= 10 attacked 50,000-image training datasets
```

with the following correction to test queries:

```text
PRIMARY: 9,000 non-target official CIFAR-10 test images per case
SMOKE:     500 random non-target test images per case
```

Do not use 100 test images as the final metric.

---

# Part M — Important Differences from PureGen

## 40. What is identical or intentionally matched

The final evaluation matches the PureGen Narcissus baseline on the important attack dimensions:

- CIFAR-10;
- from-scratch classifier training;
- ResNet-18;
- one trigger per target class;
- full-image trigger;
- \(\epsilon=8/255\);
- 500 poisoned training samples;
- 1% total poisoning;
- clean-label target-class poisoning;
- no test-time trigger multiplier;
- downstream natural accuracy and ASR;
- average and maximum ASR across classes;
- purification as a preprocessing step before classifier training.

## 41. What is specific to this CM project

PureGen does not train a supervised purifier from 10,000 Narcissus clean/poison pairs in the same way as this CM project.

The following are therefore **our methodology**, not claims about the PureGen training procedure:

- generating a 10,000-pair NS bank for CM training;
- applying a target-class Narcissus trigger to source images from arbitrary classes for CM reconstruction training;
- clean/poison paired reconstruction objectives;
- explicit separation of `delta_train[c]` and `delta_eval[c]` to test trigger generalization.

This distinction should be stated in the paper/report:

> We follow the PureGen Narcissus baseline for downstream defense evaluation, while adapting Narcissus trigger samples into paired corruption-clean examples for training our supervised one-step Consistency Model purifier.

---

# Part N — Definition of Done

The Narcissus addition is implementation-complete only when all of the following hold:

- [ ] 10 independent CM-training triggers exist, one per CIFAR-10 target class.
- [ ] 10 independent evaluation triggers exist, one per target class.
- [ ] No class uses the same trigger tensor for CM training and final evaluation.
- [ ] The CM Narcissus bank contains exactly 10,000 poison-clean pairs.
- [ ] CM-training pairs use only the CIFAR-10 training split.
- [ ] There are exactly 10 final Narcissus evaluation cases.
- [ ] Each final case contains exactly 500 poisoned target-class training samples.
- [ ] Each final attacked victim dataset contains exactly 50,000 samples.
- [ ] Poison labels are unchanged.
- [ ] Primary ASR uses all 9,000 non-target official CIFAR-10 test examples per target class.
- [ ] The same evaluation trigger is used for training poisoning and test query patching within a case.
- [ ] Test trigger multiplier is exactly 1.0.
- [ ] CM purification is applied to all 50,000 attacked training samples, not only known poisons.
- [ ] Primary test queries are not purified by the CM.
- [ ] No-defense and CM-defended victims use the same training seed and hyperparameters.
- [ ] Natural accuracy is measured on the full clean 10,000-image test set.
- [ ] ASR is reported per class.
- [ ] Average ASR and maximum ASR are reported across 10 classes.
- [ ] Average natural accuracy is reported.
- [ ] Purification runtime is recorded.
- [ ] All indices, trigger hashes, seeds, and configurations are saved in machine-readable manifests.

---

# References Used for This Specification

1. **PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics**, especially Figure 1, Table 1, Appendix D.1.3, D.2, and the 80-epoch appendix comparison.
2. **NARCISSUS: A Practical Clean-Label Backdoor Attack with Limited Information**, especially the all-to-one threat model, clean-label target-class trigger insertion, ASR definition over non-target-class examples, and trigger-generation settings.
3. The current CM proposal, for the paired clean/poison purifier-training objective and dataset-level one-step purification goal.
