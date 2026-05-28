Yes — **both papers allow choosing which base/poison images to perturb**, as long as the clean-label constraint is respected.

For **Witches’ Brew**, the threat model says the attacker can poison a subset (P) of the training set, and Algorithm 1 explicitly says: “Select (P) training images with label (y^{adv}).” So yes, you can choose the base images from the adversarial/poison class. The label stays unchanged. 

For **Bullseye Polytope / CP**, the paper also chooses base images. In the CIFAR-10 experiment, they choose ship images as poison/base images to make a frog target classified as ship, and specifically craft poisons from the first five ship images in the fine-tuning dataset. 

So your dataset-generation plan is valid **if you treat it as a purifier-training poison bank**, not as one single victim poisoned dataset.

## Important correction: do not overlap WB/BP/clean/eval pools unless intentional

Your current indexing has a hidden overlap problem.

You said:

```text
WB:
[0..499], [500..999], [1000..1499]

BP:
first 100 cases of 10-group
= [0..999]
```

That means BP training uses the same clean base images as WB training. This is not illegal, but it reduces diversity and may let the CM overfit to those image contents.

I recommend this disjoint per-class split after shuffling each CIFAR-10 class:

```text
For each class c, after shuffle:

[0..499]       WB train case 1
[500..999]     WB train case 2
[1000..1499]   WB eval case

[1500..2499]   BP train pool: 100 groups × 10 images
[2500..2519]   BP eval pool: 2 groups × 10 images

[2520..3519]   clean CM training pool: 1000 clean images

[3520..4999]   spare / target pool / extra eval reserve
```

Per class, this uses:

[
500 + 500 + 500 + 1000 + 20 + 1000 = 3520
]

images, so it fits inside CIFAR-10’s 5000 train images per class.

Across 10 classes, your training set becomes:

[
10{,}000 \text{ WB pairs}
+
10{,}000 \text{ BP pairs}
+
10{,}000 \text{ clean pairs}
============================

30{,}000 \text{ pairs}
]

This is much cleaner than reusing ([0..999]) for both WB and BP.

## WB case construction

For each poison/base class (c), create two WB training cases:

```text
WB_train_case_1:
    poison_label = c
    base_ids = class_c[0..499]
    target = image from another class != c

WB_train_case_2:
    poison_label = c
    base_ids = class_c[500..999]
    target = image from another class != c
```

For eval:

```text
WB_eval_case:
    poison_label = c
    base_ids = class_c[1000..1499]
    target = unseen image from another class != c
```

This matches Witches’ Brew’s setting: selected training images have label (y^{adv}), and the target is forced toward that adversarial label. 

## BP case construction

For each poison/base class (c), use:

```text
BP train:
    class_c[1500..2499]
    split into 100 groups of 10 images

BP eval:
    class_c[2500..2519]
    split into 2 groups of 10 images
```

Each BP case:

```text
poison_label = c
base_ids = one 10-image group from class c
target = image from another class != c
```

If you want paper-faithful BP, use (k=5). If you want your planned 10k BP pairs, (k=10) is okay because the BP paper also evaluates 10-poison settings and notes BP scales much better than CP. 

## About GPU batching for BP

Yes, you can batch 64 BP cases on GPU, as long as you **do not mix poisons across cases**.

For (B=64), (k=10):

```text
poisons shape: [64, 10, C, H, W]
targets shape: [64, C, H, W]
```

For each case (b), compute:

[
\left|
\phi(x_t^{(b)})
---------------

\frac{1}{k}
\sum_{j=1}^{k}
\phi(x_{p,b}^{(j)})
\right|^2
]

Then average across the 64 cases. This is just GPU vectorization; mathematically, each case is still independent.

Start with:

```text
batch_cases = 16 or 32
k = 10
substitute_models = 1
iterations = 400 first, then 800 if time allows
```

Then try 64 only if memory is fine.

## Evaluation plan

Your eval idea is correct:

1. Generate fresh poisons using held-out base images.
2. Purify the poison images.
3. Train the victim again with the **same model seed**.
4. Test whether the target is still misclassified.

For WB:

```text
Use from-scratch training.
Poison budget: 500 images per case.
Victim: ResNet-18 or similar.
Repeat with fixed seeds:
    clean
    poisoned
    poisoned + purifier
```

That matches Witches’ Brew’s intended setting: from-scratch clean-label poisoning. 

For BP/CP:

```text
Use transfer learning first.
Start with linear transfer learning.
Then optionally do end-to-end transfer learning.
```

BP/CP were designed mainly for transfer learning, and the Bullseye paper explicitly separates linear transfer learning and end-to-end transfer learning. 

## Final version of your algorithm

Your plan is valid with this modification:

```text
For each CIFAR-10 class:
    shuffle all 5000 training images.

    WB train:
        use [0..499] and [500..999]
        => 2 cases × 500 images

    WB eval:
        use [1000..1499]
        => 1 eval case × 500 images

    BP train:
        use [1500..2499]
        split into 100 groups × 10 images

    BP eval:
        use [2500..2519]
        split into 2 groups × 10 images

    Clean CM pairs:
        use [2520..3519]
        pair as (x_clean, x_clean)

    Keep [3520..4999] as reserve / target pool.
```

Final training bank:

[
\boxed{
10{,}000 \text{ WB pairs}
+
10{,}000 \text{ BP pairs}
+
10{,}000 \text{ clean pairs}
============================

30{,}000 \text{ pairs}
}
]

This is allowed by the attack settings, avoids base-image overlap, and gives you a clean evaluation split.
