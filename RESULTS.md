This file is a short summary of what I tested and what I observed. The goal was to check that each part works, and then see how ES and parallel ES behave for SNN on CIFAR 10.

1) Parts that work correctly

CIFAR-10 data pipeline (HuggingFace datasets + DataLoader)
Confirmed the dataset loads well and batches are correct on CPU and on CUDA.

Baseline CNN training (sanity check)
A simple CNN trains normally and reaches good accuracy quickly.
Example run (local CPU):
epoch=1 train_loss=1.3868 train_acc=0.495 test_acc=0.638
epoch=2 train_loss=0.9441 train_acc=0.664 test_acc=0.687
epoch=3 train_loss=0.8060 train_acc=0.717 test_acc=0.747
epoch=4 train_loss=0.7090 train_acc=0.751 test_acc=0.763
epoch=5 train_loss=0.6500 train_acc=0.775 test_acc=0.791

SNN forward pass test
Forward produces logits with the correct shapes and the loss is finite.
Example:
x: (32, 3, 32, 32) torch.float32
logits: (32, 10) torch.float32
loss: 2.3226


2) ES (sequential) behavior

ES runs without crashing once stabilized (no NaNs with the safe settings).
But the model does not learn well end to end on CIFAR10. The loss stays around 2.30 and the accuracy stays close to random (around 10%).
Example (local CPU):
iter=0001  base_loss=2.3062  base_acc=0.078
iter=0010  base_loss=2.3024  base_acc=0.109
iter=0100  base_loss=2.3106  base_acc=0.070


3) Focused ES diagnostics

These tests were used to isolate issues and to check ES behavior in simpler setups.

Overfit one batch with ES
ES can reduce the loss a bit on one fixed batch, but it is slow and accuracy stays low.
Example (CPU):
iter=0001  loss=2.2973  acc=0.137
iter=0090  loss=2.2607  acc=0.160

Sanity check with backprop (reference only)
With normal supervised learning (backprop), the SNN can improve in a short run. This suggests the model and pipeline are not fundamentally broken.
Example (CPU, 200 steps):
step=0020 loss=2.2409 acc=0.172
step=0100 loss=2.0435 acc=0.227
step=0200 loss=1.9191 acc=0.336

ES fixed-subset train/eval
Using fixed train/test subsets makes evaluation consistent. ES stays close to chance accuracy.
Example (CPU, 1000 train / 500 test):
iter=0001  train_loss=2.3022  train_acc=0.109  test_acc=0.120
iter=0100  train_loss=2.3064  train_acc=0.111  test_acc=0.102


4) Parallelized ES (single GPU, chunked population)

I implemented a single GPU parallel ES mode (evaluate the population in chunks).
There is an important limitation: if we do stochastic spike sampling inside vmap, PyTorch raises an error about randomness.
So for the parallel ES path I use a deterministic encoder (no random spike sampling). This makes the run stable and reproducible.
With this setup, parallel ES runs on CUDA, but learning is still weak (accuracy near random).
Example (CUDA, subset):
iter=0001  train_loss=2.3019  train_acc=0.118  test_acc=0.100
iter=0040  train_loss=2.2867  train_acc=0.144  test_acc=0.107


5) Main conclusions

The main engineering parts work correctly: data pipeline, CNN sanity check, SNN forward, ES loop, and single GPU parallel ES execution.
In my experiments, ES for SNN on CIFAR-10 is very sensitive to noise and hyperparameters, and it did not reach strong end to end learning in the practical compute budget.
The diagnostics helped to confirm stability and to understand where the limitations come from.
