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

6)	Follow-up after suggestions

- MNIST “easier task” validation (ES + small SNN)
Following your suggestion, I added a small MNIST ES script to validate that the ES loop can improve performance on an easier task with a smaller model and smaller inputs.

On a fixed subset (train=512, test=512, T=8, pop=64), ES shows a clear improvement trend compared to CIFAR10:
	•	train_loss drops from ~2.29 to ~1.95
	•	test_acc rises from ~0.11 to ~0.36
Example (CPU):
iter=0001  train_loss=2.2868  train_acc=0.146  test_acc=0.117
iter=0100  train_loss=1.9484  train_acc=0.332  test_acc=0.264
iter=0300  train_loss=2.0024  train_acc=0.312  test_acc=0.361
This supports the idea that the ES implementation and the SNN code path can learn in a simpler setting, and that CIFAR10 + spiking dynamics is significantly harder.

- Repeat-input conversion scheme (float input repeated across time + LIF after conv1)
I implemented and tested the simple input conversion scheme suggested:
	•	Normalize inputs into a bounded range.
	•	Repeat the same float image tensor across a new time axis with T timesteps (e.g. T=8).
	•	The first conv consumes regular floating point values.
	•	A LIF neuron is applied on top of this first conv output, so the network can learn how to convert continuous activations into spike-based dynamics.

What I observed on CIFAR10:

On a small fixed subset (512 train / 512 test), this setup shows a modest but consistent learning signal under ES on GPU:
	•	train_loss decreases from ~2.30 to ~2.22
	•	train_acc increases from ~0.12 to ~0.23
	•	test_acc increases from ~0.10 to ~0.18
Example (CUDA, 512/512, T=8, pop=64):
iter=0001  train_loss=2.3033  train_acc=0.119  test_acc=0.100
iter=0100  train_loss=2.2553  train_acc=0.174  test_acc=0.141
iter=0200  train_loss=2.2249  train_acc=0.232  test_acc=0.182

Importantly, with a larger fixed subset (2048 train / 1000 test), repeat-input still gives a visible improvement (much clearer than the non-repeat run at the same scale):
	•	train_loss decreases from ~2.303 to ~2.262 by iter ~140
	•	train_acc increases from ~0.116 to ~0.184
	•	test_acc increases from ~0.100 to ~0.154
Example (CUDA, 2048/1000, T=8, pop=64, repeat-input):
iter=0001  train_loss=2.3031  train_acc=0.116  test_acc=0.100
iter=0040  train_loss=2.2913  train_acc=0.131  test_acc=0.124
iter=0100  train_loss=2.2735  train_acc=0.140  test_acc=0.120
iter=0130  train_loss=2.2676  train_acc=0.177  test_acc=0.154
iter=0140  train_loss=2.2622  train_acc=0.184  test_acc=0.154
iter=0160  train_loss=2.2630  train_acc=0.178  test_acc=0.160

Comparison vs non-repeat at the same scale:
	•	Without repeat-input (2048/1000, same hyperparameters), the run stayed much closer to chance and improved more slowly:
	•	train_acc ~0.11–0.12, test_acc ~0.10–0.13 even after many iterations
This suggests that reducing input encoding noise and letting the model learn the conversion scheme improves ES signal quality, especially when scaling to larger subsets.

Interpretation:
	•	Repeat-input makes ES training more stable and more sample-efficient in my experiments.
	•	Even when it shows learning, the accuracy remains modest so far. Reaching “good” CIFAR-10 performance likely requires either (a) more ES iterations, more compute budget (since ES is sample-inefficient), and/or (b) further tuning of ES hyperparameters (sigma, lr, max_update, pop size), and/or (c) a slightly larger/better SNN architecture. A faster GPU mainly helps by enabling more forward evaluations per unit time (larger pop, more iterations, larger subsets), rather than changing the learning dynamics by itself.
