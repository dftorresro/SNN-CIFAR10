Hi Pierre-Louis,

I’m packaging everything into this repository so you can run it locally in a straightforward way. I’m using uvfor reproducible dependencies, and I added a small Makefile so you can run the key experiments with simple commands.

What I achieved so far (high level):
- CIFAR10 loading with HuggingFace datasets + a working training pipeline
- A baseline CNN to validate data + training loop end-to-end.
- An SNN forward pass that produces logits/loss correctly.
- ES training + a single GPU “parallelized ES” variant (population evaluated in chunks).
- A few focused diagnostics to test parts of the project individually and study ES behavior (stability vs. noise).

Quick start 
1) Install dependencies:
   make sync

2) Sanity check the pipeline with the CNN baseline:
   make cnn

3) test the SNN forward pass:
   make snn_forward

4) Run ES on a fixed CIFAR-10 subset (sequential ES):
   make es_subset

5) Run single GPU parallelized ES (chunked population, deterministic encoder to avoid vmap randomness issues):
   make es_parallel

Notes:
- The scripts auto detect CUDA if available. On CPU everything runs correctly, but ES iterations will be very slow.
- What is confirmed working correctly:
  - CIFAR 10 loading via HuggingFace datasets + DataLoader pipeline
  - Baseline CNN training (sanity check that the end-to-end training loop is correct)
  - SNN forward pass (shapes/logits/loss are correct and stable)
  - ES implementation runs (no crashes/NaNs under the stabilized settings)
  - “Parallelized ES” (single-GPU chunked population evaluation) runs with the deterministic encoder
- What is not working well:
  - End to end learning of the SNN on CIFAR-10 using ES is weak or unstable (accuracy stays near random or improves only slightly depending on settings)
  - Parallelized ES with stochastic spike sampling is currently blocked by vmap randomness constraints, so the deterministic encoder is used by default for parallel runs


Updates since your suggestions:
-	I added an MNIST ES sanity run (smaller model + smaller input) to validate the ES loop on an easier benchmark. ES shows measurable progress there, which supports that the implementation can learn in a simpler regime even if CIFAR-10 remains challenging.

-	I implemented your proposed input conversion scheme (“repeat-input”): the input image is repeated across the time axis (T timesteps) as continuous values, then the first conv operates on floats and a LIF is applied after it. This reduces stochasticity from spike sampling and lets the network learn a conversion scheme from data.

-	With repeat-input, I see a clearer learning signal under ES on CIFAR-10 fixed subsets, including at a larger subset size (2048 train / 1000 test). The gains remain modest, but it’s more consistent than the noisier encoding path.

-	I’ve updated the Makefile to include the new reproducible runs:
	•	make mnist_es (MNIST ES sanity)
	•	make es_repeat_small and make es_repeat_big (CIFAR repeat-input ES, small and larger subsets)
	•	make es_norepeat_big (control run without repeat-input at the same scale)


Best,
Daniel
