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


Best,
Daniel
