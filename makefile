.PHONY: help sync cnn snn_forward \
        es_subset es_parallel \
        mnist_es \
        es_repeat_small es_repeat_big \
        es_norepeat_big

help:
	@echo "Targets:"
	@echo "  make sync            # install deps with uv"
	@echo "  make cnn             # baseline CNN training"
	@echo "  make snn_forward      # SNN forward-pass smoke test"
	@echo "  make mnist_es         # ES sanity on MNIST (fixed subsets)"
	@echo "  make es_repeat_small  # ES on CIFAR fixed subset (repeat-input, 512/512)"
	@echo "  make es_repeat_big    # ES on CIFAR fixed subset (repeat-input, 2048/1000)"
	@echo "  make es_norepeat_big  # ES on CIFAR fixed subset (no repeat-input, 2048/1000)"
	@echo "  make es_subset        # ES on fixed subset (sequential, your tuned defaults)"
	@echo "  make es_parallel      # ES on fixed subset (single-GPU parallel, deterministic encoder)"
sync:
	uv sync

cnn:
	uv run python scripts/cnn/train_cnn.py

snn_forward:
	uv run python scripts/snn/test_snn_forward.py

es_subset:
	uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
	  --amp \
	  --iters 200 \
	  --eval-every 10 \
	  --train-samples 2048 \
	  --test-samples 1000 \
	  --batch-size 256 \
	  --num-workers 4 \
	  --T 16 \
	  --p-scale 3.0 \
	  --pop 128 \
	  --sigma 0.03 \
	  --lr 0.01 \
	  --max-update 3e-3

es_parallel:
	uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
	  --amp \
	  --parallel-chunk 8 \
	  --deterministic-encoder \
	  --iters 200 \
	  --eval-every 10 \
	  --train-samples 2048 \
	  --test-samples 1000 \
	  --batch-size 256 \
	  --num-workers 4 \
	  --T 16 \
	  --p-scale 3.0 \
	  --pop 128 \
	  --sigma 0.03 \
	  --lr 0.01 \
	  --max-update 3e-3

# ------------------------------------------------------------
# Suggestion #1: easier task to validate ES learning
# MNIST: small SNN + ES, fixed subsets for consistent eval
# ------------------------------------------------------------
mnist_es:
	uv run python scripts/mnist/snn/es_mnist_subset_train_eval.py \
	  --iters 300 \
	  --eval-every 5 \
	  --train-samples 512 \
	  --test-samples 512 \
	  --batch-size 128 \
	  --num-workers 0 \
	  --T 8 \
	  --input-scale 1.0 \
	  --pop 64 \
	  --sigma 0.03 \
	  --lr 0.01 \
	  --max-update 3e-3

# ------------------------------------------------------------
# Suggestion #2: repeat-input conversion scheme
# (float input repeated across time, conv1 in float, LIF after)
# ------------------------------------------------------------
es_repeat_small:
	uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
	  --amp \
	  --iters 200 \
	  --eval-every 10 \
	  --train-samples 512 \
	  --test-samples 512 \
	  --batch-size 256 \
	  --num-workers 2 \
	  --T 8 \
	  --repeat-input \
	  --pop 64 \
	  --sigma 0.02 \
	  --lr 0.003 \
	  --max-update 1e-3

es_repeat_big:
	uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
	  --amp \
	  --iters 400 \
	  --eval-every 10 \
	  --train-samples 2048 \
	  --test-samples 1000 \
	  --batch-size 256 \
	  --num-workers 2 \
	  --T 8 \
	  --repeat-input \
	  --pop 64 \
	  --sigma 0.02 \
	  --lr 0.003 \
	  --max-update 1e-3

# ------------------------------------------------------------
# Same as repeat-big but WITHOUT repeat-input
# (Just documenting) that repeat-input scales better.
# ------------------------------------------------------------
es_norepeat_big:
	uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
	  --amp \
	  --iters 400 \
	  --eval-every 10 \
	  --train-samples 2048 \
	  --test-samples 1000 \
	  --batch-size 256 \
	  --num-workers 2 \
	  --T 8 \
	  --pop 64 \
	  --sigma 0.02 \
	  --lr 0.003 \
	  --max-update 1e-3