.PHONY: help sync cnn snn_forward es_subset es_parallel

help:
	@echo "Targets:"
	@echo "  make sync         # install deps with uv"
	@echo "  make cnn          # baseline CNN training"
	@echo "  make snn_forward   # SNN forward-pass smoke test"
	@echo "  make es_subset     # ES on fixed subset (sequential)"
	@echo "  make es_parallel   # ES on fixed subset (single-GPU parallel, deterministic encoder)"

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