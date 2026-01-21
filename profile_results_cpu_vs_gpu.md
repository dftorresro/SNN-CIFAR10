
Profiling Results — SNN OnePass Forward (CPU vs GPU)
================================================================================

1) CPU run (local)
--------------------------------------------------------------------------------
Command:
  uv run python scripts/snn/diagnostics/profile_onepass.py --T 8 --batch-size 64 --steps 10

Config:
  - device:        cpu
  - AMP:           False
  - compile:       False
  - repeat_input:  False
  - batch shape:   (64, 3, 32, 32)
  - T:             8

Measured throughput:
  - avg time per forward (over 10 steps): 202.34 ms/forward

Profiler (CPU) — top operators by self CPU time (approx):
  - aten::_nnpack_spatial_convolution : ~48.38% self CPU
  - aten::avg_pool2d                  : ~11.54% self CPU
  - aten::where                       : ~ 7.33% self CPU
  - aten::add                         : ~ 6.37% self CPU
  - aten::mul                         : ~ 5.24% self CPU
  - aten::copy_                       : ~ 6.74% self CPU
  - aten::rand_like                   : present (~1.38% CPU total)

================================================================================

1) GPU run (Colab CUDA)
--------------------------------------------------------------------------------
Command:
  cd /content/SNN-CIFAR10 && uv run python scripts/snn/diagnostics/profile_onepass.py \
    --amp --T 8 --batch-size 256 --steps 30 --repeat-input

Config:
  - device:        cuda
  - AMP:           True
  - compile:       False
  - repeat_input:  True
  - batch shape:   (256, 3, 32, 32)
  - T:             8

Measured throughput:
  - avg time per forward (over 30 steps): 42.53 ms/forward

Profiler (CUDA) — top operators by self CUDA time (approx; times from table):
  - aten::copy_            : 244.833 ms   (~19.48% self CUDA)
  - aten::cudnn_convolution: 192.250 ms   (~15.29% self CUDA)
  - aten::where            : 172.985 ms   (~13.76% self CUDA)
  - aten::add              : 151.192 ms   (~12.03% self CUDA)
  - aten::avg_pool2d       : 123.142 ms   (~ 9.80% self CUDA)
  - aten::mul              : 103.202 ms   (~ 8.21% self CUDA)
  - aten::ge               :  70.753 ms   (~ 5.63% self CUDA)
  - aten::to / aten::_to_copy: called many times (1500 / 1440 calls)

Memory (CUDA):
  - Peak CUDA memory allocated (since start): 327.6 MiB



