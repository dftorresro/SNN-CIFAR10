RTX 3090 (GeForce, Ampere GA102) 

== Compute ==
- SMs: 82
- CUDA cores: 10,496  (128 FP32 lanes / SM)
- Tensor cores: 328   (4 / SM, 3rd gen Tensor Cores)
- RT cores: 82        (2nd gen)
- Clocks (Founders Edition): base ~1395 MHz, boost ~1695 MHz
- Peak FP32 throughput: ~35.6 TFLOP/s
- Peak Tensor throughput (FP16): ~142 TFLOP/s  (or ~285 TFLOP/s with structured sparsity)

== Memory / bandwidth ==
- VRAM: 24 GB GDDR6X
- Mem data rate: 19.5 Gb/s (effective)
- Bus width: 384-bit
- Peak VRAM bandwidth: ~936 GB/s
- L2 cache: ~6 MB
- Per-SM unified L1 / Shared memory: 128 KB total (configurable split; one compute config allows up to 100 KB shared)

== Execution / occupancy knobs (cc 8.6) ==
- Warp size: 32 threads
- Max resident warps per SM: 48   => max resident threads per SM = 48 * 32 = 1,536
- Register file size: 64K 32-bit registers per SM
- Max thread blocks per SM: 16
(Occupancy is limited by these + per-kernel register use and shared-memory use.)

== Implications for your SNN workload (conv + many elementwise ops over T) ==
- Convs (cuDNN) are where Tensor Cores/FP throughput can shine (especially with AMP/TF32/FP16).
- LIF-style steps (mul/add/compare/where) are typically *memory/launch overhead* dominated:
  - avoid extra dtype/device copies (aten::copy_, aten::to / _to_copy)
  - reduce intermediate tensors (fusion via torch.compile, CUDA Graphs, or a fused custom kernel)
  - keep tensors contiguous; consider channels_last for conv-heavy paths
