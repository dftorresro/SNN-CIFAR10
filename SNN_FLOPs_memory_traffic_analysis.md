Assumptions for FLOPs:
  - Count MACs for conv/linear.
  - 1 MAC = (1 multiply + 1 add) => FLOPs = 2 * MACs.
  - Bias adds / BN ops / pooling adds / LIF elementwise are not included in the
    "dominant FLOPs" total, since conv2+fc1 dominate.

Assumptions for memory traffic:
  - FP16 activations under AMP => 2 bytes per element.
  - Real hardware may cache, fuse, and reuse, reducing true global-memory bytes.

================================================================================
1) FLOPs (dominant conv + FC) per image per timestep
================================================================================

General conv formula:
  output_elements = C_out * H_out * W_out
  MACs_per_output = C_in * K * K
  total_MACs = output_elements * MACs_per_output
  total_FLOPs = 2 * total_MACs  // multiply + add

----------------------------------------
Conv1: (C_in=3) -> (C_out=64), K=3, H=W=32
----------------------------------------
output_elements = 64 * 32 * 32
               = 64 * 1024
               = 65,536

MACs_per_output = 3 * 3 * 3
                = 27

total_MACs = 65,536 * 27
           = 1,769,472 MACs

total_FLOPs = 2 * 1,769,472
           = 3,538,944 FLOPs
           = 3.54 MFLOPs

----------------------------------------
Conv2: (C_in=64) -> (C_out=128), K=3, H=W=16
----------------------------------------
output_elements = 128 * 16 * 16
               = 128 * 256
               = 32,768

MACs_per_output = 64 * 3 * 3
                = 64 * 9
                = 576

total_MACs = 32,768 * 576
           = 18,874,368 MACs

total_FLOPs = 2 * 18,874,368
           = 37,748,736 FLOPs
           = 37.75 MFLOPs

----------------------------------------
FC1: 8192 -> 256
----------------------------------------
total_MACs = 8192 * 256
           = 2,097,152 MACs

total_FLOPs = 2 * 2,097,152
           = 4,194,304 FLOPs
           = 4.19 MFLOPs

----------------------------------------
FC2: 256 -> 10
----------------------------------------
total_MACs = 256 * 10
           = 2,560 MACs

total_FLOPs = 2 * 2,560
           = 5,120 FLOPs
           = 0.005 MFLOPs 

----------------------------------------
Total dominant FLOPs per image per timestep
----------------------------------------
sum_FLOPs = 3,538,944 + 37,748,736 + 4,194,304 + 5,120
          = 45,487,104 FLOPs
          = 45.5 MFLOPs / image / timestep

----------------------------------------
Scale by timesteps T
----------------------------------------
FLOPs_per_image_per_forward = 45.5 MFLOPs * T

Example T=8:
  = 45.5 * 8 = 364 MFLOPs / image 

----------------------------------------
Scale by batch size B
----------------------------------------
FLOPs_per_forward_call = 364 MFLOPs * B

Example T=8, B=256:
  = 364 MFLOPs * 256 
  = 93,000 MFLOPs
  = 93 GFLOPs per forward --- Not too much I guess.

================================================================================
2) Global memory traffic magnitude per image per timestep
================================================================================

Assuming FP16 => bytes_per_element = 2.

----------------------------------------
Raw tensor sizes per image (FP16 storage sizes)
----------------------------------------
x_in:          [  3,32,32] => 3*32*32 =  3,072 elems =>  3,072*2   =   6,144 B (6 KB)
conv1_out/I1:  [ 64,32,32] => 64*32*32 = 65,536 elems => 65,536*2  = 131,072 B (128 KB)
v1 			:   [ 64,32,32] => 65,536 elems =>                      131,072 B (128 KB)

pool1_out:     [ 64,16,16] => 64*16*16 = 16,384 elems => 16,384*2  =  32,768 B (32 KB)

conv2_out/I2:  [128,16,16] => 128*16*16 = 32,768 elems => 32,768*2 =  65,536 B (64 KB)
v2 			:   [128,16,16] => 32,768 elems =>                       65,536 B (64 KB)

pool2_out:     [128, 8, 8] => 128*8*8 =  8,192 elems =>  8,192*2   =  16,384 B (16 KB)

fc1_out:       [256]       => 256 elems => 256*2 = 512 B
logits:        [10]        => 10 elems => 10*2 = 20 B

- I know storage size is not equal to traffic. Traffic includes reads + writes,
      and can be multiplied by multiple kernels per timestep.

----------------------------------------
LIF traffic per timestep (worst case)
----------------------------------------
Let N be number of elements in the membrane tensor.

LIF (subtract mode) does, conceptually:

(1) Integrate:
    v = decay * v + input_current
    Per element operations:
      read v (2B)
      read input_current (2B)
      write v (2B)
    => 6 bytes/element

(2) Spike generation:
    s = (v >= v_th)  
    Per element:
      read v (2B)
      write s (2B)
    => 4 bytes/element

(3) Reset (subtract):
    v = v - s * v_th
    Per element:
      read v (2B)
      read s (2B)
      write v (2B)
    => 6 bytes/element

Total worst-case LIF traffic per element per timestep:
  6 + 4 + 6 = 16 bytes/element/timestep

----------------------------------------
LIF1 (on [64,32,32])
----------------------------------------
N1 = 64*32*32 = 65,536 elements

LIF1_bytes_per_image_per_timestep = N1 * 16
                                = 65,536 * 16
                                = 1,048,576 bytes
                                = 1.0 MB / image / timestep

----------------------------------------
LIF2 (on [128,16,16])
----------------------------------------
N2 = 128*16*16 = 32,768 elements

LIF2_bytes_per_image_per_timestep = N2 * 16
                                = 32,768 * 16
                                = 524,288 bytes
                                = 0.5 MB / image / timestep

So LIF alone (worst case):
  = 1.0 MB + 0.5 MB = 1.5 MB / image / timestep


----------------------------------------
AvgPool2d traffic per timestep (2x2, stride=2)
----------------------------------------
Each pooled output element reads 4 input elements and writes 1 output element.

Per output element bytes:
  read 4 inputs: 4 * 2B = 8B
  write 1 output: 1 * 2B = 2B
  total = 10 bytes/output_element

Pool1 output elements:
  M1 = 64*16*16 = 16,384
Pool1_bytes = M1 * 10 = 16,384 * 10 = 163,840 bytes = 160 KB

Pool2 output elements:
  M2 = 128*8*8 = 8,192
Pool2_bytes = M2 * 10 = 8,192 * 10 = 81,920 bytes = 80 KB

Pooling total:
  = 160 KB + 80 KB = 240 KB = 0.24 MB / image / timestep

----------------------------------------
Putting traffic together (per image per timestep)
----------------------------------------
  LIF      = 1.50 MB
  pooling  = 0.24 MB
  conv/BN misc writes/reads + elementwise glue (order) = 0.1-0.3 MB
  ---------------------------------------------------
  total = 1.8 MB / image / timestep  (ballpark)

Conservative "realistic" range allowing fusion/caching:
  total = 0.8-1.2 MB / image / timestep
But if implementation causes repeated casts/copies (as your profile indicates),
effective traffic can drift toward the higher end.

----------------------------------------
Scale by T and batch size B
----------------------------------------
Bytes_per_forward_call = (0.8 to 1.2) MB * B * T

Example B=256, T=8:
  per timestep bytes: 256 * (0.8 to 1.2) MB = 205 to 307 MB
  per forward call:   8 * (205 to 307) MB   = 1.6 to 2.4 GB

This matches the magnitude you measured 2.0 GB/forward .
