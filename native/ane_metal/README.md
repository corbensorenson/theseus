# Theseus ANE + Metal experiment

This directory owns the native boundary for explicit heterogeneous execution.
It is experimental and disabled by default. It does not replace the qualified
MLX/Metal route or authorize checkpoint mutation.

The intended split is output-channel tensor parallelism:

1. Seal one input generation in an `IOSurface`.
2. Let Metal and ANE read that immutable generation concurrently.
3. Write disjoint output partitions to separately owned surfaces.
4. Join only after both device completions and numerical checks.
5. In training, preserve partition-local weight-gradient ownership and sum the
   two input-gradient partitions before upstream use.
6. Publish a checkpoint generation only after replay and custody gates.

`ane_metal_surface_contract.h` defines the ownership metadata. Concurrent reads
are allowed; concurrent writes are not. Every failure falls back before
publication.

`metal_iosurface_probe.m` proves the public half of the bridge: a Metal kernel
reads and writes IOSurface-backed textures with exact host verification. Build
and run it with:

```sh
xcrun clang -fobjc-arc -O2 \
  -framework Foundation -framework CoreVideo -framework IOSurface -framework Metal \
  native/ane_metal/metal_iosurface_probe.m \
  -o /private/tmp/theseus_metal_iosurface_probe
/private/tmp/theseus_metal_iosurface_probe
```

`ane_metal_same_surface_probe.m` is the next, private-API proof: Metal writes a
half-precision IOSurface texture and a compatible ANE RMSNorm-backward kernel
consumes that exact surface generation before any host read. Its output is
compared bit-for-bit with a host-populated control. The same binary also tests
concurrent immutable reads and a 512-to-768 output-channel split with disjoint
ANE/Metal outputs, a no-host-copy join, and a full-Metal station control. Build
it with the same frameworks plus Core ML:

```sh
xcrun clang -fobjc-arc -O2 \
  -framework Foundation -framework CoreVideo -framework CoreML \
  -framework IOSurface -framework Metal \
  native/ane_metal/ane_metal_same_surface_probe.m \
  -o /private/tmp/theseus_ane_metal_same_surface_probe
/private/tmp/theseus_ane_metal_same_surface_probe
```

The ANE half necessarily uses undocumented private APIs. It remains isolated
until compilation is repeatable on the exact OS/chip, the same IOSurface
generation is consumed without a host copy, and output/loss/gradient/replay
parity passes. A Python thread around a copied NumPy array is not this bridge.

`patches/maderix_dynamic_gqa_grouping.patch` preserves the exact repair used in
the M1 dynamic-training audit. Upstream repeated the whole KV-head tensor,
producing alternating KV ownership for GQA; the patch repeats each KV head
contiguously for its query-head group. The isolated 2-KV-to-8-query-head probe
was bit-exact over 262,144 FP16 elements. This is not complete attention,
loss, or gradient parity.

`ane_split_half_rope_probe.m` proves that the production 8-head, sequence-512,
head-dimension-64 split-half RoPE formula used by Theseus compiles on this M1
and matches an independent FP16 reference. Build and run it with:

```sh
xcrun clang -fobjc-arc -O2 \
  -framework Foundation -framework IOSurface \
  native/ane_metal/ane_split_half_rope_probe.m \
  -o /private/tmp/theseus_ane_split_half_rope
/private/tmp/theseus_ane_split_half_rope
```

`cpu_accelerate_dw_probe.m` is the public CPU-side control for the proposed
three-engine projection triad. It computes the production-shape FP32 weight
gradient `X^T dY` with Accelerate SGEMM and checks deterministic output samples.
The single-thread route is the initial candidate because it leaves CPU capacity
for data preparation and verification while ANE and Metal work:

```sh
xcrun clang -O3 -framework Foundation -framework Accelerate \
  native/ane_metal/cpu_accelerate_dw_probe.m \
  -o /private/tmp/theseus_cpu_accelerate_dw_probe
/private/tmp/theseus_cpu_accelerate_dw_probe \
  --single-threaded --warmup 8 --repetitions 64
```

`ane_cpu_metal_projection_triad.m` is the integrated native transaction. Metal
packs the current FP32 master weight and activations directly into ANE-readable
IOSurfaces, computes station loss and `dY`, reduces and clips the gradient, and
applies the sole FP32 AdamW update. ANE computes forward and `dX`; single-thread
Accelerate computes FP32 `dW` directly from shared FP32 `X`/`dY` buffers while
ANE computes `dX`. There is no Python, NumPy, or intermediate host tensor copy
inside the native measured step. Build it with:

```sh
xcrun clang -fobjc-arc -O3 -DACCELERATE_NEW_LAPACK \
  -framework Foundation -framework CoreVideo -framework IOSurface \
  -framework Metal -framework Accelerate \
  native/ane_metal/ane_cpu_metal_projection_triad.m \
  -o /private/tmp/theseus_native_projection_triad
```

`scripts/native_ane_cpu_metal_projection_qualification.py` freezes identical
setup tensors for the native and compiled-MLX routes, alternates two 64-update
rounds, compares full output/`dX`/`dW` and optimizer tensors, and rejects the
route unless its slowest round beats MLX's fastest round. The M1 receipt passes
all registered numerical, custody, replay, save/reload, finite-state, resource,
and zero-swap checks, but the native station averages `4.344–4.459 ms` versus
`3.582–3.624 ms` for MLX. The exact projection offload is therefore not
selected. This does not falsify ANE recomputation, whole-microbatch work, or
other long-window uses that avoid this station's synchronization pattern.

`ane_swiglu_activation_recompute.m` closes the next dependency-ordered
candidate. It reconstructs the batch-4, sequence-512, width-1,536 SwiGLU
gate/up activations as six compile-once 512-channel ANE chunks while Metal
packs FP32 inputs and weights directly into IOSurfaces. Full comparison against
Accelerate has zero errors above `0.001`. The recompute fits inside a later-layer
MLX attention backward window, but shared-resource contention expands the
joined critical path to `20.251 ms` versus `19.377 ms` for standalone MLX
(`0.956842x` mean control-over-candidate, `0.948546x` conservative). The exact
schedule is not selected and no custom backward is authorized from it. Its
288 MiB twelve-layer release is a tensor-size ceiling, not an observed
allocator or larger-batch result.

The same native triad now has a `--gradient-only` mode for whole-microbatch
work. It returns forward, `dX`, and FP32 `dW` while explicitly performing zero
local optimizer steps. `scripts/heterogeneous_microbatch_contract.py` rejects
mixed generations, sampler overlap/gaps, local normalization, non-FP32 or
nonfinite gradients, and per-device updates before one global clip/AdamW
publication. The exact two-shard `q_proj` qualification passes full
gradient/update parity and reaches `1.167119x` mean and `1.108747x`
conservative critical-path speedup versus two MLX shards. It remains
station-only: the Python/NumPy qualification join and missing complete ANE
decoder gradient tree block production eligibility.

`configs/exact_decoder_block_qualification.json` and
`scripts/exact_decoder_block_qualification.py` now freeze the next native
target before importing more kernels. The independent PyTorch/MLX reference
covers the exact width-512, FFN-1536, 8-query/2-KV-head feature geometry,
contiguous 4:1 GQA ownership, split-half RoPE, unscaled residuals, both
RMSNorm leaves, all seven bias-free linear leaves, one masked scalar loss,
`dX`, and all nine parameter gradients (3,015,680 parameters). The reference
gate is GREEN with `4.18e-6` maximum output delta, `1.50e-8` loss delta,
`2.68e-9` input-gradient delta, `4.20e-9` worst parameter-gradient delta, and
exact replay. This is the numerical ABI for the native port, not evidence that
the native ANE block exists or is faster.

`ane_exact_attention_forward.m` implements the first native slice against that
ABI. It fuses dynamic attention-RMSNorm scale, mutable Q/K/V weights,
split-half RoPE, corrected contiguous 4:1 GQA, causal softmax, and the Q/K/V
and normalized-input taps required for backward. A runtime bisect found an
exact-shape M1 constraint: the natural `512 × 897` packed surface compiles but
fails evaluation with status `0x1d`; padding the one-position norm-scale
segment to 128 positions produces `512 × 1024`, after which passthrough,
RMSNorm, QKV, RoPE, and full attention all execute. The guarded full slice
averages `0.409 ms`, has zero registered mismatches, and has `0.00352` worst
absolute delta. This does not yet include attention backward, out projection,
residuals, SwiGLU, scalar loss, or an optimizer update.

`ane_exact_attention_backward.m` now closes the exact attention gradient tree.
The ANE graph emits full-head `dQ`, `dK`, and `dV`; contiguous KV reduction and
inverse split-half RoPE feed single-thread FP32 Accelerate Q/K/V gradients and
attention-RMSNorm `dX`/scale. The guarded ANE core averages `0.399 ms`, the six
projection-gradient GEMMs average `0.263 ms`, and the Accelerate operator
matches an independent scalar implementation within `1.94e-7`. Every
ANE-originated downstream delta remains inside its analytical FP16-boundary
propagation bound with zero registered mismatches.

`exact_decoder_block_remainder.m` closes the other side of the frozen block
boundary with native Metal elementwise/loss/reduction/update kernels and
single-thread FP32 Accelerate GEMMs. It covers out projection, residuals, the
second RMSNorm, SwiGLU/down, scalar loss, all five remainder parameter leaves,
both returned boundary gradients, one global clip, and one AdamW publication.
All `2,621,952` gradient elements participate, replay is byte-exact, and 64
updates remain finite. The guarded mean is `6.678 ms`. The attention and
remainder executables are not yet one generation-tagged transaction, so this
is not a complete block or speedup claim.

`exact_decoder_block_join.m` composes both owners into one process and one
generation-tagged IOSurface transaction. The joined path executes compile-once
ANE forward/backward, native Metal remainder work, single-thread FP32
Accelerate gradients, one combined hidden gradient, and one global
norm/clip/AdamW publication across all nine leaves (`3,015,680` parameters).
Replay is exact, the initial transaction has full per-leaf gradient coverage,
and 64 joined updates remain finite. The guarded 64-update mean is `10.334 ms`
with zero swap growth. A matched compiled-MLX control, file save/reload, and
sustained thermal evidence remain mandatory before selection.

The alternating matched selector has now closed this exact route. Two native
64-update means (`10.176`, `10.353 ms`) lose to compiled MLX (`5.809`,
`5.537 ms`) under the same mixed-precision, parameter, objective, clip,
update, replay, and stability authority. Mean and conservative control-over-
candidate ratios are `0.552679x` and `0.534825x`, so compiled MLX remains
selected. This is an exact batch-one/sequence-128 engineering disposition,
not a broad ANE falsification.

The public Core ML alternative is owned by
`scripts/coreml_state_weight_probe.py`. It requires `coremltools==9.0`, builds
only temporary model packages, and records compute-plan placement, a
nonidentity state transition, output parity, and state-update versus read-only
timing. `scripts/mlx_fp16_projection_control.py` is its exact-shape MLX control.
`scripts/cpu_gpu_ane_coexistence_probe.py` then alternates standalone order and
launches all three workers concurrently from
`configs/cpu_gpu_ane_coexistence_m1.json`. That receipt measures mechanics only:
the workers are not yet one Theseus optimizer step and their rates cannot be
summed into a training speedup.

Prior art was inspected, not vendored:

- `maderix/ANE` at `d91c9845c0784dec7753048954fc6d0e8411fe29`
- `Mininglamp-AI/cider` at `4d91fcee9439f7aea17ae6e965271d9536c604a0`

Both are MIT licensed. Their private-API and hardware results are research
evidence, not Theseus production claims.
