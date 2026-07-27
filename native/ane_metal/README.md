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

Prior art was inspected, not vendored:

- `maderix/ANE` at `d91c9845c0784dec7753048954fc6d0e8411fe29`
- `Mininglamp-AI/cider` at `4d91fcee9439f7aea17ae6e965271d9536c604a0`

Both are MIT licensed. Their private-API and hardware results are research
evidence, not Theseus production claims.
