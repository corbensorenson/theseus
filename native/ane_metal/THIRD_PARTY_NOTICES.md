# Third-party notices

The private Apple Neural Engine runtime invocation, weight-blob layout, MIL
RMSNorm-backward formulation in `ane_metal_same_surface_probe.m`, split-half
RoPE probe runtime in `ane_split_half_rope_probe.m`, the compile-once dynamic
QKV/attention transport in `ane_exact_attention_forward.m`, the causal
softmax-gradient construction in `ane_exact_attention_backward.m`, and the
auditable GQA repair in `patches/maderix_dynamic_gqa_grouping.patch` are
derived from or modify `maderix/ANE` at commit
`d91c9845c0784dec7753048954fc6d0e8411fe29`.

Copyright (c) maderix

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
