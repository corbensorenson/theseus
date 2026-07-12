# Corpus Ingress Policy

Project Theseus keeps a hard boundary between model weights and training data.

Open/base/pretrained model weights remain forbidden for the from-scratch student
lanes. Theseus may train only weights it initializes and updates itself.

License-clean text and code are allowed as self-supervised pretraining data when
they are admitted through the governed corpus manifest. Every admitted source
must record provenance, license, path, SHA-256, content type, decontamination
status, and training use. Unknown-license sources are rejected.

Static openly licensed corpora may contain model-derived text. They are admitted as
data, not live teacher calls, only when quality tier, provenance class, license,
permitted use, contamination, retention, and recursive synthetic position share are
explicit. Live teacher authority remains OpenAI-only and external tokens are never
served at runtime.

Public benchmarks remain calibration-only. Public benchmark prompts, tests,
hidden tests, solutions, traces, answer templates, task ids, and benchmark-
derived metadata must not enter training rows or self-supervised corpus
payloads. Private in-family and family-disjoint eval rows are also treated as
firewalled scoreboards for corpus decontamination.

The current first real rung uses the local Apple Command Line Tools Python 3.9
standard-library source under the Python Software Foundation License 2.0. This
is code-rich, license-known, locally present on this Mac, and does not use any
open model weights or external inference.

Corpus pretraining is not promotion evidence by itself. Promotion still requires
the unchanged strict comparator, candidate-integrity audit, blind information-
flow audit, and no fallback/template/router/tool credit as learned generation.
