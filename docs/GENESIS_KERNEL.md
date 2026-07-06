# Genesis Kernel For Theseus

Genesis is now treated as the artifact substrate around Theseus rather than a separate platform. RMI, Benchmaxxing, Cognitive Loop Closure, Octopus Router, and SymLiquid keep their jobs; Genesis compiles their live evidence into typed artifacts that can be checked, released, reused, and improved.

The local vertical slice is:

```text
live reports -> artifacts -> claim ledger -> critique log -> primitive candidates
-> release manifest -> artifact debt -> feedback plan
```

The Reality Manipulator MVP now sits beside Genesis as the world-facing intent
compiler:

```text
raw goal -> eight-limb spell -> portal world -> artifacts -> claims/critiques
-> specialist arms -> compile targets -> gates -> residuals -> feedback plan
```

Run it with:

```powershell
python scripts\reality_manipulator.py --out reports/reality_manipulator.json --markdown-out reports/reality_manipulator.md --bundle-dir reports/reality_manipulator/latest_world
```

Genesis treats `reports/reality_manipulator.json` as an optional source report.
The two layers have different jobs: Reality Manipulator structures new intent
into a world; Genesis compiles live Theseus evidence into releaseable artifact
memory.

Run it with:

```powershell
python scripts/genesis_kernel.py ingest-theseus --out reports/genesis_kernel/report.json --bundle-dir reports/genesis_kernel/latest_release
python scripts/genesis_kernel.py check --report reports/genesis_kernel/report.json
```

The generated release bundle contains:

- `artifacts.json`
- `edges.jsonl`
- `events.jsonl`
- `claim_ledger.csv`
- `critique_log.md`
- `primitive_candidates.json`
- `artifact_debt.json`
- `feedback_plan.md`
- `release_manifest.json`

The key rule is simple: claims, critiques, benchmark results, tool promotions, architecture decisions, and feedback should survive the run that created them. This lets Theseus keep cumulative invention memory without growing the student model just to compensate for lost context.
