from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_tokenizer_counts_complete_prompt_without_generation(tmp_path: Path) -> None:
    prompt = "Implement this task.\n" + ("source line\n" * 100)
    prompts = tmp_path / "prompts.json"
    out = tmp_path / "receipt.json"
    prompts.write_text(
        json.dumps(
            {
                "prompts": {
                    "direct_target_generation": prompt,
                    "natural_language_plan_control": prompt + "plan",
                    "typed_semantic_ir_treatment": prompt + "semantic",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(ROOT / "runtime/venvs/mlx-0.32.0-py312/bin/python"),
            str(ROOT / "scripts/theseus_exact_prompt_token_count.py"),
            "--worker-config",
            str(ROOT / "configs/core_evidence_tmax_9b_completion_worker.json"),
            "--prompts",
            str(prompts),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report = json.loads(out.read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert report["trigger_state"] == "GREEN"
    assert report["prompt_count"] == 3
    assert report["minimum_context_residual_tokens"] > 0
    assert report["candidate_or_control_calls"] == 0
    assert report["project_selected_quality_token_cap"] is None
    assert all(
        row["exact_chat_templated_prompt_tokens"] > 0
        for row in report["prompts"].values()
    )
