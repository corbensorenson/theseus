from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from open_code_training_pantry import language_for_extension, repos_from_config  # noqa: E402


def test_repo_config_deduplicates_and_respects_disabled_rows() -> None:
    config = {
        "repos": [
            {"repo": "owner/one"},
            {"repo": "owner/one"},
            {"repo": "owner/two", "enabled": False},
            "owner/three",
        ]
    }
    assert repos_from_config(config) == ["owner/one", "owner/three"]


def test_web_and_rust_extensions_map_to_canonical_languages() -> None:
    assert language_for_extension(".tsx") == "typescript"
    assert language_for_extension(".jsx") == "javascript"
    assert language_for_extension(".html") == "html"
    assert language_for_extension(".css") == "css"
    assert language_for_extension(".scss") == "css"
    assert language_for_extension(".less") == "css"
    assert language_for_extension(".rs") == "rust"
