from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import theseus_vcm_language_replacements as replacements  # noqa: E402
import theseus_vcm_source_materialization as materialize  # noqa: E402


def test_real_preflight_is_green_and_call_free() -> None:
    report = replacements.preflight()
    assert report["trigger_state"] == "GREEN"
    assert report["state"] == "SIX_LANGUAGE_REPLACEMENT_PREFLIGHT_GREEN"
    assert report["replacement_set_admitted"] is False
    assert report["source_content_retrieval_opened"] is False
    assert all(value == 0 for value in report["counters"].values())


def test_bound_language_classifier_accepts_english_and_rejects_reviewed_families() -> None:
    config = replacements.p2a.read_json(replacements.DEFAULT_CONFIG)
    policy = config["english_language_policy"]
    accepted, receipt = replacements.classify_english("Fix the compatibility window and preserve retries", policy)
    assert accepted is True
    assert receipt["dominant_language"] == "en"
    for title in (
        "Tenant-Namen in der Seitenleiste anzeigen",
        "locație de start specifică",
        "fix: 리뷰 스킬의 build/test/lint 실행 요구 제거",
        "модель выбирает проект из существующих",
    ):
        accepted, receipt = replacements.classify_english(title, policy)
        assert accepted is False


def test_forbidden_script_filter_rejects_mixed_title_even_if_classifier_calls_it_english(monkeypatch) -> None:
    class Completed:
        stdout = '{"dominant_language":"en","hypotheses":[]}\n'

    monkeypatch.setattr(replacements.subprocess, "run", lambda *args, **kwargs: Completed())
    accepted, receipt = replacements.classify_english(
        "fix: 리뷰 build test requirements",
        {
            "swift_executable": "/usr/bin/swift",
            "module_cache_path": "/private/tmp/cache",
            "classifier_source": "scripts/theseus_language_scope.swift",
            "required_dominant_language": "en",
        },
    )
    assert accepted is False
    assert receipt["dominant_language"] == "en"
    assert receipt["forbidden_unicode_scripts"] == ["HANGUL"]


def test_full_mocked_sixty_page_selection_binds_exact_six_slots(tmp_path: Path, monkeypatch) -> None:
    config = replacements.p2a.read_json(replacements.DEFAULT_CONFIG)
    config["output_directory"] = str(tmp_path / "archives")
    config_path = tmp_path / "replacement.json"
    replacements.p2a.write_json(config_path, config)
    nodes: dict[str, dict[str, object]] = {}
    titles: dict[str, str] = {}

    def language_from_query(query: object) -> str:
        text = str(query)
        for language in ("JavaScript", "Python", "TypeScript"):
            if f"language:{language}" in text:
                return language
        raise AssertionError(text)

    def fake_rest(resource: str, fields: dict[str, object]):
        assert resource == "search/issues"
        language = language_from_query(fields["q"])
        items = []
        for index in range(8):
            slug = language.lower().replace("script", "s")
            repository = f"replacement-fixture/{slug}-{index:02d}"
            node_id = f"PR_{slug}_{index}"
            title = f"Fix {language} compatibility behavior {index}"
            titles[repository] = title
            items.append({
                "repository_url": f"https://api.github.com/repos/{repository}",
                "number": index + 1,
                "node_id": node_id,
                "title": title,
            })
            suffix = {"JavaScript": "js", "Python": "py", "TypeScript": "ts"}[language]
            nodes[node_id] = {
                "__typename": "PullRequest", "id": node_id, "number": index + 1,
                "url": f"https://github.com/{repository}/pull/{index + 1}",
                "state": "MERGED", "isDraft": False,
                "createdAt": "2026-08-01T00:00:00Z", "mergedAt": "2026-08-01T02:00:00Z",
                "additions": 5, "deletions": 1, "changedFiles": 2,
                "baseRefOid": "a" * 40, "headRefOid": "b" * 40,
                "mergeCommit": {"oid": "c" * 40}, "author": {"login": "human"},
                "repository": {
                    "nameWithOwner": repository, "isFork": False, "isArchived": False,
                    "isDisabled": False, "stargazerCount": 5,
                    "primaryLanguage": {"name": language}, "licenseInfo": {"spdxId": "MIT"},
                },
                "files": {"nodes": [
                    {"path": f"src/module.{suffix}", "changeType": "MODIFIED"},
                    {"path": f"tests/test_module.{suffix}", "changeType": "MODIFIED"},
                ]},
                "commits": {"nodes": [{"commit": {"oid": "b" * 40, "committedDate": "2026-08-01T01:00:00Z"}}]},
            }
        return {"items": items}, "a" * 64

    def fake_graphql(resource: str, fields: dict[str, object]):
        assert resource == "graphql:nodes"
        return {"data": {"nodes": [nodes[str(node_id)] for node_id in fields["ids"]]}}, "b" * 64

    class FakeSourceClient:
        def __init__(self) -> None:
            self.title_requests = 0
            self.source_requests = 0

        def title(self, repository: str, number: int) -> str:
            self.title_requests += 1
            return titles[repository]

        def license(self, repository: str, revision: str):
            self.source_requests += 1
            return "LICENSE", b"MIT fixture\n"

        def file(self, repository: str, revision: str, path: str):
            self.source_requests += 1
            return f"{path}:{revision}\n".encode()

    monkeypatch.setattr(replacements.v1, "api_json", fake_rest)
    monkeypatch.setattr(replacements.v6, "graphql_api", fake_graphql)
    monkeypatch.setattr(replacements.time, "sleep", lambda _seconds: None)
    retry_policy = config["transport_retry_policy"]
    ledger = materialize.SourceLedger(tmp_path / "checkpoint.json", config_path, retry_policy)
    client = FakeSourceClient()
    report = replacements.acquire(
        config_path,
        ledger,
        client,
        retry_policy,
        classifier=lambda title, policy: (True, {"dominant_language": "en", "accepted_english": True}),
    )
    assert report["trigger_state"] == "GREEN"
    assert report["replacement_set_admitted"] is True
    assert [row["index"] for row in report["replacement_rows"]] == [1, 12, 19, 48, 51, 56]
    assert [(row["panel"], row["query_language"]) for row in report["replacement_rows"]] == [
        ("claim", "JavaScript"), ("claim", "Python"), ("claim", "Python"),
        ("claim", "TypeScript"), ("claim", "TypeScript"),
        ("control_qualification", "Python"),
    ]
    assert len({row["repository"] for row in report["replacement_rows"]}) == 6
    assert report["counters"]["public_metadata_selection_requests"] == 33
    assert report["counters"]["source_archives_materialized"] == 24
    assert len(list((tmp_path / "archives").glob("*.tar.gz"))) == 24
