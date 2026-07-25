from __future__ import annotations

import json
import gzip
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from moecot_language_tokenizer import encode_document  # noqa: E402
from moecot_source_conditioned_pretraining import (  # noqa: E402
    encode_kerc_global_target,
    kerc_code_space,
    kerc_code_tokens,
)
from theseus_corpus_acceleration import (  # noqa: E402
    ARRAY_NAMES,
    CATEGORIES,
    compare_restart_identity,
    compare_runs,
    python_materialize,
)


@pytest.fixture(scope="module")
def corpus_binary() -> Path:
    subprocess.run(
        ["cargo", "build", "-p", "theseus-corpus"], cwd=ROOT, check=True
    )
    path = ROOT / "target/debug/theseus-corpus"
    assert path.is_file()
    return path


@pytest.fixture()
def exact_vocab(tmp_path: Path) -> tuple[dict[str, int], Path]:
    tokens = ["<pad>", "<unk>", "<bos>", "<target_token_bytes>", "</target_token_bytes>"]
    tokens.extend(f"<byte:{value:02x}>" for value in range(256))
    tokens.extend(["hello", " ", "\n", "<bytes:20202020>", "<bytes:64656620>"])
    vocab = {token: index for index, token in enumerate(tokens)}
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"target_vocab": vocab}), encoding="utf-8")
    return vocab, path


def write_sample(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, category in enumerate(CATEGORIES):
            handle.write(
                json.dumps(
                    {
                        "id": f"{category}:{index}",
                        "category": category,
                        "text": texts[index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_rust_encoder_matches_python_on_adversarial_language_boundaries(
    corpus_binary: Path, exact_vocab: tuple[dict[str, int], Path], tmp_path: Path
) -> None:
    vocab, vocab_path = exact_vocab
    texts = [
        "hello\r\n  world λ🙂",
        "$value 0x1_f 0b10_ 12.5e-2\n",
        "def f(x: str):\n    return x[::-1]\n",
        "const x = `a\\n${value}`; // 🙂\n",
        "<main data-x=\"a&b\"><style>.x { color: red; }</style></main>",
        "fn main() { println!(\"héllo\"); }\n",
    ]
    sample = tmp_path / "sample.jsonl"
    output = tmp_path / "encoded.jsonl"
    write_sample(sample, texts)
    subprocess.run(
        [
            str(corpus_binary),
            "encode",
            "--vocab",
            str(vocab_path),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--include-tokens",
        ],
        check=True,
    )
    observed = [json.loads(row) for row in output.read_text(encoding="utf-8").splitlines()]
    for category, text, row in zip(CATEGORIES, texts, observed):
        tokens, ids, receipt = encode_document(text, vocab, category=category)
        assert row["logical_tokens"] == tokens
        assert row["ids"] == ids
        assert row["receipt"]["fallback_token_count"] == receipt["fallback_token_count"]
        assert row["receipt"]["fallback_byte_count"] == receipt["fallback_byte_count"]
        assert row["receipt"]["unknown_token_count"] == 0
        assert row["receipt"]["exact_text_equal"] is True


def test_rust_materializer_is_byte_identical_to_python_reference(
    corpus_binary: Path, exact_vocab: tuple[dict[str, int], Path], tmp_path: Path
) -> None:
    _vocab, vocab_path = exact_vocab
    sample = tmp_path / "sample.jsonl"
    write_sample(
        sample,
        [
            "hello world\n" * 3,
            "A broad English paragraph with punctuation: yes.\n",
            "def answer(x):\n    return x + 1\n",
            "export const answer = (x) => x + 1;\n",
            "<button class=\"answer\">OK</button>\n",
            "pub fn answer(x: i32) -> i32 { x + 1 }\n",
        ],
    )
    python_output = tmp_path / "python"
    rust_output = tmp_path / "rust"
    reference = python_materialize(
        sample, vocab_path, python_output, target_offset=300, sequence_length=16
    )
    result = subprocess.run(
        [
            str(corpus_binary),
            "materialize",
            "--vocab",
            str(vocab_path),
            "--input",
            str(sample),
            "--output-dir",
            str(rust_output),
            "--target-offset",
            "300",
            "--sequence-length",
            "16",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    candidate = json.loads(result.stdout)
    parity = compare_runs([reference], [candidate])
    assert parity["state"] == "GREEN"
    second_output = tmp_path / "rust-restart"
    second_result = subprocess.run(
        [
            str(corpus_binary),
            "materialize",
            "--vocab",
            str(vocab_path),
            "--input",
            str(sample),
            "--output-dir",
            str(second_output),
            "--target-offset",
            "300",
            "--sequence-length",
            "16",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    restarted = json.loads(second_result.stdout)
    assert compare_restart_identity([candidate, restarted])["exact_restart"] is True
    verified = subprocess.run(
        [
            str(corpus_binary),
            "verify-materialization",
            "--output-dir",
            str(rust_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(verified.stdout)["state"] == "GREEN"
    for name in ARRAY_NAMES.values():
        assert (python_output / name).read_bytes() == (rust_output / name).read_bytes()

    with (rust_output / ARRAY_NAMES["labels"]).open("ab") as sink:
        sink.write(b"corruption")
    corrupted = subprocess.run(
        [
            str(corpus_binary),
            "verify-materialization",
            "--output-dir",
            str(rust_output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert corrupted.returncode != 0
    assert "integrity fault: labels" in corrupted.stderr


def test_compressed_and_parallel_materialization_preserve_exact_artifacts(
    corpus_binary: Path, exact_vocab: tuple[dict[str, int], Path], tmp_path: Path
) -> None:
    zstandard = pytest.importorskip("zstandard")
    _vocab, vocab_path = exact_vocab
    sample = tmp_path / "sample.jsonl"
    write_sample(
        sample,
        [
            "English instruction with a deliberately_long_identifier_0123456789.\n" * 8,
            "Broad prose 1234567890.123e-9 and punctuation.\n" * 8,
            "def answer(value):\n    return value + 1\n" * 8,
            "export const answer = (value) => value + 1;\n" * 8,
            '<main data-value="0123456789abcdef">OK</main>\n' * 8,
            "pub fn answer(value: i64) -> i64 { value + 1 }\n" * 8,
        ],
    )
    gzip_path = tmp_path / "sample.jsonl.gz"
    with gzip.open(gzip_path, "wb") as sink:
        sink.write(sample.read_bytes())
    zstd_path = tmp_path / "sample.jsonl.zst"
    zstd_path.write_bytes(zstandard.ZstdCompressor(level=3).compress(sample.read_bytes()))

    outputs: list[Path] = []
    receipts: list[dict] = []
    for index, source in enumerate((sample, gzip_path, zstd_path)):
        output = tmp_path / f"materialized-{index}"
        result = subprocess.run(
            [
                str(corpus_binary),
                "materialize",
                "--vocab",
                str(vocab_path),
                "--input",
                str(source),
                "--output-dir",
                str(output),
                "--target-offset",
                "300",
                "--sequence-length",
                "32",
                "--workers",
                "3",
                "--chunk-rows",
                "2",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        outputs.append(output)
        receipts.append(json.loads(result.stdout))

    assert [row["input_codec"] for row in receipts] == ["plain", "gzip", "zstd"]
    assert all(row["worker_count"] == 3 for row in receipts)
    assert all(row["deterministic_row_order"] is True for row in receipts)
    for name in ARRAY_NAMES.values():
        expected = (outputs[0] / name).read_bytes()
        assert expected == (outputs[1] / name).read_bytes()
        assert expected == (outputs[2] / name).read_bytes()


def test_content_addressed_cache_reuses_evicts_and_rejects_corruption(
    corpus_binary: Path, exact_vocab: tuple[dict[str, int], Path], tmp_path: Path
) -> None:
    _vocab, vocab_path = exact_vocab
    first_source = tmp_path / "first.jsonl"
    write_sample(first_source, ["hello world\n" * 4] * len(CATEGORIES))
    cache = tmp_path / "cache"

    def invoke(source: Path, budget: int = 1_000_000) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(corpus_binary),
                "cache-materialize",
                "--vocab",
                str(vocab_path),
                "--input",
                str(source),
                "--cache-root",
                str(cache),
                "--target-offset",
                "300",
                "--sequence-length",
                "16",
                "--workers",
                "2",
                "--chunk-rows",
                "2",
                "--cache-budget-bytes",
                str(budget),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    cold = invoke(first_source)
    assert cold.returncode == 0, cold.stderr
    cold_receipt = json.loads(cold.stdout)
    assert cold_receipt["cache_hit"] is False
    warm = invoke(first_source)
    assert warm.returncode == 0, warm.stderr
    warm_receipt = json.loads(warm.stdout)
    assert warm_receipt["cache_hit"] is True
    assert warm_receipt["entry_path"] == cold_receipt["entry_path"]
    assert warm_receipt["materialization_identity_sha256"] == cold_receipt[
        "materialization_identity_sha256"
    ]
    assert warm_receipt["manifest_sha256"] == cold_receipt["manifest_sha256"]

    first_entry = Path(cold_receipt["entry_path"])
    second_source = tmp_path / "second.jsonl"
    write_sample(second_source, ["hello changed world\n" * 4] * len(CATEGORIES))
    second = invoke(second_source)
    assert second.returncode == 0, second.stderr
    second_receipt = json.loads(second.stdout)
    second_entry = Path(second_receipt["entry_path"])
    entry_size = lambda path: sum(row.stat().st_size for row in path.rglob("*") if row.is_file())
    budget = max(entry_size(first_entry), entry_size(second_entry)) + 128
    evicted = invoke(second_source, budget=budget)
    assert evicted.returncode == 0, evicted.stderr
    second_receipt = json.loads(evicted.stdout)
    assert second_receipt["cache"]["evicted_entry_count"] == 1
    assert not first_entry.exists()

    with (second_entry / ARRAY_NAMES["mask"]).open("ab") as sink:
        sink.write(b"corruption")
    corrupted = invoke(second_source, budget=budget)
    assert corrupted.returncode != 0
    assert "integrity fault: mask" in corrupted.stderr


def test_oversized_unknown_rejects_without_publishing_partial_tensors(
    corpus_binary: Path, exact_vocab: tuple[dict[str, int], Path], tmp_path: Path
) -> None:
    _vocab, vocab_path = exact_vocab
    sample = tmp_path / "oversized.jsonl"
    write_sample(sample, [" " * 513, "ok", "ok", "ok", "ok", "ok"])
    output = tmp_path / "rejected"
    result = subprocess.run(
        [
            str(corpus_binary),
            "materialize",
            "--vocab",
            str(vocab_path),
            "--input",
            str(sample),
            "--output-dir",
            str(output),
            "--target-offset",
            "300",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unrepresentable document" in result.stderr
    assert not output.exists()


def test_rust_kerc_dual_space_encoding_matches_typed_python_reference(
    corpus_binary: Path, exact_vocab: tuple[dict[str, int], Path], tmp_path: Path
) -> None:
    vocab, _vocab_path = exact_vocab
    code_vocab_path = tmp_path / "code-vocab.json"
    code_vocab_path.write_text(
        json.dumps({"kernel_vocab": vocab, "pointer_vocab": vocab}), encoding="utf-8"
    )
    rows = [
        {
            "id": "kernel-1",
            "objective": "surface_to_kernel_program_v1",
            "target": '["KNODE_CALL","@A",-12.5e+2,"KPROGRAM_END"]',
        },
        {
            "id": "kernel-2",
            "objective": "kernel_program_to_answer_packet_v1",
            "target": '{"KANSWER_VERSION:1":"P1","value":"héllo🙂"}\n',
        },
    ]
    sample = tmp_path / "kerc.jsonl"
    sample.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "kerc-output.jsonl"
    subprocess.run(
        [
            str(corpus_binary),
            "kerc-encode",
            "--code-vocab",
            str(code_vocab_path),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--kernel-offset",
            "1000",
            "--pointer-offset",
            "2000",
            "--include-tokens",
        ],
        check=True,
    )
    observed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    code_vocab = {"kernel_vocab": vocab, "pointer_vocab": vocab}
    for source, candidate in zip(rows, observed):
        tokens = kerc_code_tokens(source["target"])
        ids, receipt = encode_kerc_global_target(
            source["target"],
            code_vocabulary=code_vocab,
            kernel_offset=1000,
            pointer_offset=2000,
        )
        assert candidate["ids"] == ids
        assert candidate["tokens"] == [
            {"text": str(token), "space": kerc_code_space(token)} for token in tokens
        ]
        assert candidate["receipt"]["encoded_tokens_by_space"] == receipt[
            "encoded_tokens_by_space"
        ]
        assert candidate["receipt"]["unknown_token_count"] == 0
        assert candidate["receipt"]["exact_text_equal"] is True
