#!/usr/bin/env python3
"""Discover and verify render-killed semantic HTML implementation holes.

The adapter is deliberately source-span based.  It never derives a target from
tests or screenshots: candidate selection is frozen from licensed source text,
then an independent local-browser mutation check decides whether removing the
subtree destroys observable DOM/render behavior.
"""

from __future__ import annotations

import fnmatch
import hashlib
import html.parser
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from PIL import Image, ImageChops, __version__ as PILLOW_VERSION

from neural_seed_functional_verifiers import CHROME, _render_chrome


VERIFIER_ABI = "project_theseus_html_css_render_killed_subtree_v1"
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
DEFAULT_SEMANTIC_TAGS = {
    "article", "aside", "details", "dialog", "div", "fieldset", "figure",
    "footer", "form", "header", "main", "nav", "ol", "section", "table", "ul",
}
UNSAFE_EMBEDDED_TAGS = {"iframe", "object", "embed"}
NON_RENDERED_ANCESTORS = {"head", "script", "style", "template", "noscript"}
HIDDEN_CLASS_TOKENS = {"d-none", "hidden", "is-hidden", "uk-hidden", "visually-hidden"}
TEMPLATE_MARKERS = ("{{", "}}", "{%", "%}", "<%", "%>", "@@include")
VIEWPORTS = ((800, 600, "wide"), (375, 667, "narrow"))


class SemanticSpanParser(html.parser.HTMLParser):
    """Record exact, balanced element spans without normalizing source text."""

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = [0]
        self.stack: list[dict[str, Any]] = []
        self.spans: list[dict[str, Any]] = []
        self.discarded_unbalanced_count = 0
        self._attrs_by_start: dict[int, list[tuple[str, str | None]]] = {}
        self.line_offsets.extend(match.end() for match in re.finditer("\n", source))

    def char_offset(self) -> int:
        line, column = self.getpos()
        return self.line_offsets[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        start = self.char_offset()
        raw = self.get_starttag_text() or ""
        end = start + len(raw)
        tag = tag.lower()
        if tag in VOID_TAGS:
            return
        self.stack.append({
            "tag": tag,
            "start_char": start,
            "opening_end_char": end,
            "attrs": attrs,
            "depth": len(self.stack),
            "non_rendered_ancestor": any(
                row["tag"] in NON_RENDERED_ANCESTORS or row.get("statically_hidden")
                for row in self.stack
            ),
            "statically_hidden": attrs_statically_hidden(attrs),
        })

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        match_index = next(
            (index for index in range(len(self.stack) - 1, -1, -1)
             if self.stack[index]["tag"] == tag),
            None,
        )
        if match_index is None:
            self.discarded_unbalanced_count += 1
            return
        if match_index != len(self.stack) - 1:
            self.discarded_unbalanced_count += len(self.stack) - match_index - 1
            del self.stack[match_index + 1:]
        opened = self.stack.pop()
        closing_start = self.char_offset()
        closing_end = self.source.find(">", closing_start)
        if closing_end < 0:
            self.discarded_unbalanced_count += 1
            return
        self.spans.append({
            **opened,
            "closing_start_char": closing_start,
            "end_char": closing_end + 1,
        })

    def close(self) -> None:
        super().close()
        self.discarded_unbalanced_count += len(self.stack)
        self.stack.clear()


class DOMSignature(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: Counter[str] = Counter()
        self.attrs: Counter[str] = Counter()
        self.visible_text_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.tags[tag] += 1
        for key, value in attrs:
            key = key.lower()
            if key in {"role", "aria-label", "aria-labelledby", "alt", "for", "scope"}:
                self.attrs[f"{tag}:{key}:{value or ''}"] += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        self.visible_text_chars += len(" ".join(data.split()))


def discover_html_subtree_holes(source_root: Path, source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semantic_tags = {str(value).lower() for value in source.get("semantic_tags", DEFAULT_SEMANTIC_TAGS)}
    minimum_bytes = int(source.get("minimum_target_bytes", 384))
    maximum_bytes = int(source.get("maximum_target_bytes", 16384))
    minimum_text = int(source.get("minimum_visible_text_chars", 24))
    minimum_tags = int(source.get("minimum_descendant_tag_count", 2))
    maximum_document_bytes = int(source.get("maximum_document_bytes", 240000))
    paths = selected_source_paths(source_root, source)
    holes: list[dict[str, Any]] = []
    diagnostics = Counter()
    for path in paths:
        raw = path.read_bytes()
        if len(raw) > maximum_document_bytes:
            diagnostics["document_too_large"] += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            diagnostics["non_utf8_document"] += 1
            continue
        parser = SemanticSpanParser(text)
        try:
            parser.feed(text)
            parser.close()
        except Exception:
            diagnostics["parser_fault"] += 1
            continue
        diagnostics["discarded_unbalanced_elements"] += parser.discarded_unbalanced_count
        relative = path.relative_to(source_root).as_posix()
        source_sha = sha256_text(text)
        for span in parser.spans:
            if span["tag"] not in semantic_tags:
                continue
            if span.get("non_rendered_ancestor") or span.get("statically_hidden"):
                diagnostics["statically_non_rendered_rejected"] += 1
                continue
            target = text[span["start_char"]:span["end_char"]]
            target_bytes = len(target.encode("utf-8"))
            if not minimum_bytes <= target_bytes <= maximum_bytes:
                diagnostics["target_size_rejected"] += 1
                continue
            lowered = target.lower()
            if any(f"<{tag}" in lowered for tag in UNSAFE_EMBEDDED_TAGS):
                diagnostics["embedded_content_rejected"] += 1
                continue
            if any(marker in target for marker in TEMPLATE_MARKERS):
                diagnostics["template_syntax_rejected"] += 1
                continue
            signature = dom_signature(target)
            descendant_tags = sum(signature["tags"].values()) - 1
            if signature["visible_text_chars"] < minimum_text or descendant_tags < minimum_tags:
                diagnostics["insufficient_semantics"] += 1
                continue
            opening = text[span["start_char"]:span["opening_end_char"]]
            closing = text[span["closing_start_char"]:span["end_char"]]
            if not opening.startswith("<") or not closing.lower().startswith(f"</{span['tag']}"):
                diagnostics["source_span_integrity_rejected"] += 1
                continue
            hole_key = sha256_text(f"{relative}:{span['start_char']}:{target}")[:16]
            visible_starter = (
                opening
                + f"\n<!-- <THESEUS_IMPLEMENTATION_HOLE:{hole_key}> -->\n"
                + closing
            )
            visible_source = text[:span["start_char"]] + visible_starter + text[span["end_char"]:]
            if target in visible_source:
                diagnostics["target_repeated_in_visible_context"] += 1
                continue
            existing_id = re.search(
                r"\bid\s*=\s*(['\"])(.*?)\1", opening,
                flags=re.IGNORECASE | re.DOTALL,
            )
            marker_id = (
                str(existing_id.group(2))
                if existing_id and existing_id.group(2).strip()
                else f"theseus-hole-{hole_key}"
            )
            target_render_body = add_marker_id(opening, marker_id) + target[len(opening):]
            starter_render_body = add_marker_id(opening, marker_id) + closing
            hole = {
                "path": relative,
                "tag": span["tag"],
                "depth": span["depth"],
                "start_char": span["start_char"],
                "opening_end_char": span["opening_end_char"],
                "closing_start_char": span["closing_start_char"],
                "end_char": span["end_char"],
                "start_byte": len(text[:span["start_char"]].encode("utf-8")),
                "end_byte": len(text[:span["end_char"]].encode("utf-8")),
                "source_sha256": source_sha,
                "target_sha256": sha256_text(target),
                "target_body": target,
                "visible_source": visible_source,
                "visible_starter_body": visible_starter,
                "target_render_body": target_render_body,
                "starter_render_body": starter_render_body,
                "marker_id": marker_id,
                "static_signature": signature,
                "selection_key": sha256_json({
                    "path": relative,
                    "tag": span["tag"],
                    "target_sha256": sha256_text(target),
                    "start_char": span["start_char"],
                }),
            }
            if reconstruct_document(text, hole, target) != text:
                diagnostics["source_reconstruction_failed"] += 1
                continue
            holes.append(hole)
    holes.sort(key=lambda row: (row["path"], row["start_char"], row["end_char"]))
    return holes, {
        "source_file_count": len(paths),
        "candidate_count": len(holes),
        "candidate_target_bytes": sum(len(row["target_body"].encode("utf-8")) for row in holes),
        "diagnostics": dict(sorted(diagnostics.items())),
    }


def selected_source_paths(source_root: Path, source: dict[str, Any]) -> list[Path]:
    includes = [str(value) for value in source.get("source_globs", ["**/*.html", "**/*.htm"])]
    excludes = [str(value) for value in source.get("exclude_source_globs", [])]
    paths: set[Path] = set()
    for pattern in includes:
        paths.update(path.resolve() for path in source_root.glob(pattern) if path.is_file())
    root = source_root.resolve()
    selected = []
    for path in sorted(paths):
        if path != root and root not in path.parents:
            continue
        relative = path.relative_to(root).as_posix()
        if any(part.startswith(".theseus-") for part in Path(relative).parts):
            continue
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in excludes):
            continue
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        selected.append(path)
    return selected


def select_verification_candidates(source: dict[str, Any], holes: list[dict[str, Any]]) -> dict[str, Any]:
    policy = source.get("verification_candidate_selection") or {
        "kind": "content_hash_file_round_robin_v1",
        "maximum_candidates": 0,
        "maximum_candidates_per_file": 0,
        "selection_uses_verifier_outcomes": False,
    }
    if policy.get("kind") != "content_hash_file_round_robin_v1":
        raise ValueError("unsupported HTML/CSS verification candidate selection policy")
    if policy.get("selection_uses_verifier_outcomes") is not False:
        raise ValueError("HTML/CSS verification selection may not use verifier outcomes")
    maximum = max(0, int(policy.get("maximum_candidates") or 0))
    per_file = max(0, int(policy.get("maximum_candidates_per_file") or 0))
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(holes):
        grouped[row["path"]].append((index, row))
    for rows in grouped.values():
        rows.sort(key=lambda pair: (pair[1]["selection_key"], pair[1]["start_char"]))
        if per_file:
            del rows[per_file:]
    paths = sorted(grouped, key=lambda path: sha256_text(f"{source['id']}:{path}"))
    selected_indexes: list[int] = []
    round_index = 0
    while paths and (not maximum or len(selected_indexes) < maximum):
        next_paths = []
        for path in paths:
            rows = grouped[path]
            if round_index < len(rows):
                selected_indexes.append(rows[round_index][0])
                if maximum and len(selected_indexes) >= maximum:
                    break
            if round_index + 1 < len(rows):
                next_paths.append(path)
        if maximum and len(selected_indexes) >= maximum:
            break
        paths = next_paths
        round_index += 1
    selected = [holes[index] for index in selected_indexes]
    inventory_records = [selection_record(row) for row in holes]
    selected_records = [selection_record(row) for row in selected]
    return {
        "selected": selected,
        "selected_indexes": selected_indexes,
        "public_receipt": {
            "kind": policy["kind"],
            "selection_uses_verifier_outcomes": False,
            "maximum_candidates": maximum,
            "maximum_candidates_per_file": per_file,
            "inventory_candidate_count": len(holes),
            "selected_candidate_count": len(selected),
            "unselected_candidate_count": len(holes) - len(selected),
            "selected_source_file_count": len({row["path"] for row in selected}),
            "ordered_inventory_sha256": sha256_json(inventory_records),
            "selected_inventory_sha256": sha256_json(selected_records),
            "rationale": str(policy.get("rationale") or ""),
        },
    }


def verify_html_subtree_hole(
    source_root: Path,
    hole: dict[str, Any],
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    source_path = (source_root / hole["path"]).resolve()
    root = source_root.resolve()
    if source_path != root and root not in source_path.parents:
        return failed_receipt("source_path_escape", toolchain)
    if not source_path.is_file():
        return failed_receipt("source_file_missing", toolchain)
    try:
        original_bytes = source_path.read_bytes()
        original = original_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return failed_receipt("source_read_fault", toolchain)
    if sha256_text(original) != hole["source_sha256"]:
        return failed_receipt("source_identity_mismatch", toolchain)
    start, end = int(hole["start_char"]), int(hole["end_char"])
    if original[start:end] != hole["target_body"]:
        return failed_receipt("source_span_mismatch", toolchain)
    target_document = reconstruct_document(original, hole, hole["target_render_body"])
    starter_document = reconstruct_document(original, hole, hole["starter_render_body"])
    static_target = dom_signature(hole["target_body"])
    static_starter = dom_signature(hole["starter_render_body"])
    assertions = signature_delta(static_starter, static_target)
    render = render_document_pair(
        source_root=root,
        source_path=source_path,
        original_bytes=original_bytes,
        target_document=target_document,
        starter_document=starter_document,
        marker_id=str(hole["marker_id"]),
        toolchain=toolchain,
    )
    restored = source_path.is_file() and source_path.read_bytes() == original_bytes
    minimum_fraction = float(toolchain.get("minimum_changed_pixel_fraction", 0.0005))
    viewport_results = render.get("viewports") or []
    render_passed = bool(
        len(viewport_results) == len(VIEWPORTS)
        and all(
            row.get("target", {}).get("ok")
            and row.get("starter", {}).get("ok")
            and float(row.get("pixel_delta", {}).get("changed_pixel_fraction") or 0.0)
                >= minimum_fraction
            for row in viewport_results
        )
    )
    passed = bool(
        assertions["target_specific_assertion_count"] > 0
        and render_passed
        and restored
        and render.get("fault") is None
    )
    return {
        "kind": VERIFIER_ABI,
        "strength": "dom_a11y_layout_render_delta",
        "state": "passed" if passed else "failed",
        "target_static_signature": static_target,
        "starter_static_signature": static_starter,
        "target_specific_assertions": assertions,
        "render": render,
        "target_rendered": bool(viewport_results and all(row.get("target", {}).get("ok") for row in viewport_results)),
        "starter_rendered": bool(viewport_results and all(row.get("starter", {}).get("ok") for row in viewport_results)),
        "both_viewports_changed": render_passed,
        "minimum_changed_pixel_fraction": minimum_fraction,
        "source_restored": restored,
        "network_policy": "remote_ip_denied_by_macos_sandbox",
        "toolchain": web_toolchain_identity(toolchain),
    }


def render_document_pair(
    *,
    source_root: Path,
    source_path: Path,
    original_bytes: bytes,
    target_document: str,
    starter_document: str,
    marker_id: str,
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    if not CHROME.is_file():
        return {"fault": "chrome_unavailable", "viewports": []}
    timeout = int(toolchain["timeout_seconds"])
    rows: list[dict[str, Any]] = []
    fault: str | None = None
    render_path: Path | None = None
    try:
        descriptor, raw_render_path = tempfile.mkstemp(
            prefix=".theseus-render-", suffix=source_path.suffix, dir=source_path.parent
        )
        os.close(descriptor)
        render_path = Path(raw_render_path).resolve()
        with tempfile.TemporaryDirectory(prefix=".theseus-web-hole-", dir=source_root) as raw:
            workspace = Path(raw).resolve()
            for width, height, label in VIEWPORTS:
                target_shot = workspace / f"{label}-target.png"
                starter_shot = workspace / f"{label}-starter.png"
                render_path.write_text(target_document, encoding="utf-8")
                target_run = run_chrome_render(
                    source_root, render_path, target_shot, workspace / f"chrome-{label}-target",
                    width, height, marker_id, timeout,
                )
                render_path.write_text(starter_document, encoding="utf-8")
                starter_run = run_chrome_render(
                    source_root, render_path, starter_shot, workspace / f"chrome-{label}-starter",
                    width, height, marker_id, timeout,
                )
                delta = image_delta(target_shot, starter_shot)
                rows.append({
                    "label": label,
                    "width": width,
                    "height": height,
                    "target": public_render_receipt(target_run, target_shot),
                    "starter": public_render_receipt(starter_run, starter_shot),
                    "pixel_delta": delta,
                })
    except Exception as exc:  # noqa: BLE001 - verifier faults fail closed in receipt.
        fault = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if render_path is not None:
            try:
                render_path.unlink(missing_ok=True)
            except OSError as exc:
                fault = f"temporary_render_cleanup_failed:{type(exc).__name__}:{exc}"[:500]
        if not source_path.is_file() or source_path.read_bytes() != original_bytes:
            fault = "source_changed_during_verification"
    return {"fault": fault, "viewports": rows}


def run_chrome_render(
    source_root: Path,
    source_path: Path,
    screenshot: Path,
    user_data: Path,
    width: int,
    height: int,
    marker_id: str,
    timeout: int,
) -> dict[str, Any]:
    url = f"{source_path.as_uri()}#{quote(marker_id, safe='')}"
    command = [
        str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-javascript", "--disable-extensions", "--disable-sync",
        "--disable-background-networking", "--disable-component-update",
        "--disable-crash-reporter", "--disable-crashpad", "--disable-breakpad",
        "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
        "--force-device-scale-factor=1", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=250", f"--user-data-dir={user_data}",
        f"--window-size={width},{height}", f"--screenshot={screenshot}", url,
    ]
    return _render_chrome(command, source_root, screenshot, timeout)


def image_delta(target: Path, starter: Path) -> dict[str, Any]:
    if not target.is_file() or not starter.is_file():
        return {"ok": False, "fault": "screenshot_missing", "changed_pixel_fraction": 0.0}
    try:
        with Image.open(target) as target_image, Image.open(starter) as starter_image:
            target_rgb = target_image.convert("RGB")
            starter_rgb = starter_image.convert("RGB")
            if target_rgb.size != starter_rgb.size:
                return {"ok": False, "fault": "screenshot_dimension_mismatch", "changed_pixel_fraction": 0.0}
            difference = ImageChops.difference(target_rgb, starter_rgb).convert("L")
            histogram = difference.histogram()
            threshold = 8
            changed = sum(histogram[threshold:])
            total = target_rgb.size[0] * target_rgb.size[1]
            weighted = sum(index * count for index, count in enumerate(histogram))
            return {
                "ok": True,
                "width": target_rgb.size[0],
                "height": target_rgb.size[1],
                "changed_pixel_count": changed,
                "changed_pixel_fraction": round(changed / total if total else 0.0, 8),
                "mean_absolute_luma_delta": round(weighted / total if total else 0.0, 6),
                "threshold": threshold,
            }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "fault": f"image_decode_fault:{type(exc).__name__}", "changed_pixel_fraction": 0.0}


def dom_signature(text: str) -> dict[str, Any]:
    parser = DOMSignature()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return {"tags": {}, "accessibility_attrs": {}, "visible_text_chars": 0}
    return {
        "tags": dict(sorted(parser.tags.items())),
        "accessibility_attrs": dict(sorted(parser.attrs.items())),
        "visible_text_chars": parser.visible_text_chars,
    }


def attrs_statically_hidden(attrs: Iterable[tuple[str, str | None]]) -> bool:
    values = {str(key).lower(): str(value or "") for key, value in attrs}
    if "hidden" in values or values.get("aria-hidden", "").lower() == "true":
        return True
    style = re.sub(r"\s+", "", values.get("style", "").lower())
    if "display:none" in style or "visibility:hidden" in style:
        return True
    classes = set(values.get("class", "").lower().split())
    return bool(classes & HIDDEN_CLASS_TOKENS)


def signature_delta(starter: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    assertions = []
    for section in ("tags", "accessibility_attrs"):
        for key, count in target[section].items():
            if int(count) > int(starter[section].get(key, 0)):
                assertions.append(f"{section}:{key}>={count}")
    if int(target["visible_text_chars"]) > int(starter["visible_text_chars"]):
        assertions.append(f"visible_text_chars>={target['visible_text_chars']}")
    return {
        "target_specific_assertion_count": len(assertions),
        "assertions": assertions[:200],
        "starter_fails_at_least_one": bool(assertions),
    }


def reconstruct_document(source: str, hole: dict[str, Any], replacement: str) -> str:
    return source[:int(hole["start_char"])] + replacement + source[int(hole["end_char"]):]


def add_marker_id(opening_tag: str, marker_id: str) -> str:
    existing = re.search(r"\bid\s*=\s*(['\"])(.*?)\1", opening_tag, flags=re.IGNORECASE | re.DOTALL)
    if existing:
        # Preserve the original id and add a zero-cost alias immediately before
        # the close; duplicate id attributes would make browser behavior undefined.
        return opening_tag[:opening_tag.rfind(">")] + f' data-theseus-hole="{marker_id}"' + opening_tag[opening_tag.rfind(">"):]
    close = opening_tag.rfind(">")
    if close < 0:
        raise ValueError("opening tag has no terminator")
    return opening_tag[:close] + f' id="{marker_id}" data-theseus-hole="true"' + opening_tag[close:]


def selection_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "start_char": row["start_char"],
        "end_char": row["end_char"],
        "tag": row["tag"],
        "source_sha256": row["source_sha256"],
        "target_sha256": row["target_sha256"],
        "selection_key": row["selection_key"],
    }


def web_toolchain_identity(toolchain: dict[str, Any]) -> dict[str, Any]:
    return {
        **toolchain,
        "verifier_abi": VERIFIER_ABI,
        "chrome_path": str(CHROME),
        "chrome_version": chrome_version(),
        "pillow_version": PILLOW_VERSION,
        "viewports": [list(row) for row in VIEWPORTS],
    }


@lru_cache(maxsize=1)
def chrome_version() -> str:
    if not CHROME.is_file():
        return "unavailable"
    try:
        completed = subprocess.run(
            [str(CHROME), "--version"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=False,
        )
        return completed.stdout.strip() or completed.stderr.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "probe_failed"


def public_render_receipt(run: dict[str, Any], screenshot: Path) -> dict[str, Any]:
    return {
        "ok": bool(run.get("ok")),
        "returncode": run.get("returncode"),
        "fault": run.get("fault"),
        "duration_ms": run.get("duration_ms"),
        "screenshot_sha256": file_sha256(screenshot) if screenshot.is_file() else None,
        "screenshot_bytes": screenshot.stat().st_size if screenshot.is_file() else 0,
        "stdout_sha256": sha256_text(str(run.get("stdout") or "")),
        "stderr_sha256": sha256_text(str(run.get("stderr") or "")),
    }


def failed_receipt(reason: str, toolchain: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": VERIFIER_ABI,
        "strength": "dom_a11y_layout_render_delta",
        "state": "failed",
        "reason": reason,
        "target_rendered": False,
        "starter_rendered": False,
        "both_viewports_changed": False,
        "source_restored": reason != "source_restore_failed",
        "toolchain": web_toolchain_identity(toolchain),
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
