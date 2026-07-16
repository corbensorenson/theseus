#!/usr/bin/env python3
"""Discover and verify source-bound CSS rule implementation holes."""

from __future__ import annotations

import fnmatch
import hashlib
import html.parser
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageChops, __version__ as PILLOW_VERSION

from neural_seed_functional_verifiers import CHROME, _render_chrome, _sandbox_profile


VERIFIER_ABI = "project_theseus_css_render_killed_rule_v1"
VIEWPORTS = ((800, 600, "wide"), (375, 667, "narrow"))
UNSAFE_SELECTOR_MARKERS = (":hover", ":active", ":focus", ":visited", "::before", "::after")


class LinkSpanParser(html.parser.HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = [0]
        self.line_offsets.extend(match.end() for match in re.finditer("\n", source))
        self.links: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "link":
            return
        values = {str(key).lower(): str(value or "") for key, value in attrs}
        if "stylesheet" not in values.get("rel", "").lower().split():
            return
        line, column = self.getpos()
        start = self.line_offsets[line - 1] + column
        raw = self.get_starttag_text() or ""
        self.links.append({"start": start, "end": start + len(raw), "href": values.get("href", "")})


def parse_css_records(
    paths: list[Path], parser_toolchain: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        return [], {"parsed_file_count": 0, "parse_error_count": 0}
    with tempfile.TemporaryDirectory(prefix="theseus-css-ast-") as raw:
        workdir = Path(raw).resolve()
        input_path = workdir / "input.json"
        output_path = workdir / "output.json"
        input_path.write_text(json.dumps({
            "toolchainRoot": parser_toolchain["runtime_root"],
            "paths": [str(path) for path in paths],
        }), encoding="utf-8")
        command = [
            parser_toolchain["node_executable"], parser_toolchain["parser_script"],
            str(input_path), str(output_path),
        ]
        completed = subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", _sandbox_profile(workdir), *command],
            cwd=workdir,
            env={"HOME": str(Path.home()), "PATH": os.environ.get("PATH", ""), "TMPDIR": str(workdir)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode or not output_path.is_file():
            raise RuntimeError(f"CSS AST parser failed: {completed.stderr[-1000:]}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("postcssVersion") != parser_toolchain["postcss_version"]:
        raise RuntimeError("CSS AST parser version drift")
    return list(payload.get("records") or []), {
        "parsed_file_count": len(paths),
        "parse_error_count": len(payload.get("parseErrors") or {}),
        "parse_errors": payload.get("parseErrors") or {},
        "postcss_version": payload.get("postcssVersion"),
    }


def discover_css_rule_holes(
    source_root: Path,
    source: dict[str, Any],
    parser_toolchain: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = source_root.resolve()
    css_paths = selected_paths(
        root,
        source.get("source_globs", ["**/*.css"]),
        {".css"},
        excludes=source.get("exclude_source_globs", []),
    )
    page_paths = selected_paths(
        root,
        source.get("page_globs", ["**/*.html"]),
        {".html", ".htm"},
        excludes=source.get("exclude_page_globs", []),
    )
    page_rows = [page_record(root, path) for path in page_paths]
    records, parser_receipt = parse_css_records(css_paths, parser_toolchain)
    minimum_bytes = int(source.get("minimum_target_bytes", 64))
    maximum_bytes = int(source.get("maximum_target_bytes", 4096))
    context_chars = int(source.get("context_characters_each_side", 12000))
    page_excerpt_chars = int(source.get("page_excerpt_characters", 12000))
    alias_policy = source.get("stylesheet_href_aliases") or {}
    holes: list[dict[str, Any]] = []
    rejected: dict[str, int] = defaultdict(int)
    css_by_path = {str(path): path for path in css_paths}
    for record in records:
        path = css_by_path.get(str(Path(record["path"]).resolve()))
        if path is None:
            rejected["parser_path_outside_inventory"] += 1
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        start = int(record["start_char"])
        end = int(record["end_char"])
        target = str(record["target_body"])
        if text[start:end] != target:
            rejected["source_span_mismatch"] += 1
            continue
        target_bytes = len(target.encode("utf-8"))
        if not minimum_bytes <= target_bytes <= maximum_bytes:
            rejected["target_size_rejected"] += 1
            continue
        selector = str(record["selector"]).strip()
        if any(marker in selector.lower() for marker in UNSAFE_SELECTOR_MARKERS):
            rejected["interaction_selector_rejected"] += 1
            continue
        aliases = [relative, *[str(value) for value in alias_policy.get(relative, [])]]
        matching_pages = [row for row in page_rows if page_links_stylesheet(row, root, path, aliases)]
        if not matching_pages:
            rejected["no_linked_page"] += 1
            continue
        selector_tokens = tokens_for_selector(selector)
        ranked_pages = sorted(
            matching_pages,
            key=lambda row: (
                -page_selector_overlap(row["source"], selector_tokens),
                sha256_text(f"{relative}:{selector}:{row['relative']}")
            ),
        )
        page = ranked_pages[0]
        if selector_tokens and page_selector_overlap(page["source"], selector_tokens) == 0:
            rejected["selector_absent_from_linked_pages"] += 1
            continue
        hole_key = sha256_text(f"{relative}:{start}:{target}")[:16]
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        stylesheet_context = (
            text[context_start:start]
            + f"\n/* <THESEUS_CSS_IMPLEMENTATION_HOLE:{hole_key}> */\n"
            + text[end:context_end]
        )
        page_excerpt = relevant_page_excerpt(page["source"], selector_tokens, page_excerpt_chars)
        if target in stylesheet_context or target in page_excerpt:
            rejected["target_repeated_in_visible_context"] += 1
            continue
        holes.append({
            "path": relative,
            "page_path": page["relative"],
            "stylesheet_aliases": aliases,
            "start_char": start,
            "end_char": end,
            "start_byte": int(record["start_byte"]),
            "end_byte": int(record["end_byte"]),
            "selector": selector,
            "declaration_count": int(record["declaration_count"]),
            "ancestor_kinds": list(record.get("ancestor_kinds") or []),
            "source_sha256": sha256_text(text),
            "page_sha256": sha256_text(page["source"]),
            "target_sha256": sha256_text(target),
            "target_body": target,
            "stylesheet_context": stylesheet_context,
            "page_excerpt": page_excerpt,
            "selection_key": sha256_json({
                "path": relative,
                "page_path": page["relative"],
                "start_char": start,
                "target_sha256": sha256_text(target),
            }),
        })
    holes.sort(key=lambda row: (row["path"], row["start_char"], row["page_path"]))
    return holes, {
        "css_file_count": len(css_paths),
        "page_file_count": len(page_paths),
        "candidate_count": len(holes),
        "candidate_target_bytes": sum(len(row["target_body"].encode("utf-8")) for row in holes),
        "parser": parser_receipt,
        "rejected": dict(sorted(rejected.items())),
    }


def select_verification_candidates(source: dict[str, Any], holes: list[dict[str, Any]]) -> dict[str, Any]:
    policy = source.get("verification_candidate_selection") or {}
    if policy.get("kind") != "content_hash_file_round_robin_v1":
        raise ValueError("unsupported CSS verification candidate selection policy")
    if policy.get("selection_uses_verifier_outcomes") is not False:
        raise ValueError("CSS verification selection may not use verifier outcomes")
    maximum = max(0, int(policy.get("maximum_candidates") or 0))
    per_file = max(0, int(policy.get("maximum_candidates_per_file") or 0))
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(holes):
        grouped[row["path"]].append((index, row))
    for rows in grouped.values():
        rows.sort(key=lambda pair: (pair[1]["selection_key"], pair[1]["start_char"]))
        if per_file:
            del rows[per_file:]
    paths = sorted(grouped, key=lambda value: sha256_text(f"{source['id']}:{value}"))
    selected_indexes: list[int] = []
    round_index = 0
    while paths and (not maximum or len(selected_indexes) < maximum):
        next_paths = []
        for value in paths:
            rows = grouped[value]
            if round_index < len(rows):
                selected_indexes.append(rows[round_index][0])
                if maximum and len(selected_indexes) >= maximum:
                    break
            if round_index + 1 < len(rows):
                next_paths.append(value)
        if maximum and len(selected_indexes) >= maximum:
            break
        paths = next_paths
        round_index += 1
    selected = [holes[index] for index in selected_indexes]
    records = [selection_record(row) for row in holes]
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
            "selected_source_file_count": len({row["path"] for row in selected}),
            "ordered_inventory_sha256": sha256_json(records),
            "selected_inventory_sha256": sha256_json(selected_records),
            "rationale": str(policy.get("rationale") or ""),
        },
    }


def verify_css_rule_hole(
    source_root: Path, hole: dict[str, Any], toolchain: dict[str, Any]
) -> dict[str, Any]:
    root = source_root.resolve()
    css_path = (root / hole["path"]).resolve()
    page_path = (root / hole["page_path"]).resolve()
    if not within(root, css_path) or not within(root, page_path):
        return failed_receipt("source_path_escape", toolchain)
    try:
        css_bytes = css_path.read_bytes()
        page_bytes = page_path.read_bytes()
        css = css_bytes.decode("utf-8")
        page = page_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return failed_receipt("source_read_fault", toolchain)
    if sha256_text(css) != hole["source_sha256"] or sha256_text(page) != hole["page_sha256"]:
        return failed_receipt("source_identity_mismatch", toolchain)
    start, end = int(hole["start_char"]), int(hole["end_char"])
    if css[start:end] != hole["target_body"]:
        return failed_receipt("source_span_mismatch", toolchain)
    starter_css = css[:start] + f"/* THESEUS_CSS_RULE_HOLE:{hole['target_sha256'][:16]} */" + css[end:]
    render = render_css_pair(
        root=root,
        css_path=css_path,
        page_path=page_path,
        page=page,
        target_css=css,
        starter_css=starter_css,
        aliases=list(hole["stylesheet_aliases"]),
        toolchain=toolchain,
    )
    unchanged = css_path.read_bytes() == css_bytes and page_path.read_bytes() == page_bytes
    minimum = float(toolchain.get("minimum_changed_pixel_fraction", 0.0005))
    viewports = render.get("viewports") or []
    renders_succeeded = bool(
        len(viewports) == len(VIEWPORTS)
        and all(row.get("target", {}).get("ok") and row.get("starter", {}).get("ok") for row in viewports)
    )
    changed_viewports = [
        str(row.get("label")) for row in viewports
        if float(row.get("pixel_delta", {}).get("changed_pixel_fraction") or 0.0) >= minimum
    ]
    passed = bool(renders_succeeded and changed_viewports and unchanged and render.get("fault") is None)
    return {
        "kind": VERIFIER_ABI,
        "strength": "layout_render_delta",
        "state": "passed" if passed else "failed",
        "selector_sha256": sha256_text(str(hole["selector"])),
        "declaration_count": int(hole["declaration_count"]),
        "render": render,
        "all_viewports_rendered": renders_succeeded,
        "changed_viewports": changed_viewports,
        "changed_viewport_count": len(changed_viewports),
        "minimum_changed_viewports": 1,
        "minimum_changed_pixel_fraction": minimum,
        "canonical_sources_unchanged": unchanged,
        "network_policy": "remote_ip_denied_by_macos_sandbox",
        "toolchain": css_toolchain_identity(toolchain),
    }


def render_css_pair(
    *,
    root: Path,
    css_path: Path,
    page_path: Path,
    page: str,
    target_css: str,
    starter_css: str,
    aliases: list[str],
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    if not CHROME.is_file():
        return {"fault": "chrome_unavailable", "viewports": []}
    timeout = int(toolchain["timeout_seconds"])
    rows: list[dict[str, Any]] = []
    fault: str | None = None
    page_temp: Path | None = None
    css_temp: Path | None = None
    try:
        page_fd, page_name = tempfile.mkstemp(prefix=".theseus-css-page-", suffix=page_path.suffix, dir=page_path.parent)
        css_fd, css_name = tempfile.mkstemp(prefix=".theseus-css-rule-", suffix=css_path.suffix, dir=css_path.parent)
        os.close(page_fd)
        os.close(css_fd)
        page_temp = Path(page_name).resolve()
        css_temp = Path(css_name).resolve()
        linked_page = replace_stylesheet_link(page, page_path, css_temp, aliases)
        with tempfile.TemporaryDirectory(prefix=".theseus-css-render-", dir=root) as raw:
            workdir = Path(raw).resolve()
            page_temp.write_text(linked_page, encoding="utf-8")
            for width, height, label in VIEWPORTS:
                target_shot = workdir / f"{label}-target.png"
                starter_shot = workdir / f"{label}-starter.png"
                css_temp.write_text(target_css, encoding="utf-8")
                target_run = run_chrome(root, page_temp, target_shot, workdir / f"chrome-{label}-target", width, height, timeout)
                css_temp.write_text(starter_css, encoding="utf-8")
                starter_run = run_chrome(root, page_temp, starter_shot, workdir / f"chrome-{label}-starter", width, height, timeout)
                rows.append({
                    "label": label,
                    "width": width,
                    "height": height,
                    "target": public_render_receipt(target_run, target_shot),
                    "starter": public_render_receipt(starter_run, starter_shot),
                    "pixel_delta": image_delta(target_shot, starter_shot),
                })
    except Exception as exc:  # noqa: BLE001
        fault = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        for temporary in (page_temp, css_temp):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    fault = f"temporary_render_cleanup_failed:{type(exc).__name__}:{exc}"[:500]
    return {"fault": fault, "viewports": rows}


def replace_stylesheet_link(page: str, page_path: Path, css_temp: Path, aliases: list[str]) -> str:
    parser = LinkSpanParser(page)
    parser.feed(page)
    parser.close()
    normalized_aliases = {normalize_href(value) for value in aliases}
    matches = [row for row in parser.links if normalize_href(row["href"]) in normalized_aliases]
    if not matches:
        raise ValueError("configured stylesheet link not found in page")
    relative = os.path.relpath(css_temp, page_path.parent).replace(os.sep, "/")
    output = page
    for row in reversed(matches):
        original = output[row["start"]:row["end"]]
        replacement = re.sub(
            r"(\bhref\s*=\s*)(['\"])(.*?)\2",
            lambda match: f"{match.group(1)}{match.group(2)}{relative}{match.group(2)}",
            original,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if replacement == original:
            raise ValueError("stylesheet href rewrite failed")
        output = output[:row["start"]] + replacement + output[row["end"]:]
    return output


def page_record(root: Path, path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    parser = LinkSpanParser(source)
    parser.feed(source)
    parser.close()
    return {
        "path": path,
        "relative": path.relative_to(root).as_posix(),
        "source": source,
        "links": [row["href"] for row in parser.links],
    }


def page_links_stylesheet(
    page: dict[str, Any], root: Path, css_path: Path, aliases: list[str]
) -> bool:
    normalized_aliases = {normalize_href(value) for value in aliases}
    for href in page["links"]:
        normalized = normalize_href(href)
        if normalized in normalized_aliases:
            return True
        if not urlsplit(href).scheme and not href.startswith("//"):
            candidate = (page["path"].parent / unquote(urlsplit(href).path)).resolve()
            if candidate == css_path and within(root, candidate):
                return True
    return False


def tokens_for_selector(selector: str) -> list[str]:
    tokens = re.findall(r"[.#]([A-Za-z_][A-Za-z0-9_-]*)", selector)
    if not tokens:
        tokens = re.findall(r"(?:^|[\s>,+~])([A-Za-z][A-Za-z0-9-]*)", selector)
    return sorted({value.lower() for value in tokens if len(value) >= 2})


def page_selector_overlap(page: str, tokens: list[str]) -> int:
    lowered = page.lower()
    return sum(bool(re.search(rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])", lowered)) for token in tokens)


def relevant_page_excerpt(page: str, tokens: list[str], maximum: int) -> str:
    if len(page) <= maximum:
        return page
    lowered = page.lower()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    center = min(positions) if positions else len(page) // 2
    start = max(0, center - maximum // 2)
    end = min(len(page), start + maximum)
    return page[start:end]


def selected_paths(
    root: Path, patterns: list[str], suffixes: set[str], *, excludes: list[str] | tuple[str, ...] = ()
) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path.resolve() for path in root.glob(str(pattern)) if path.is_file())
    selected = []
    for path in paths:
        if not within(root, path) or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root).as_posix()
        if any(part.startswith(".theseus-") for part in Path(relative).parts):
            continue
        if any(fnmatch.fnmatchcase(relative, str(pattern)) for pattern in excludes):
            continue
        selected.append(path)
    return sorted(selected)


def run_chrome(
    root: Path, page: Path, screenshot: Path, user_data: Path,
    width: int, height: int, timeout: int,
) -> dict[str, Any]:
    command = [
        str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu",
        "--disable-javascript", "--disable-extensions", "--disable-sync",
        "--disable-background-networking", "--disable-component-update",
        "--disable-crash-reporter", "--disable-crashpad", "--disable-breakpad",
        "--no-first-run", "--no-default-browser-check", "--hide-scrollbars",
        "--force-device-scale-factor=1", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=250", f"--user-data-dir={user_data}",
        f"--window-size={width},{height}", f"--screenshot={screenshot}", page.as_uri(),
    ]
    return _render_chrome(command, root, screenshot, timeout)


def image_delta(target: Path, starter: Path) -> dict[str, Any]:
    if not target.is_file() or not starter.is_file():
        return {"ok": False, "fault": "screenshot_missing", "changed_pixel_fraction": 0.0}
    try:
        with Image.open(target) as target_image, Image.open(starter) as starter_image:
            target_rgb = target_image.convert("RGB")
            starter_rgb = starter_image.convert("RGB")
            if target_rgb.size != starter_rgb.size:
                return {"ok": False, "fault": "screenshot_dimension_mismatch", "changed_pixel_fraction": 0.0}
            histogram = ImageChops.difference(target_rgb, starter_rgb).convert("L").histogram()
            changed = sum(histogram[8:])
            total = target_rgb.size[0] * target_rgb.size[1]
            weighted = sum(index * count for index, count in enumerate(histogram))
            return {
                "ok": True,
                "width": target_rgb.size[0],
                "height": target_rgb.size[1],
                "changed_pixel_count": changed,
                "changed_pixel_fraction": round(changed / total if total else 0.0, 8),
                "mean_absolute_luma_delta": round(weighted / total if total else 0.0, 6),
                "threshold": 8,
            }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "fault": f"image_decode_fault:{type(exc).__name__}", "changed_pixel_fraction": 0.0}


def css_toolchain_identity(toolchain: dict[str, Any]) -> dict[str, Any]:
    return {
        **toolchain,
        "verifier_abi": VERIFIER_ABI,
        "verifier_source_sha256": file_sha256(Path(__file__).resolve()),
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
        "strength": "layout_render_delta",
        "state": "failed",
        "reason": reason,
        "all_viewports_rendered": False,
        "changed_viewports": [],
        "changed_viewport_count": 0,
        "minimum_changed_viewports": 1,
        "canonical_sources_unchanged": not reason.endswith("source_changed"),
        "toolchain": css_toolchain_identity(toolchain),
    }


def selection_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": row["path"],
        "page_path": row["page_path"],
        "start_char": row["start_char"],
        "end_char": row["end_char"],
        "selector_sha256": sha256_text(row["selector"]),
        "source_sha256": row["source_sha256"],
        "target_sha256": row["target_sha256"],
        "selection_key": row["selection_key"],
    }


def normalize_href(value: str) -> str:
    return unquote(urlsplit(str(value)).path).replace("\\", "/").lstrip("./")


def within(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
