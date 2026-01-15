#!/usr/bin/env python3
"""
自举型 TDB 构建与校验流水线（full_pipeline / extract_only / validate_only）。

功能概览：
- 路由：根据输入文本或显式 --mode 选择 full_pipeline / extract_only / validate_only。
- 状态：记录 DOI、已下载文件、抽取的参数行、tdb 版本号、上次错误与 missing_hints，持久化到 JSON。
- 循环：full_pipeline 下持续迭代，直到 tdb_smoke_test 通过或达到最大迭代次数。
- 工具集（MVP）：web_search → fetch_source → pdf_extract_text → tdb_merge → tdb_smoke_test。
- 兼容增强：stub 出 supplement_discover/table_extract/tdb_lint/dedupe_and_rank_candidates，便于后续替换。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_TDB = ROOT / "thermo" / "Database" / "COST507.tdb"
DEFAULT_OUTPUT_TDB = ROOT / "thermo" / "Database" / "pipeline_output.tdb"
DEFAULT_STATE_PATH = ROOT / ".tdb_pipeline_state.json"


# ---------------------- 数据模型与状态 ---------------------- #
@dataclass
class SearchResult:
    title: str
    url: str
    doi: Optional[str] = None
    year: Optional[str] = None


@dataclass
class FetchResult:
    doi_or_url: str
    path: Optional[Path]
    auth_required: bool
    message: str


@dataclass
class ExtractedParam:
    line: str
    doi_or_url: Optional[str]
    page: Optional[int]
    evidence: Optional[str]


@dataclass
class SmokeTestResult:
    passed: bool
    error: Optional[str]
    missing_hints: List[str] = field(default_factory=list)


@dataclass
class WorkflowState:
    processed_dois: List[str] = field(default_factory=list)
    downloaded_sources: dict[str, str] = field(default_factory=dict)
    extracted_params: List[str] = field(default_factory=list)
    tdb_version: int = 0
    last_error: Optional[str] = None
    missing_hints: List[str] = field(default_factory=list)


def load_state(path: Path) -> WorkflowState:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return WorkflowState(**data)
        except Exception:
            return WorkflowState()
    return WorkflowState()


def save_state(path: Path, state: WorkflowState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------- 工具函数（MVP） ---------------------- #
def classify_message(message: str | None) -> str:
    if not message:
        return "full_pipeline"
    text = message.lower()
    if "validate" in text or "验证" in text:
        return "validate_only"
    if "extract" in text or "提取" in text:
        return "extract_only"
    return "full_pipeline"


def web_search(query: str, max_results: int = 5, hints: Optional[Sequence[str]] = None) -> List[SearchResult]:
    """
    简易 CrossRef 检索；若 requests 未安装或网络不可用，则返回空列表。
    """
    try:
        import requests  # type: ignore
    except Exception:
        return []

    params = {"query": query, "rows": max_results}
    if hints:
        params["query"] = f"{query} " + " ".join(hints)
    try:
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    results: List[SearchResult] = []
    try:
        items = resp.json().get("message", {}).get("items", [])
    except Exception:
        return []
    for item in items:
        doi = item.get("DOI")
        title_list = item.get("title") or []
        title = title_list[0] if title_list else doi or "untitled"
        url = f"https://doi.org/{doi}" if doi else item.get("URL", "")
        year = None
        if item.get("issued", {}).get("date-parts"):
            year = str(item["issued"]["date-parts"][0][0])
        results.append(SearchResult(title=title, url=url, doi=doi, year=year))
    return results


def fetch_source(doi_or_url: str, out_dir: Path) -> FetchResult:
    """
    下载 OA PDF；若遇到 401/403 则返回 auth_required=True。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / sanitize_filename(doi_or_url)  # temp name; may adjust when content-type known
    try:
        import requests  # type: ignore
    except Exception as exc:
        return FetchResult(doi_or_url=doi_or_url, path=None, auth_required=False, message=f"requests not available: {exc}")

    url = doi_or_url
    if doi_or_url.startswith("10."):
        url = f"https://doi.org/{doi_or_url}"
    try:
        with requests.get(url, stream=True, timeout=15, allow_redirects=True) as resp:
            if resp.status_code in (401, 403):
                return FetchResult(doi_or_url=doi_or_url, path=None, auth_required=True, message="Login required")
            resp.raise_for_status()
            # Guess filename
            content_type = resp.headers.get("content-type", "")
            ext = ".pdf" if "pdf" in content_type else ".bin"
            target_path = target_path.with_suffix(ext)
            with open(target_path, "wb") as f:
                shutil.copyfileobj(resp.raw, f)
    except Exception as exc:
        return FetchResult(doi_or_url=doi_or_url, path=None, auth_required=False, message=str(exc))

    return FetchResult(doi_or_url=doi_or_url, path=target_path, auth_required=False, message="ok")


def pdf_extract_text(pdf_path: Path, max_pages: int = 5) -> List[Tuple[int, str]]:
    """
    返回 (page_number, text) 列表；若缺依赖则返回空列表。
    """
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return []

    pages: List[Tuple[int, str]] = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for idx, page in enumerate(reader.pages[:max_pages]):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                pages.append((idx + 1, text))
    except Exception:
        return []
    return pages


def extract_params_from_pages(pages: List[Tuple[int, str]], source: str) -> List[ExtractedParam]:
    pattern = re.compile(r"^(FUNCTION|PARAMETER|TYPE_DEFINITION|PHASE|CONSTITUENT)", re.IGNORECASE)
    extracted: List[ExtractedParam] = []
    for page_no, text in pages:
        for line in text.splitlines():
            stripped = line.strip()
            if pattern.match(stripped):
                evidence = stripped[:200]
                extracted.append(ExtractedParam(line=stripped, doi_or_url=source, page=page_no, evidence=evidence))
    return extracted


def tdb_merge(
    base_tdb_path: Path,
    new_params: Sequence[ExtractedParam],
    fragments: Sequence[Path],
    out_tdb_path: Path,
) -> Path:
    base_text = ""
    if base_tdb_path and Path(base_tdb_path).exists():
        base_text = Path(base_tdb_path).read_text(encoding="utf-8")
    elif base_tdb_path:
        base_text = f"! Base TDB missing: {base_tdb_path}\n"
    else:
        base_text = "! Empty base.\n"

    seen = set(line.strip() for line in base_text.splitlines())
    lines: List[str] = base_text.splitlines()

    for frag in fragments:
        frag_path = Path(frag)
        if not frag_path.exists():
            continue
        frag_text = frag_path.read_text(encoding="utf-8")
        lines.append(f"! --- begin fragment {frag_path.name} ---")
        lines.extend(frag_text.splitlines())
        lines.append(f"! --- end fragment {frag_path.name} ---")

    for param in new_params:
        if param.line.strip() in seen:
            continue
        seen.add(param.line.strip())
        if param.evidence:
            lines.append(f"! source={param.doi_or_url or 'unknown'} page={param.page or '-'} evidence={param.evidence}")
        lines.append(param.line)

    out_tdb_path.parent.mkdir(parents=True, exist_ok=True)
    out_tdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_tdb_path


def tdb_smoke_test(
    tdb_path: Path,
    components: Sequence[str],
    phases: Optional[Sequence[str]],
    temperature: float,
    grid_points: int,
) -> SmokeTestResult:
    if not tdb_path.exists():
        return SmokeTestResult(passed=False, error=f"TDB not found: {tdb_path}", missing_hints=["provide_tdb"])
    try:
        from calphad.calphad_core import load_database, simple_equilibrium  # type: ignore
    except Exception as exc:
        return SmokeTestResult(passed=False, error=f"pycalphad not available: {exc}", missing_hints=["install_pycalphad"])

    try:
        db = load_database(tdb_path)
        _ = simple_equilibrium(db, components, phases=None if phases is None else list(phases), temperature=temperature, grid_points=grid_points)
    except Exception as exc:  # pragma: no cover - runtime dependent
        missing = infer_missing_from_error(str(exc))
        return SmokeTestResult(passed=False, error=str(exc), missing_hints=missing)

    return SmokeTestResult(passed=True, error=None, missing_hints=[])


# ---------------------- 增强工具 stub ---------------------- #
def supplement_discover(pdf_path: Path) -> List[str]:
    return []


def table_extract(pdf_path: Path) -> List[str]:
    return []


def tdb_lint(tdb_path: Path) -> List[str]:
    if not tdb_path.exists():
        return ["tdb_missing"]
    hints: List[str] = []
    text = tdb_path.read_text(encoding="utf-8")
    if "FUNCTION" not in text:
        hints.append("no_function_defined")
    return hints


def dedupe_and_rank_candidates(candidates: List[SearchResult], hints: Sequence[str]) -> List[SearchResult]:
    seen = set()
    ranked: List[SearchResult] = []
    for c in candidates:
        key = c.doi or c.url
        if not key or key in seen:
            continue
        seen.add(key)
        ranked.append(c)
    return ranked


# ---------------------- 核心控制逻辑 ---------------------- #
@dataclass
class WorkflowConfig:
    mode: str
    message: Optional[str]
    query: Optional[str]
    base_tdb: Path
    output_tdb: Path
    fragments: List[Path]
    sources: List[Path]
    new_lines: List[str]
    state_path: Path
    max_iters: int
    max_results: int
    max_pages: int
    components: List[str]
    phases: Optional[List[str]]
    temperature: float
    grid_points: int
    skip_validation: bool


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def infer_missing_from_error(error: str) -> List[str]:
    hints: List[str] = []
    lowered = error.lower()
    if "function" in lowered and "not found" in lowered:
        hints.append("missing_function_definition")
    if "phase" in lowered and "not found" in lowered:
        hints.append("missing_phase_definition")
    if "pycalphad" in lowered:
        hints.append("install_pycalphad")
    return hints or ["check_tdb_content"]


def run_full_pipeline(cfg: WorkflowConfig, state: WorkflowState) -> SmokeTestResult:
    hints = list(state.missing_hints)
    result = SmokeTestResult(passed=False, error="not_started", missing_hints=hints)
    for iteration in range(cfg.max_iters):
        print(f"[full_pipeline] iteration {iteration + 1}/{cfg.max_iters}")
        # 检索
        query = cfg.query or cfg.message or "Fe B TDB"
        candidates = web_search(query, max_results=cfg.max_results, hints=hints)
        candidates = dedupe_and_rank_candidates(candidates, hints)

        # 下载
        fetched_paths: List[Path] = []
        for cand in candidates:
            if cand.doi and cand.doi in state.processed_dois:
                continue
            fetch = fetch_source(cand.doi or cand.url, ROOT / "sources")
            if fetch.auth_required:
                state.missing_hints.append(f"login_required:{fetch.doi_or_url}")
                continue
            if fetch.path:
                fetched_paths.append(fetch.path)
                if cand.doi:
                    state.processed_dois.append(cand.doi)
                    state.downloaded_sources[cand.doi] = str(fetch.path)

        # 手动提供的 source 也纳入
        all_sources = list(cfg.sources) + fetched_paths

        # 抽取
        extracted_params: List[ExtractedParam] = []
        for src in all_sources:
            pages = pdf_extract_text(src, max_pages=cfg.max_pages)
            extracted_params.extend(extract_params_from_pages(pages, source=str(src)))
            # 尝试发现补充材料（stub）
            supplement_discover(src)

        # 追加手工 new_lines
        for line in cfg.new_lines:
            extracted_params.append(ExtractedParam(line=line.strip(), doi_or_url="manual", page=None, evidence=line.strip()[:100]))

        merged_tdb = tdb_merge(cfg.base_tdb, extracted_params, cfg.fragments, cfg.output_tdb)
        state.extracted_params.extend([p.line for p in extracted_params])
        state.tdb_version += 1
        lint_hints = tdb_lint(merged_tdb)

        if cfg.skip_validation:
            result = SmokeTestResult(passed=True, error=None, missing_hints=lint_hints)
            break

        result = tdb_smoke_test(merged_tdb, cfg.components, cfg.phases, cfg.temperature, cfg.grid_points)
        if lint_hints and not result.missing_hints:
            result.missing_hints = lint_hints
        state.last_error = result.error
        state.missing_hints = result.missing_hints
        if result.passed:
            break
        hints = result.missing_hints
    return result


def run_extract_only(cfg: WorkflowConfig, state: WorkflowState) -> SmokeTestResult:
    extracted_params: List[ExtractedParam] = []
    for line in cfg.new_lines:
        extracted_params.append(ExtractedParam(line=line.strip(), doi_or_url="manual", page=None, evidence=line.strip()[:100]))
    merged_tdb = tdb_merge(cfg.base_tdb, extracted_params, cfg.fragments, cfg.output_tdb)
    state.extracted_params.extend([p.line for p in extracted_params])
    state.tdb_version += 1
    return SmokeTestResult(passed=True, error=None, missing_hints=tdb_lint(merged_tdb))


def run_validate_only(cfg: WorkflowConfig, state: WorkflowState) -> SmokeTestResult:
    target = cfg.output_tdb if cfg.output_tdb.exists() else cfg.base_tdb
    result = tdb_smoke_test(target, cfg.components, cfg.phases, cfg.temperature, cfg.grid_points)
    state.last_error = result.error
    state.missing_hints = result.missing_hints
    return result


# ---------------------- CLI ---------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TDB builder/validator pipeline with persistent state.")
    parser.add_argument("--mode", choices=["full_pipeline", "extract_only", "validate_only"], help="Explicit mode; otherwise auto-classify from message.")
    parser.add_argument("--message", help="Free-text task description to classify.")
    parser.add_argument("--query", help="Search query override.")
    parser.add_argument("--base-tdb", type=Path, default=DEFAULT_BASE_TDB, help=f"Baseline TDB (default: {DEFAULT_BASE_TDB}).")
    parser.add_argument("--output-tdb", type=Path, default=DEFAULT_OUTPUT_TDB, help=f"Output TDB (default: {DEFAULT_OUTPUT_TDB}).")
    parser.add_argument("--fragments", type=Path, nargs="*", default=[], help="TDB fragments to merge.")
    parser.add_argument("--sources", type=Path, nargs="*", default=[], help="Pre-downloaded PDF/supplement sources.")
    parser.add_argument("--new-lines", type=Path, help="Text file containing TDB lines to merge (one per line).")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH, help=f"State file path (default: {DEFAULT_STATE_PATH}).")
    parser.add_argument("--max-iters", type=int, default=3, help="Max iterations for full_pipeline.")
    parser.add_argument("--max-results", type=int, default=5, help="Max search results to fetch per iteration.")
    parser.add_argument("--max-pages", type=int, default=5, help="Max pages per PDF to OCR/extract.")
    parser.add_argument("--components", nargs="*", default=["FE", "B", "VA"], help="Components for smoke test.")
    parser.add_argument("--phases", nargs="*", help="Phases for smoke test (optional).")
    parser.add_argument("--temp", type=float, default=1200.0, help="Temperature for smoke test.")
    parser.add_argument("--grid", type=int, default=9, help="Grid points per dimension for smoke test.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip smoke test (useful for offline extract-only).")
    parser.add_argument("--human", action="store_true", help="Human-readable output instead of JSON.")
    return parser.parse_args()


def load_new_lines(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"new-lines file not found: {path}")
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    mode = args.mode or classify_message(args.message)
    new_lines = load_new_lines(args.new_lines)
    cfg = WorkflowConfig(
        mode=mode,
        message=args.message,
        query=args.query,
        base_tdb=args.base_tdb,
        output_tdb=args.output_tdb,
        fragments=list(args.fragments),
        sources=list(args.sources),
        new_lines=new_lines,
        state_path=args.state_file,
        max_iters=args.max_iters,
        max_results=args.max_results,
        max_pages=args.max_pages,
        components=list(args.components),
        phases=list(args.phases) if args.phases else None,
        temperature=args.temp,
        grid_points=args.grid,
        skip_validation=args.skip_validation,
    )

    state = load_state(cfg.state_path)

    if cfg.mode == "full_pipeline":
        result = run_full_pipeline(cfg, state)
    elif cfg.mode == "extract_only":
        result = run_extract_only(cfg, state)
    else:
        result = run_validate_only(cfg, state)

    save_state(cfg.state_path, state)

    output = {
        "mode": cfg.mode,
        "status": "ok" if result.passed else "error",
        "output_text": result.error or "ok",
        "missing_hints": result.missing_hints,
        "output_tdb": str(cfg.output_tdb),
        "tdb_version": state.tdb_version,
    }
    if args.human:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] mode={cfg.mode} tdb={cfg.output_tdb}")
        if result.error:
            print(f"error: {result.error}")
        if result.missing_hints:
            print("missing_hints:", ", ".join(result.missing_hints))
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
