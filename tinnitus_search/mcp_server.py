from __future__ import annotations
# Путь: D:\Tinnitus-Search\tinnitus_search\mcp_server.py
#
# КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: все тяжёлые импорты (sentence-transformers, chromadb,
# indexer) перенесены внутрь get_state() — lazy loading.
# Это позволяет MCP-серверу мгновенно начать STDIO-хэндшейк с Qoder,
# а модель загружается только при первом вызове tool'а.

import contextlib
import io
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# ╔════════════════════════════════════════════════════════════════╗
# ║  НЕ импортируем здесь тяжёлые модули!                        ║
# ║  sentence_transformers, chromadb, indexer — только в get_state ║
# ╚════════════════════════════════════════════════════════════════╝

# -----------------------------
# Logging: keep stderr only
# -----------------------------
log = logging.getLogger("tinnitus_search.mcp")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,          # явно stderr, чтобы не мешать STDIO
)

# -----------------------------
# Env
# -----------------------------
REPO_ROOT = Path(os.environ.get("REPO_ROOT", ""))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ""))
COLLECTION = os.environ.get("CHROMA_COLLECTION", "tinnitus_repo_smart_v1")
MODEL_DIR = os.environ.get("MODEL_DIR", "")

MAX_PREVIEW_CHARS = int(os.environ.get("TS_MAX_PREVIEW_CHARS", "320"))
MAX_PER_FILE = int(os.environ.get("TS_MAX_PER_FILE", "2"))
OVERSAMPLE = int(os.environ.get("TS_OVERSAMPLE", "4"))


def _require_dir(p: Path, name: str) -> None:
    if not p or str(p) in ("", "."):
        raise ValueError(f"Missing env var {name}")
    if not p.exists():
        raise FileNotFoundError(f"{name} path does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {p}")


@contextlib.contextmanager
def _suppress_stdout():
    """
    Подавляем ТОЛЬКО stdout (JSON-RPC канал).
    stderr оставляем для логирования.
    """
    saved = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = saved


def _disable_hf_progress():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()
    except Exception:
        pass


def _posix_to_abs(file_path_posix: str) -> Path:
    return (REPO_ROOT / PurePosixPath(file_path_posix)).resolve()


def _flatten(results: Dict[str, Any], key: str) -> List[Any]:
    v = results.get(key)
    if v is None:
        return []
    if isinstance(v, list) and v and isinstance(v[0], list):
        return v[0]
    return v


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _make_preview(doc: str, max_chars: int) -> Dict[str, Any]:
    doc = doc or ""
    lines = doc.splitlines()
    header = lines[0].strip() if lines else ""

    signature = ""
    for ln in lines[1:12]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("def ", "async def ", "class ")):
            signature = s
            break

    snippet = doc.strip().replace("\r\n", "\n")[:max_chars]
    return {"header": header, "signature": signature, "text": snippet, "chars": len(snippet)}


def _diversify_by_file(sorted_hits: List[Dict[str, Any]], limit: int, max_per_file: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for h in sorted_hits:
        fp = h.get("file_path") or ""
        buckets.setdefault(fp, [])
        if len(buckets[fp]) < max_per_file:
            buckets[fp].append(h)

    out: List[Dict[str, Any]] = []
    round_i = 0
    while len(out) < limit:
        progressed = False
        for fp, lst in buckets.items():
            if round_i < len(lst):
                out.append(lst[round_i])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        round_i += 1
    return out


def _quality(best_sim: Optional[float], count: int) -> Dict[str, Any]:
    if count == 0:
        return {"best_similarity": None, "verdict": "none", "notes": ["No hits returned."]}
    if best_sim is None:
        return {"best_similarity": None, "verdict": "weak", "notes": ["Similarity not available."]}
    if best_sim >= 0.75:
        verdict = "good"
    elif best_sim >= 0.60:
        verdict = "ok"
    else:
        verdict = "weak"
    notes: List[str] = []
    if verdict == "weak":
        notes.append("Low semantic match. Consider rephrasing query or using grep fallback.")
    return {"best_similarity": best_sim, "verdict": verdict, "notes": notes}


# ══════════════════════════════════════════════════════════════════
#  MCP server — регистрация МГНОВЕННАЯ, тяжёлая работа — lazy
# ══════════════════════════════════════════════════════════════════
mcp = FastMCP(
    name="Tinnitus-Search (AST semantic repo search)",
    json_response=True,
    instructions="Use search_code then get_chunk_by_id for exact fragments; update_index if stale.",
)

_state_lock = Lock()
_warmup_started = False


@lru_cache(maxsize=1)
def get_state():
    """
    Lazy init — вызывается ТОЛЬКО при первом обращении к tool'у.
    Все тяжёлые импорты (torch, sentence-transformers, chromadb) — здесь.
    """
    with _state_lock:
        _require_dir(REPO_ROOT, "REPO_ROOT")

        if not CHROMA_DIR.exists():
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _require_dir(CHROMA_DIR, "CHROMA_DIR")

        if MODEL_DIR:
            os.environ["MODEL_DIR"] = MODEL_DIR

        _disable_hf_progress()

        log.info("Initializing embedder/indexer/store (lazy, first tool call)...")
        t0 = time.perf_counter()

        # ── Тяжёлые импорты ТОЛЬКО ЗДЕСЬ ──
        with _suppress_stdout():
            from tinnitus_search.embed.cpu import CPUEmbedder          # noqa: imports torch
            from tinnitus_search.index.indexer import Indexer           # noqa
            from tinnitus_search.index.chroma_store import ChromaStore  # noqa: imports chromadb

            _embedder = CPUEmbedder()
            _indexer = Indexer(repo_path=str(REPO_ROOT), persist_directory=str(CHROMA_DIR))

        store = getattr(_indexer, "store", None)
        if store is None:
            store = ChromaStore(persist_dir=str(CHROMA_DIR), collection_name=COLLECTION)

        if not hasattr(store, "get_by_ids"):
            def _get_by_ids(ids: List[str]):
                return store.collection.get(ids=ids, include=["documents", "metadatas"])
            setattr(store, "get_by_ids", _get_by_ids)

        ms = int((time.perf_counter() - t0) * 1000)
        log.info("State initialized in %s ms", ms)

        return store, _indexer


def _background_warmup():
    """
    Фоновый прогрев: запускается в отдельном потоке ПОСЛЕ handshake.
    Qoder сразу видит tools, а модель грузится параллельно.
    """
    try:
        log.info("Background warmup: starting heavy imports...")
        get_state()
        log.info("Background warmup: done, tools are ready")
    except Exception as e:
        log.error("Background warmup failed: %s", e)


# ══════════════════════════════════════════════════════════════════
#  Tools
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
def search_code(query: str, limit: int = 8, freshness: str = "ttl") -> Dict[str, Any]:
    store, indexer = get_state()

    t0 = time.perf_counter()

    t_idx0 = time.perf_counter()
    index_checked = False
    if freshness in ("ttl", "strict"):
        indexer.ensure_fresh_index()
        index_checked = True
    t_idx_ms = int((time.perf_counter() - t_idx0) * 1000)

    n_results = max(limit * OVERSAMPLE, limit)

    t_q0 = time.perf_counter()
    results = store.search(query=query, n_results=n_results)
    t_q_ms = int((time.perf_counter() - t_q0) * 1000)

    ids = _flatten(results, "ids")
    docs = _flatten(results, "documents")
    metas = _flatten(results, "metadatas")
    dists = _flatten(results, "distances")

    raw_hits: List[Dict[str, Any]] = []
    n = min(len(ids), len(docs), len(metas), len(dists))
    for i in range(n):
        md = metas[i] or {}
        doc = docs[i] or ""
        dist = float(dists[i])
        sim = _clamp01(1.0 - dist)
        fp = md.get("file_path")
        file_abs = str(_posix_to_abs(fp)) if fp else None
        raw_hits.append({
            "chunk_id": ids[i],
            "similarity": sim,
            "file_path": fp,
            "file_abs": file_abs,
            "start_line": md.get("start_line"),
            "end_line": md.get("end_line"),
            "parent_context": md.get("parent_context"),
            "symbol_name": md.get("symbol_name"),
            "symbol_type": md.get("symbol_type"),
            "is_async": md.get("is_async"),
            "file_version_hash": md.get("content_hash"),
            "preview": _make_preview(doc, MAX_PREVIEW_CHARS),
        })

    raw_hits.sort(key=lambda h: h.get("similarity", 0.0), reverse=True)
    hits = _diversify_by_file(raw_hits, limit=limit, max_per_file=MAX_PER_FILE)
    for idx, h in enumerate(hits, start=1):
        h["rank"] = idx

    best_sim = hits[0]["similarity"] if hits else None
    quality = _quality(best_sim, len(hits))
    total_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "ok": True,
        "schema": "tinnitus_search.search_code.v1",
        "query": query,
        "limit": limit,
        "index": {"freshness_mode": freshness, "index_checked": index_checked, "collection": COLLECTION},
        "timing_ms": {"ensure_fresh_index": t_idx_ms, "chroma_query": t_q_ms, "total": total_ms},
        "quality": quality,
        "hits": hits,
        "recommended_next": [{"tool": "get_chunk_by_id", "args": {"chunk_id": h["chunk_id"]}} for h in hits[:2]],
        "warnings": [],
    }


@mcp.tool()
def get_chunk_by_id(chunk_id: str) -> Dict[str, Any]:
    store, _ = get_state()
    res = store.get_by_ids([chunk_id])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    doc = docs[0] if docs else None
    md = metas[0] if metas else {}
    fp = md.get("file_path") if isinstance(md, dict) else None
    file_abs = str(_posix_to_abs(fp)) if fp else None
    return {"ok": True, "chunk_id": chunk_id, "found": bool(ids), "text": doc, "metadata": {**(md or {}), "file_abs": file_abs}}


@mcp.tool()
def get_code_fragment(file_path: str, start_line: int, end_line: int, max_chars: int = 8000) -> Dict[str, Any]:
    _require_dir(REPO_ROOT, "REPO_ROOT")
    abs_path = _posix_to_abs(file_path)
    lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
    frag = "\n".join(lines[max(0, start_line - 1): min(len(lines), end_line)])
    if len(frag) > max_chars:
        frag = frag[:max_chars] + "\n...<truncated>..."
    return {"ok": True, "file_path": file_path, "file_abs": str(abs_path), "start_line": start_line, "end_line": end_line, "code": frag}


@mcp.tool()
def update_index() -> Dict[str, Any]:
    _, indexer = get_state()
    t0 = time.perf_counter()
    indexer.ensure_fresh_index()
    ms = int((time.perf_counter() - t0) * 1000)
    return {"ok": True, "timing_ms": {"ensure_fresh_index": ms}}


@mcp.tool()
def index_status() -> Dict[str, Any]:
    return {"ok": True, "repo_root": str(REPO_ROOT), "chroma_dir": str(CHROMA_DIR), "collection": COLLECTION, "model_dir": MODEL_DIR}


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Запускаем фоновый прогрев ПОСЛЕ того, как mcp.run() начнёт
    # слушать STDIO (через daemon-поток).
    # FastMCP.run() блокирующий, поэтому стартуем warmup ДО него,
    # но в daemon-потоке — он не помешает handshake.
    warmup_thread = Thread(target=_background_warmup, daemon=True)
    warmup_thread.start()
    log.info("MCP server starting (STDIO), warmup in background...")

    mcp.run(transport="stdio")