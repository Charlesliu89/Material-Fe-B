# CALPHAD 目录结构 / Layout

- 核心 / Core：`calphad_core.py`（pycalphad 封装）。
- 示例 / Examples：`run_demo.py`（含 `equilibrium` / `fe-cr-gm` / `fe-b-t0` 子命令）。
- 流水线 / Pipelines：`pipeline/tdb_builder.py`（单次构建/校验）、`pipeline/tdb_pipeline.py`（迭代检索→下载→抽取→合并→烟测；默认状态保存在 `tmp/tdb_pipeline_state.json`）、`pipeline/tdb_local_pipeline.py`（本地已有 PDF/text 时跳过在线检索/下载，直接抽取→合并→烟测）。
- 工具 / Tools：`tools/` 下的文献检索/下载/抽取/合并/诊断脚本（如 `lit_search.py`、`search_only.py`、`topic_search_agent.py`、`fetch_sources.py`、`extract_tdb_lines.py`、`build_tdb.py`、`tdb_smoke_test.py`、`net_diag.py`）。`search_only.py`/`topic_search_agent.py` 自动尝试 certifi 证书，可用 `--ca-bundle` 或 `--insecure` 诊断网络。  
  文献向量库：`vector_store_builder.py` 会递归上传 `literature_library/` 下的 PDF 并创建/复用 OpenAI 向量库（为 Codex/Assistants 提供 file_search）。
- 数据 / Data：`tdb/`（Fe-B 占位/模板），`thermo/Database/`（实际 TDB，如 `COST507.tdb`、`crfeni_mie.tdb`），`thermo/Handbook/`（参考 PDF），`plots/`（生成图）。
- 来源与临时 / Sources & temp：`sources/`（文献下载与阻断诊断），`tmp/`（临时状态文件）。
- 同步提示 / Sync note：`**/__pycache__/.gitkeep` 仅为占位，`.pyc` 缓存不提交；`tdb/`、`thermo/Database/` 等数据目录也使用 `.gitkeep` 占位但不含实际 TDB/手册文件。
