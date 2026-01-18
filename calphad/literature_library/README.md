# Literature Library

Place PDFs related to CALPHAD references and supporting documents here. Create subfolders (e.g. `systems/`, `methods/`, `reviews/`) if you want to keep items organized.

## 构建 OpenAI 向量库
- 安装依赖：`pip install -r requirements.txt`
- 运行：`python calphad/tools/vector_store_builder.py`（默认读取本目录下的全部 PDF 并创建/上传向量库）
  - 已有向量库时用 `--vector-store-id <id>` 复用同一个库
  - 仅查看将上传的文件用 `--dry-run`
- 说明：脚本会全量上传目录下的所有 PDF，API 不会按文件名自动去重；重复运行会产生多份同名文件。如需避免重复，记住向量库 ID，删除多余 file_id 后再上传新增文件。
- 使用：在 Codex/Assistants 对话里绑定你的向量库 ID（Storage → Vector stores），或在创建 assistant 时把 `vector_store_ids` 传给 `file_search` 工具。
