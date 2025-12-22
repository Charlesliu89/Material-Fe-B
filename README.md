# Fe-B &Delta;H<sub>mix</sub> Toolkit / Fe-B &Delta;H<sub>mix</sub> 工具集

A set of Python utilities for evaluating and visualising alloy enthalpy of mixing from the Omega interaction matrices stored in `Data/Element pair data base matrices.xlsx`.
本工具集依托 `Data/Element pair data base matrices.xlsx` 中的 Omega 互作用矩阵，提供一组用于评估和可视化合金混合焓 (&Delta;H<sub>mix</sub>) 的 Python 脚本。

It includes:
功能涵盖：

- An interactive single-composition calculator that reports pairwise contributions.
  交互式单点计算器，可输出成分对的贡献明细。
- A batch plotter that exports binary curves, ternary contour plots, quaternary previews/slices, and equimolar quinary rankings.
  批处理绘图器，可批量生成二元曲线、三元等高线、四元四面体预览/切片以及等摩尔五元组合排名。
- A dedicated Fe-B-Cr plot that overlays reference phase boundaries on top of the computed &Delta;H<sub>mix</sub> field.
  专门的 Fe-B-Cr 绘图脚本，可在计算出的 &Delta;H<sub>mix</sub> 场上叠加已知相界。

The code base is modular: `enthalpy_core.py` handles data loading and math, `enthalpy_plot.py` builds Plotly figures, while high-level scripts orchestrate workflows.
代码结构模块化：`enthalpy_core.py` 负责数据加载和数学计算，`enthalpy_plot.py` 负责 Plotly 图形构建，高层脚本则组织具体流程。

## Installation / 安装

1. Use Python 3.10+.
   使用 Python 3.10 及以上版本。
2. (Optional) Create a virtual environment.
   （可选）创建虚拟环境。
3. Install dependencies:
   安装依赖：

```bash
python -m pip install -r requirements.txt
```

> PNG export relies on [Kaleido](https://github.com/plotly/Kaleido). If it fails, scripts fall back to HTML.
> PNG 导出依赖 [Kaleido](https://github.com/plotly/Kaleido)（已在依赖中列出）；若导出失败，将自动回退为交互式 HTML。

## Data preparation / 数据准备

All scripts expect the workbook at `Data/Element pair data base matrices.xlsx`.
全部脚本默认使用 `Data/Element pair data base matrices.xlsx` 作为数据源。

Ensure sheets `U0`, `U1`, `U2`, `U3` contain pair interaction coefficients ordered by atomic number; update the Excel file whenever new data arrive.
请确认工作簿中的 `U0`、`U1`、`U2`、`U3` 工作表已按原子序排列并填充配对互作用系数；若公式或元素范围更新，请同步 Excel。

## Scripts and usage / 脚本与用法

### 1. `enthalpy of mixing.py`

Interactive or single-shot calculator for 2-5 element compositions.
支持 2-5 元素成分的交互式/一次性计算器。

```bash
python "enthalpy of mixing.py" [--excel PATH] [--line "Fe 20 Al 80"] [--list-elements]
```

Key options / 主要选项：

- `--excel PATH` – override the default Omega workbook.
  指定自定义的 Omega Excel 源。
- `--line` – compute a single composition non-interactively (percentages auto-normalised).
  提供单行成分，脚本直接运算并归一化百分比。
- `--list-elements` – print supported symbols.
  输出当前 Excel 中可用的元素列表。

Without `--line`, the script opens an interactive REPL; enter `Fe 20 Al 30 Ni 50` or `Fe=20,Al=30,Ni=50` to view normalised fractions, pair contributions, and total &Delta;H<sub>mix</sub>.
未指定 `--line` 时进入交互模式，可输入 `Fe 20 Al 30 Ni 50` 或 `Fe=20,Al=30,Ni=50`，查看归一化摩尔分数、对成对贡献以及总 &Delta;H<sub>mix</sub>。

### 2. `batch_enthalpy.py`

Generates large batches of plots. Run without arguments to open the menu.
批量绘图脚本；直接运行可打开菜单。

```
python batch_enthalpy.py
```

Menu entries / 菜单选项：

1. Batch binary curves → `Data/plots/binary`.
   二元曲线批量导出。
2. Batch ternary contour plots → `Data/plots/ternary`.
   三元等高线批量导出。
3. Quaternary tetrahedron preview + ternary slices → `Data/plots/quaternary`.
   四元四面体预览与切片。
4. Custom plots for any 2-4 elements → `Data/plots/custom`.
   自定义 2-4 元素组合。
5. Equimolar 5-component rankings + optional Excel export → `Data/plots/quinary`.
   等摩尔五元系统排名，并可导出 Excel。

Useful CLI switches / 常用命令行参数：

- `--calculator PATH` – swap in another calculator module (defaults to `enthalpy of mixing.py`).
  指定不同的计算核心脚本。
- `--excel-db PATH` – force a specific Omega workbook.
  强制使用指定 Excel。
- `--elements Fe B Ni ...` – restrict element pool.
  限制组合所用元素范围。
- `--output-dir Data/plots` – change output root (subfolders auto-created).
  改变输出根目录，子目录会自动创建。
- `--auto-combo Fe,B,Ni --auto-combo Fe,Co` – render listed combos (2-4 elements each) straight to `custom/`.
  直接渲染指定组合，绕过菜单，结果存入 `custom/`。
- `--workers 8` – number of worker processes (default: CPU count).
  并行进程数。
- `--chunk-size 50` + `--chunk-auto-continue` – control interactive batching / run unattended.
  设置每批组合数量及是否自动继续，方便长任务或 CI。
- `--list-elements` – show supported elements and exit.
  查看元素集合后退出。

If Kaleido cannot create PNG files (e.g., headless servers), `.html` fallbacks are written.
若 Kaleido 无法导出 PNG（例如在无头服务器上），脚本会自动保存 `.html` 版交互图。

### 3. `Fe-B-Cr deltHmix covered with phase boundaries.py`

Renders a high-resolution Fe-B-Cr ternary plot with optional phase-boundary overlays snapped to the sampling grid.
生成高分辨率 Fe-B-Cr 三元图，并可将相界曲线对齐采样网格后叠加。

```bash
python "Fe-B-Cr deltHmix covered with phase boundaries.py" \
    [--excel Data/Element pair data base matrices.xlsx] \
    [--step 0.001] [--boundary-csv Data/FeBCr phase diagram line.csv] \
    [--output Data/FeBCr_phase_boundaries.png] [--show] [--no-boundaries]
```

- `--step` – sampling resolution (fraction value, default 0.001 = 0.1%).
  设置采样精度（默认 0.001，即 0.1%）。
- `--boundary-csv` – CSV with boundary points (a/b/c columns) that are cleaned and snapped to the grid.
  指定包含 a/b/c 列的相界 CSV，程序会清洗并对齐网格。
- `--output` – PNG destination (folders auto-created). Add `--show` for an interactive preview.
  输出 PNG 路径，可配合 `--show` 打开交互预览。
- `--no-boundaries` – disable overlay altogether.
  若无需叠加相界可使用此参数。

## Tips / 使用提示

- Always verify that the script lists the same elements as the Excel workbook; missing entries usually mean header/row formatting issues.
  首次运行时确认终端中列出的元素与 Excel 完全一致，缺项往往意味着 Excel 行列格式需要清理。
- Narrow the element pool via `--elements` during batching to avoid unsupported combinations; skipped systems will be reported.
  批处理时通过 `--elements` 限制元素范围，可减少由于缺少数据导致的跳过。
- Quaternary previews support slices such as `Fe=25`, exporting ternary PNG/HTML files inside `Data/plots/quaternary`.
  四元预览可固定某元素百分比（例如 `Fe=25`）并输出三元切片图。
- Equimolar quinary rankings split combinations by the sign of &Delta;H<sub>mix</sub>, and Excel export helps downstream analysis.
  等摩尔五元排名会按 &Delta;H<sub>mix</sub> 正负分组，并可导出 Excel 便于后续处理。

---

When extending the workflow (new calculators, colour maps, etc.), keep shared constants in `enthalpy_config.py` so every script stays consistent.
若需扩展新脚本或风格，请将通用常量放在 `enthalpy_config.py`，以便各模块共享统一配置。
