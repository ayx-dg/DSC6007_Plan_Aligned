# DSC6007 Group 14 — Hybrid LinUCB 对齐复现

本仓库是 Li et al. (2010) *A Contextual-Bandit Approach to Personalized News
Article Recommendation* 的独立复现与课程报告。它同时服务三类读者：

| 读者 | 你应该先看 |
| --- | --- |
| 本地 / 接手的组员 | [仓库结构与约定](#仓库结构与约定) |
| Codex, claude code etc. | [AGENTS.md](AGENTS.md) |
| 评分老师 / 助教 | [从全新机器复现](#从全新机器复现)、[报告与仓库的对应](#报告与仓库的对应) |
| 只想看结果的人 | 直接打开 `report.pdf` 和 `slides.pdf`，不需要运行任何命令 |

**仓库根目录就是本项目目录**：`git clone` 之后直接进入
`DSC6007_Plan_Aligned/`，里面就是 `run.sh`、`report.tex` 等文件，不存在再嵌一层的
`hybrid_aligned/`。

---

## 仓库结构与约定

```
DSC6007_Plan_Aligned/          # clone 后的根目录 = 本目录
├── AGENTS.md                  # agent / 接手者必读的项目约定与复现契约
├── run.sh                     # 唯一实验入口：测试 → 核心实验 → 扩展 → 稀疏诊断 → 图 → 自查
├── build_report.sh            # 编译 report.pdf 与 slides.pdf
├── verify_consistency.py      # 数字、哈希、页数、评审新鲜度自查
├── pyproject.toml / uv.lock / .python-version
│                              # Python 3.13.2 与精确依赖版本；禁止 uv lock --upgrade
├── project_desc.pdf           # 课程要求原文（SHA-256 见 AGENTS.md）
├── report.tex / report.pdf    # 课程报告（10 页，身份信息固定，不可改名）
├── slides.tex / slides_numbers.tex / slides.pdf
│                              # 15 页答辩幻灯片；numbers 由结果 JSON 自动生成
├── reproduction/              # Yahoo R6A 原始日志 + 核心策略与实验代码
│   ├── experiments.py         # LinUCBDisjoint / LinUCBHybrid / 两个 greedy baseline
│   ├── test_experiments.py    # 14 项正确性检查
│   └── Yahoo_recommendation/data/   # 原始日志与文章特征（约 21 MB，已随仓库提供）
├── extend_and_audit.py        # 20 个独立种子、窗口与截距消融
├── sparsity/run_sparsity.py   # 固定测试集的稀疏 warmup 诊断
├── make_figures.py / make_slide_numbers.py / make_submission.py
├── core/ robustness/ sparsity/ figures/   # 机器可读结果与图
├── references/                # Li et al. 2010 / 2011 原文
└── submission/                # 按课程命名打包的提交件
    ├── Group 14 report.pdf
    ├── Group 14 slides.pdf
    ├── Group 14 c *.txt
    └── Group 14.zip
```

---

## 从全新机器复现

这一节写给没有见过我们终端历史的人：按下面顺序执行即可得到本报告的全部数字和 PDF。

### 1. 前置依赖

需要的只是 `git`、`uv` 和（可选）`pdflatex`：

```bash
# 安装 uv（若已安装可跳过）；官方脚本会把可执行文件放到 ~/.local/bin/uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 若 uv 不在 ~/.local/bin，告诉 run.sh 它在哪里
export UV="$(command -v uv)"
```

`pyproject.toml` 固定 `requires-python == 3.13.2`。首次 `uv run --frozen` 会让 uv
下载该解释器和锁定版本的 wheel（**这一步需要联网**）；此后的所有实验阶段都使用
`uv.lock` 且不再联网。原始 Yahoo 日志已经包含在仓库中，不需要另外下载。

若只查看报告，不需要 LaTeX；若要重新生成 PDF，需要任意包含 `pdflatex` 的 TeX
发行版，并具备 `beamer`、`amsmath`、`booktabs`、`geometry`、`microtype`、
`hyperref`、`xurl`、`lmodern`、`iftex` 等宏包。Debian/Ubuntu 可用：

```bash
sudo apt-get install -y texlive-latex-base texlive-latex-extra \
                        texlive-fonts-recommended texlive-pictures
```

### 2. 拉取代码

```bash
git clone https://github.com/ayx-dg/DSC6007_Plan_Aligned.git
cd DSC6007_Plan_Aligned
```

### 3. 运行实验

```bash
./run.sh
```

它会依次执行：14 项正确性检查 → Yahoo 核心实验 → 20 种子扩展与审计 → 稀疏数据诊断
→ 图与幻灯片数值 → 一致性自查。CPU 单线程 BLAS，大约需要 10–15 分钟；不需要 GPU
或任何 API key。

> **非 x86-64 机器（例如 Apple Silicon）**：`run.sh` 里固定了
> `OPENBLAS_CORETYPE=Haswell` 以锁定数值路径。若出现 BLAS 报错或非法指令，请先把
> 这一行注释掉再运行；末位浮点差异不影响报告结论。



### 4. 自查与产物

```bash
$UV run --frozen python verify_consistency.py
```

期望看到：

```
CONSISTENCY CHECK PASSED correctness_tests=14 report_pages=10
```

在全新 clone 中还会额外出现
`review_artifacts_local_only: public clone skips review checks`，这是预期行为。

关键产物与校验点：

| 产物 | 内容 / 期望值 |
| --- | --- |
| `core/results.json` | 七策略测试表、30 种子均值/标准差、验证选择的参数 |
| `robustness/results.json` | 20 种子、窗口敏感性、截距消融与配对区间 |
| `sparsity/results.json` | 4 档 warmup × 2 候选池 × 4 策略，共 32 条 |
| `report.pdf` | 10 页；数据哈希与源码哈希见 Appendix A |
| `slides.pdf` | 15 页；数值来自 `slides_numbers.tex` |
| `submission/Group 14.zip` | 按课程命名打包的报告、幻灯片和代码 |

时间切分固定为 66,204 / 18,010 / 15,786 个事件，核心随机种子 2026，模拟为
3,000 轮 × 6 个候选、原始 10 种子 (0–9) 与独立 20 种子 (100–119)。这些数字写在
`AGENTS.md` 的复现契约里，也是自查脚本检查的内容。

---
## For Codex

If you are an agent, see [AGENTS.md](AGENTS.md) for contributing rules.

## 报告与仓库的对应

- `report.tex` 是报告源，`report.pdf` 是提交稿；`slides.tex` 生成答辩幻灯片。
- 报告中的每一个数字都能在 `core/results.json`、`robustness/results.json`、
  `sparsity/results.json` 中找到；哈希和精确版本在报告 Appendix A 与相应 manifest。
- `submission/` 中的文件按课程要求命名为 `Group 14 ...`，`Group 14.zip` 是最终
  提交包；它由 `make_submission.py` 从当前 PDF/源码重新生成。
- paper-judge 的 C1–C5 草稿只在本地保留，不随公开仓库分发；草稿建议由 owner
  判断后再决定是否纳入报告修订。

如果只想核对结论，直接打开 `report.pdf`：Yahoo 回放没有确立稳定赢家，
稀疏 warmup 诊断未复现论文的 hybrid 优势，受控漂移中窗口在指定变化下降低遗憾，
但这些都不等价于线上收益。
