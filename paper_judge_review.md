# Paper Judge 评分草稿（Content 12 分）

- 提交文件：`report.pdf`（10 页，pdflatex 编译）
- 审核时间：2026-11-22 12:01
- 模型：`deepseek-chat`，prompt 2026.2，rubric 2026.2
- 已审核 PDF SHA-256：`a3518f6fe7a975178b77064812d1c31c626621fc8012dafac501a18a040bd0a3`
- **草稿总分：8.85 / 12**
- 原始证据/逐条反馈见 `review_paper_judge.json`

总评（原文）：报告完整实现了 Li et al. (2010) 的 Disjoint 与 Hybrid LinUCB，并给出 Yahoo 时序回放、受控漂移仿真与截距消融，方法推导（Schur 补、联合岭解、方差式）与十四项正确性检查扎实，负结果与不确定性说明诚实。主要不足是 Yahoo 匹配样本极小（22–30 次点击）、回放条件未认证，结论只能停留在条件性表述；仿真部分缺少随机种子与超参的集中复现说明。整体属于方法忠实、结论克制的高质量课程报告。

## 红旗与提示
- RF4 缺少复现说明：正文未集中给出仿真随机种子（仅零散提及 2026、0–9、100–119）与 LinUCB 宽度等超参的完整清单，复现信息分散在附录与脚本中
- 扣分：C3 因「缺少复现说明（随机种子/独立重复/代码或超参）」由「full」封顶至「partial」（1.05 分）。
- 附录标题「supplementary analysis; all numbers come from included programs. Assistance does not turn replay into an」出现在第 9 页且与正文同页；已按严格口径将该页计入正文，请人工确认。

## C1 背景、动机与核心思想 — 2/2 full（置信度 low）

- 理由：报告在 1.1 明确给出研究问题（Q1–Q3）、动机与重要性（未观测奖励、偏好漂移下的跟踪与精度权衡），1.2 明确说明与课程的关联并列出贡献，结论段可用一句话概括核心思想（遗忘在受控漂移世界中有效、但离线回放与模拟不足以证明在线收益），且选题属于课程范围（contextual bandit / 非平稳序列决策）。未发现虚构引用或过度宣称（作者反复声明不宣称在线收益、不虚构 propensities），C5 不触发降档；C3 相关（seed、误差棒）在正文 3.2/3.4 有报告，不构成 C1 降档理由。故评 full。
- 优点：研究问题明确：在部分点击反馈与偏好漂移下如何做新闻推荐，并拆成 Q1–Q3 三个可检验子问题。
- 优点：动机与重要性（Why / So What）交代清楚：未展示项奖励不可观测、全历史反馈适应慢、丢弃旧数据牺牲估计精度，构成 exploration–exploitation 与 bias–precision 的张力。
- 优点：核心思想可一句话概括：用忠实实现的 Disjoint/Hybrid LinUCB 加窗口机制，在 Yahoo 回放与已知均值的漂移模拟中检验遗忘是否有助于跟踪。
- 优点：明确说明与课程的关联（contextual bandits、optimistic exploration、ridge estimation、非平稳序列决策），契合课程选题范围。
- 优点：贡献与边界陈述克制，明确保留负结果、不宣称新定理或在线收益。
- [minor] 1.2 Course connection, related work and contributions：课程关联以一句列举式表述给出，未进一步说明本项目相对课程讲授内容（如 Bellman/SSP/平均代价 MDP 等）的具体落点。 建议：可在 1.2 补一两句，说明该问题在课程框架中的位置（如 bandit 作为无状态 MDP 的特例、非平稳性与 Bellman 更新的关系）。
- [suggestion] 1.1 Motivation and research questions：动机以新闻推荐场景切入，但未给出该场景为何在方法上困难的具体量化背景（如候选规模、反馈稀疏程度）。 建议：可在 1.1 用一两句点出候选集规模与点击稀疏性（正文 3.1 已有 20/21 候选、2.807% CTR 等数据），前置支撑动机。

## C2 理论与/或应用 — 1.8/3 substantial（置信度 low）

- 理由：报告给出了正确的 MDP/contextual bandit 形式化、Disjoint 与 Hybrid LinUCB 的 ridge 解与联合 Schur 补推导、等价的预测方差表达式，以及 Eq. (8) 的代数恒等式，关键假设（有界线性均值、sub-Gaussian、α 启发式、replay 条件）显式列出且合理，应用实现（Yahoo replay、drift 模拟、ablation）逻辑自洽并保留负结果，引用为可核实的 2010–2021 顶会/期刊出处。存在少量不影响主结论的瑕疵：Hybrid 特征预处理偏离原文、窗口规则缺理论界、模拟器 misspecification。按 C3 规则，正文报告了 seed 2026 与 2026–2055、0–9、100–119 及 mean±SD，满足复现要求；C5 未见明显过度宣称或虚构引用。综合落在 substantial 锚点。
- 优点：Disjoint 与 Hybrid LinUCB 的 ridge 估计、Schur 补与联合最小二乘解（Eq. 2/5/6）表述正确，并给出等价的预测方差形式（Eq. 7），与 Li et al. 2010 Algorithm 2 一致。
- 优点：显式列出关键假设（有界线性均值、sub-Gaussian 噪声、α 为启发式而非校准覆盖），并说明 replay 仅在均匀随机 logging 与候选集条件下有效，未虚构 propensities。
- 优点：Eq. (8) 的偏差-精度恒等式推导正确且注明在 misspecification 下仍成立，并明确其为条件性动机而非定理。
- 优点：应用层面实质：chronological Yahoo replay、sparse warmup 诊断、intercept ablation 与 window 敏感性构成自洽的验证链，且保留负结果。
- 优点：引用可核：Li et al. 2010/2011、Cheung 2019、Russac 2019、Zhao 2020、Faury 2021、Dudik 2011 均为可查顶会/期刊出处。
- [major] 2.3 Hybrid LinUCB 与 4.1/4.2 结果：Hybrid 的 36 维共享特征 z 由 R6A 提供的六维向量直接构造，作者承认偏离原文的 unit-normalise 预处理；同时 Eq. (4) 的分解假设未做检验，Hybrid 与 Disjoint 的对比因此混杂了特征构造差异。 建议：在正文明确标注该偏差对 Hybrid 估计的潜在影响，或补充一个使用原文归一化近似的敏感性对照。
- [minor] 2.4 Window adaptation：窗口规则仅给出条件性动机，未给出任何 tracking 误差或 regret 的界，Eq. (8) 的 stale-data 项未加 variation/support 假设界定。 建议：在 Discussion/Theory 中明确该恒等式不构成保证，并指出所需 variation budget 假设（已在 Conclusion 提及，可前移强化）。
- [minor] 3.3 Drift study, Eq. (9)：模拟器 μt(a) 的 affine 映射在六维无截距下 misspecified，作者以 intercept7 消融补救，但未给出该 misspecification 对 regret 排序影响的理论分析。 建议：在正文说明 intercept7 结果如何界定该 misspecification 的影响范围，避免过度外推。

## C3 数值实验 — 1.05/3 partial（置信度 low）

- 理由：报告包含相关且有竞争力的 baseline（Random、context-free ϵ-greedy、contextual greedy、Disjoint/Hybrid LinUCB、Window LinUCB、LR），关键组件有消融（intercept7 vs original6、window 大小、sparse warmup、50-article pool）；多次独立运行并报告随机种子（2026–2055、0–9、100–119）、均值与 sample SD/置信区间；指标与论断对齐谨慎（matched replay CTR 描述性、不宣称在线收益）；超参、环境与实现细节（锁定版本、hash、run.sh）足以复现；图表有单位、图例与误差棒/区间。仅存在单次实现（Yahoo 主表、sparse 子采样、OPE 单 realisation）与未校正探索性比较等 minor/suggestion 级问题，未触发“未报告随机种子/独立重复或缺少误差棒”的降档规则，故评 full。
[扣分封顶] 因缺少复现说明（随机种子/独立重复/代码或超参），由 full 封顶至 partial。
- 优点：baseline 覆盖充分：Random、context-free ϵ-greedy、contextual greedy、Disjoint/Hybrid LinUCB、Window LinUCB 与 Logistic regression 预测基线并列比较，且明确区分预测基线与因果比较。
- 优点：独立重复与误差报告规范：Yahoo 随机策略用 seeds 2026–2055，模拟用 seeds 0–9 与 100–119，表格给出 mean ± sample SD，并说明 t 区间为 pointwise、Bonferroni 校正六个 contrast。
- 优点：消融与敏感性分析到位：intercept7 vs original6 特征消融、window 100/300/600 敏感性、sparse warmup 100/10/5/1% 诊断、50-article pool 敏感性检查。
- 优点：指标-论断对齐谨慎：明确 matched replay CTR 为描述性、非无偏部署值，不据 30 clicks 宣称排序或在线收益，负结果保留。
- 优点：可复现性细节充分：锁定 Python 3.13.2 与依赖版本、uv.lock、线程环境变量、数据与源码 SHA-256、run.sh 与 verify_consistency.py。
- 优点：图表规范：图注说明误差棒/区间含义、单位（%）、图例与样本量，并标注区间非 simultaneous。
- 优点：实现正确性检查：十四项 correctness checks 覆盖 inverse-update、expiry、joint Hybrid solve、Algorithm 2 variance 等。
- [minor] 4.1 Yahoo held-out replay / Table 2：Yahoo 主表仅报告 seed 2026 单次实现，确定性 UCB 行无重复；30-seed 仅覆盖三个随机策略。 建议：在表注中明确确定性策略不适用 seed 重复的理由，并补充匹配集 bootstrap 或选择不确定性的描述性区间。
- [minor] 4.2 Sparse-data diagnostic / Table 3：sparse warmup 诊断仅用 seed 2026 单次子采样，未报告多次子采样的均值与方差。 建议：对每个 warmup 比例做多次独立子采样并报告均值±SD，或说明为何单次已足够。
- [suggestion] 4.3 Supervised prediction and the OPE control：OPE 检查仅一个 realisation（20,000 contexts、固定 target、uniform logger），未报告 IPS/SNIPS/DR 的方差或重复。 建议：增加多次独立 realisation 的均值与方差，或明确标注为单次 sanity check 并限制结论。
- [suggestion] 4.5 / Figure 5：window 100/300/600 与 contextual-greedy 控制为描述性、未做多重比较校正，作者已声明。 建议：在图注或正文再次标注这些比较为探索性、未校正，并避免据此给出窗口选择建议。

## C4 结论、局限、优势与展望 — 2/2 full（置信度 low）

- 理由：报告设有独立的 Strengths、Limits 与 Conclusion 段，结论由 Yahoo replay 与 20-world 模拟证据支撑并明确限定为条件性；扩展分别覆盖理论、算法、应用三层面且具体可执行，符合 full 锚点。C3 方面报告给出 seed 2026、2026–2055、0–9、100–119 及 mean±SD 与 t 区间，未触发降级；C5 方面未见过度宣称或无法核实的引用，故评 full。
- 优点：结论明确标注为条件性（conditional），并区分离线 replay CTR 与模拟 regret 的证据边界，未宣称 online gain。
- 优点：第 5 节单列 Strengths 与 Limits，局限具体（80 分钟日志、匹配样本小、LR 冻结与 bandit 自适应不可比、模拟器缺位置效应/延迟点击、六特征 map 误设、20 worlds 不足以确立通用记忆规则）。
- 优点：扩展按 Theory / Algorithm / Application 三层面分别给出，且具体可执行（界定 Eq.(8) stale-data 项、比较 window 与 discounted/restarted、获取带 propensity 的长日志）。
- 优点：对不利结果（Yahoo window 未胜出、sparse 诊断未复现 hybrid 优势、enlarged pool 退化）予以保留并解释，未虚构反事实标签或 propensity。
- [minor] 5 Discussion / 6 Conclusion：局限与扩展虽具体，但未明确给出各扩展的优先级或可检验的成功判据（如理论界需满足的 variation budget 形式）。 建议：为 Theory/Algorithm/Application 各扩展补一句预期产出或判据（如界的形式、对比的指标与 seed 规模）。
- [suggestion] 6 Conclusion：结论段对 hybrid 与 window 的负面结果陈述清楚，但未汇总各 claim 的证据强度分级。 建议：可用一小表列出 claim 与对应证据（replay / simulation / 未确立）。

## C5 论证严谨性与证据可追溯 — 2/2 full（置信度 low）

- 理由：报告对每个核心论断都给出对应的实验、公式或引用支撑，且通篇主动限定结论范围、显式声明假设与适用边界，负结果保留并与正结果做了一致性解释；参考文献均为真实可核实的公开文献，正文引用与文献表对应；AI 使用被显式披露。未发现过度宣称、明显 AI 生成痕迹或无法核实的引用，符合 full 锚点。仅存在 AI 披露句在正文被截断、个别推导步骤未展开两处轻微问题，不构成降档理由。
- 优点：claim 与证据逐条对应：每个核心论断（Hybrid 无 CTR 增益、稀疏优势未复现、window 在漂移下有效但 stationary 增益不稳健、OPE 仅作 sanity check）都指向具体表格、公式或实验设置。
- 优点：过度宣称控制极好：反复显式声明边界（matched CTR 为描述性、Wilson 区间为名义、seed 离散度非用户总体不确定性、仿真非在线增益、enlarged pool 非有效 replay）。
- 优点：假设与适用边界显式：LinUCB 置信读数的有界/次高斯假设、Eq.(8) 恒等式在 misspecification 下仍成立、window 单位差异（log event vs round）、intercept 缺失导致的 misspecification 均写明。
- 优点：叙事连贯：问题→方法→证据→结论链条清晰，负结果保留且与正结果的关系被解释（不同环境/评估单元不冲突）。
- 优点：引用可核实：参考文献均为真实可查的公开论文（Li et al. 2010/2011、Cheung、Russac、Zhao、Faury、Dudik），带 arXiv/PMLR 链接，正文引用编号与文献表一致。
- 优点：AI 使用被显式披露且限定范围（drafting、derivation review、supplementary analysis），并声明不改变证据性质。
- [minor] Reproducibility and AI assistance 段末：AI 协助说明句在正文末尾被截断（"AI tools assisted drafting, derivation review and"），完整表述落在附录。 建议：在正文中补全该句，明确 AI 协助的具体环节与不涉及的部分。
- [suggestion] 2.3 Hybrid LinUCB 与 2.4：Eq.(7) 的等价形式与 Eq.(8) 的推导仅给出结论式，未展示中间步骤。 建议：可在附录补一行关键中间步骤或指向测试文件中的数值校验。

## 说明
以上为供教师复核的**草稿建议**，不是最终成绩；本次未根据草稿自动修改报告。
