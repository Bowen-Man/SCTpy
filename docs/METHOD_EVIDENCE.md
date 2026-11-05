# Method evidence review / 方法证据待办

审查日期：2026-11-05。基线：`9dd88227ffeed74de26e7607aaa6118bad526dbd`；位置指向本 PR 中的代码。
本表用于逐块补证，不表示整套算法没有依据，也不表示已经证明代码复制。
“待补”指当前仓库尚未给出足够精确、可核对的证据映射，不等于文献中不存在。
本轮只读取本项目代码、论文正文和公开 API 文档，没有读取外部 R 实现源码。

## 阅读方法

- **统计依据**：需要公式、目标函数、假设或方法学论证；R 输出相近不能替代。
- **数值依据**：需要误差界、极限、边界和高精度实验；不必强找单细胞论文。
- **兼容/工程契约**：需要明确选择及测试；不能直接标成论文要求。
- **部分已有**：一般原理已找到依据，但下面列出的具体组合或常数仍待核对。

## 优先审查的统计代码

| ID | 代码位置与关键实现 | 已有依据及其边界 | 要补充的证据 / 判断 |
|---|---|---|---|
| M1 | [`_fit._cox_reid_score_batch`](../src/sctpy/_fit.py#L254)、[`_gpu._cox_reid_score_cupy`](../src/sctpy/_gpu.py#L34)：`0.5 * phi * 0.99 * sum(w*w) / (sum(w) + 1e-6)` | 仓库引用 [glmGamPoi 2020][G]；高精度测试验证的是这个给定表达式的数值计算。**尚未闭合统计目标。** | 写出包含 `0.99` 与该位置 `1e-6` 的调整目标，推导导数及尺度/单位；说明它们是理论结果、稳定化还是经验修正。不能把其他位置的同值常数当成依据。 |
| M2 | [`_fit.fit_cox_reid_offset_batch`](../src/sctpy/_fit.py#L292)、[`fit_cox_reid_offset`](../src/sctpy/_fit.py#L415)、[`core.fit_transform`](../src/sctpy/core.py#L378)：`(variance - mean / mean(exposure)) / mean**2` 初始化；固定初始 `mu` 求 dispersion，再用趋势 dispersion 重拟合 intercept，但保留 raw theta | [P22] 支持 NB offset 框架；[G] 是拟合方法的进一步核对入口。本轮未重新取得其补充材料，不能将历史对话概述当作已核对的逐式证据。 | 对照补充材料的具体公式/页码，逐阶段说明优化哪个量、固定哪个量；解释最终混合参数对的含义。修正 objective 只使诊断自洽，不证明这套流程是联合 MLE。 |
| M3 | [`_fit.local_median_dispersion_trend`](../src/sctpy/_fit.py#L517)：有序 500-gene 分组；`max(100, 0.1*size)` 邻域；`exp(-0.5*linspace(-3,3,...)**2)` 权重；边缘填充 | [V] 确认 `bin_size=500` 的公开参数及计算分块描述，**未据此证明分组应改变统计估计**。 | 补 local median 的定义、邻域/权重/端点依据，解释基因顺序和 500 分组的统计作用。与仅限制工作集的 64-gene 批量求根分开评估。 |
| M4 | [`_regularize._dispersion_target`](../src/sctpy/_regularize.py#L365)、[`_theta_from_dispersion`](../src/sctpy/_regularize.py#L379)：`log1p(gmean/theta)/log(10)` 及反变换 | [P22] 支持按丰度正则化；[V] 提供 `od_factor` 名称，但名称不是完整数学定义。 | 证明为何平滑该变换，给出与其他 dispersion 参数化的关系、偏差及适用条件；核对几何均值和伪计数的使用位置。 |
| M5 | [`_regularize.r_stats_binned_outliers`](../src/sctpy/_regularize.py#L344)、[`_robust_scale_by_bin`](../src/sctpy/_regularize.py#L330)：两套平移网格、`ptp(x)*bw/2`、MAD、阈值 10；至少保留 3 点才剔除异常值 | MAD 原理可独立说明；现有数值 golden 仅支持指定样例的行为。**完整组合待补。** | 解释双网格、bin width、阈值与退化 MAD 处理；解释异常值过多时重新使用原训练点的规则。用冻结的模拟/留出集评估鲁棒性。 |
| M6 | [`_fit.theta_moment_ratio_mask`](../src/sctpy/_fit.py#L30)、[`core.fit_transform`](../src/sctpy/core.py#L378)：`(mean**2/(variance-mean))/theta < 1e-3` 时设 theta 为 Inf | NB 方差恒等式可推导矩估计；**不直接给出 `1e-3` 或后处理顺序。** | 说明该阈值控制什么误差，何时与 Poisson 极限等效，及其与异常值剔除/平滑先后顺序的依据。 |
| M7 | [`core.fit_transform`](../src/sctpy/core.py#L378)：Poisson intercept 分母用 eligible genes 总和，offset 用全部输入 library size；residual floor 的中位数也用 eligible genes | [P22] 给出 Poisson offset 公式与非零中位数 floor；仓库 [规范](SCT_V2_SPEC.md) 将 eligible-total 不对称标为实测兼容约定。 | 明确过滤前后基因全集，并与论文 offset 定义对照。兼容约定可以保留，但不能直接称为 full-exposure MLE；需评估过滤和极端深度的影响。 |
| M8 | [`core._residuals_and_summary`](../src/sctpy/core.py#L247)：`np.maximum(np.rint(target_mu + pearson*sqrt(target_variance)), 0)`；原深度 floor、未 clipping residual、目标深度不加 floor | [P22] 给出固定深度逆变换，但正文公式使用 floor；当前代码使用 nearest-even rounding。已有 R fixture 比较不能消除这一区别。 | 明确论文数学输出与 R `vst` 兼容输出的契约；补取整阶段、零截断、两端方差 floor 的说明与误差分析。不要为了“忠实论文”静默改变默认结果。 |

## 数值实现及工程选择

| ID | 代码位置与关键实现 | 已有依据及其边界 | 要补充的证据 / 判断 |
|---|---|---|---|
| N1 | [`_regularize._r_stats_pair_counts`](../src/sctpy/_regularize.py#L129)：1000 bins、`size > bins//2` 分支、`ptp(x)*1.01/bins`、trunc、int32 范围分支 | [B] 明确 bins 和 `nb/2` 的算法规模切换，不能把这两项写成完全无依据。 | 尚需说明 `1.01`、精确舍入/平移规则和 int32 分支的误差及必要性；添加平移、反号、极端范围的独立验证。 |
| N2 | [`_regularize.r_stats_sj_bandwidth`](../src/sctpy/_regularize.py#L191)、[`sheather_jones_bandwidth`](../src/sctpy/_regularize.py#L61)：pilot `1.24/1.23` 与 `0.920/0.912`，`1.357`、`1.144`、区间扩展 `1.2`；粗精度求根与 fallback | [SJ] 为方法来源；[B] 确认 STE、扩展区间和 `tol=0.1*lower` 等公开契约。 | 为每个 pilot 常数建立公式/尺度转换映射；分开审查数学估计器、离散近似、搜索策略和退化回退。不能因用了 R 风格数值契约就断言整个 SJ 方法无论文支持。 |
| N3 | [`_regularize.r_stats_ksmooth_normal`](../src/sctpy/_regularize.py#L294)：`sigma=bw*0.3706506`、4-sigma 截断、空窗口 NaN；clamp 支持域 | [K] 给出 Gaussian quartile 的带宽约定，可推导精确比例 `1/(4*Phi^-1(0.75))`。 | 补该比例舍入、4-sigma 截断后的归一化误差、空窗口行为及 clamp 对边缘预测的影响。 |
| N4 | [`selection.create_selection_manifest`](../src/sctpy/selection.py#L77)、[`_inverse_density_weights`](../src/sctpy/selection.py#L140)：cell-first、PCG64、排序、512 网格、`1.06` 带宽、Gaussian filter 截断 | [P22] 支持按基因丰度密度抽样；[B] 提供 `1.06` 带宽变体的公开背景。精确网格和随机序列属于实现选择。 | 区分逆密度原则、数值密度近似与采样顺序。补全数据/抽样细胞统计口径及尾部权重误差，不要求同 seed 自动等于 R 清单。 |
| N5 | [`_fit._cox_reid_score_batch`](../src/sctpy/_fit.py#L254)、[`_scaled_digamma_difference`](../src/sctpy/_fit.py#L229)，及 GPU 对应路径：`phi*mu < 1e-10/1e-4` 的分段近似与 clip；digamma B12 展开 | [D] 支持 digamma 展开与正实数余项分析；现有 `test_dispersion_accuracy.py` 有独立有限和参考。 | 不要把已补的 digamma 依据抹去；剩余重点是 likelihood-term 的每个切换阈值、总 score 消减误差和很大 counts 的覆盖。 |
| N6 | [`_fit.fit_nb_offset`](../src/sctpy/_fit.py#L137)、Cox–Reid 求根及 [`_gpu._dispersion_roots_cupy`](../src/sctpy/_gpu.py#L84)：intercept 范围、theta bounds、`phi=[1e-8,1e16]`、48 轮、容差、CPU 分支核对 | 这是数值/工程契约，不必全部归为缺少生物学论文。本 PR 证明最大 log 区间经 48 次二分宽度小于 `2e-13`，并测试失败状态。 | 保留“区间足够窄”与“统计估计正确”的区别；补参数边界的适用域、边界敏感性及实际 CUDA 验证。当前环境无 CUDA，不能把 skip 当成通过。 |

## 本轮已经解决、无需作为待找论文事项的部分

- int64 转换的上界：float64 下 `int64.max` 被表示为 `2**63`，因此在转换前使用 `>= 2**63` 拒绝；邻近可表示值已测试。
- 最终 raw 参数的 objective 重算：直接用 NB/Poisson 分布 log-PMF 交叉验证，覆盖 refit、theta 后处理和抽样细胞。
- 收敛状态：有效根/Poisson 边界可成功；上界无根、非有限值、迭代耗尽、refit 失败明确失败。初始拟合失败项不进入 dispersion trend；拟合或 refit 失败项不进入后续 smoother。
- 这些修复不补齐 M1–M8 的统计论证，也不改变既有开发来源记录。

## 补证提交格式

建议先处理 M1、M2、M3、M4、M8，再处理 M5–M7 和数值近似。
每项补齐：`ID → 文献/公开文档链接 → 精确页码/公式/段落 → 本项目变量映射 → 假设与适用域 → 独立推导/数值测试 → 决定`。
决定可以是“已有支持”“明确为兼容约定”“需单独实验”“拟替换”；不要用新增引用自动关闭尚未覆盖的具体常数。
用于发现问题和选择方案的数据应标作开发集，最终评价另用留出数据；不要事后放宽既有门槛。

## 资料入口

本轮核对了 P22 正文、V/B/K 公开文档及 D。G 的补充材料和 SJ 原文需要后续逐式核对；本表不声称完成了新的穷尽文献检索。

- [P22: Choudhary & Satija 2022，Methods][P22]
- [G: Ahlmann-Eltze & Huber 2020，及其 Supplementary Appendix S1][G]
- [SJ: Sheather & Jones 1991][SJ]
- [V: sctransform `vst` 公开 API 文档][V]
- [B: R bandwidth 公开 API 文档][B]
- [K: R `ksmooth` 公开 API 文档][K]
- [D: NIST DLMF 5.11.2 与 5.11(ii)][D]

[P22]: https://link.springer.com/article/10.1186/s13059-021-02584-9
[G]: https://doi.org/10.1093/bioinformatics/btaa1009
[SJ]: https://doi.org/10.1111/j.2517-6161.1991.tb01857.x
[V]: https://rdrr.io/cran/sctransform/man/vst.html
[B]: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/bandwidth.html
[K]: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ksmooth.html
[D]: https://dlmf.nist.gov/5.11
