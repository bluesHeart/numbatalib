# NumbaTA-Lib 重构计划（上游基线 TA-Lib v0.6.4）

> 目标：把 TA-Lib 的 C 实现「等价」迁移到 Python + Numba（无 C 扩展），在保持算法/边界行为一致的前提下做到接近原生 C 的性能。

> 2026-01-29 当前状态：TA-Lib Core **161/161** 已在 `numbatalib` 中实现；本仓库默认以本机已安装的 `talib` 作为对齐基准（`generated/pytest_results.txt`、`generated/parity_results.csv`、`generated/bench_results.csv`）。

## 0. 现状盘点（上游基线）

- 上游源码：`upstream/ta-lib`（tag `v0.6.4`，commit `43f9d50`）
- 指标函数（Core indicators）：**161** 个（`TA_XXX`）
  - 每个指标在 `include/ta_func.h` 中对应 **3** 个入口：
    - `TA_XXX`（double 输入版本）
    - `TA_S_XXX`（float 输入版本）
    - `TA_XXX_Lookback`
  - `include/ta_func.h` 共 **483** 个 `TA_LIB_API` 原型（= 161×3）
- `src/ta_func` 下有 **164** 个 `ta_*.c` 文件：其中 **3** 个未出现在 `include/ta_func.h`（`NVI`/`PVI`/`utility`）
- 其他公共 API：
  - `include/ta_common.h`：11 个 `TA_LIB_API`
  - `include/ta_abstract.h`：22 个 `TA_LIB_API`

> 本项目主线优先覆盖 `ta_func.h` 的 161 个指标（含 lookback 与 float 输入版本）；`NVI/PVI` 视需求作为“扩展指标”单独纳入。

## 1. Done Definition（“一致 + 不掉性能”如何验收）

### 1.1 一致性（Correctness）

- **数值一致性**：对每个指标，在同一输入与参数下：
  - `outBegIdx/outNBElement` 一致（或在 Python API 中等价映射）
  - 输出数组长度/对齐规则一致（例如 Python 层用 `NaN` 填充前缀）
  - 数值误差阈值：默认 `rtol=1e-12, atol=1e-12`（必要时按函数类型分类调参）
- **边界行为一致**：
  - `startIdx/endIdx` 的越界检查、`timeperiod` 等参数范围、空数组、全 NaN、包含 inf 等
  - `TA_SetCompatibility`、`TA_SetUnstablePeriod` 影响的指标行为必须可复现

### 1.2 性能（Performance）

- 基准：同机器、同输入规模下对比上游 C（或 Python TA-Lib 包装）：
  - **大数组（>= 1e6）**：单函数运行时间目标接近 C（经验目标：不慢于 C 的 1.2×；先以 1.5× 作为阶段性门槛）
  - **小数组（<= 1e4）**：允许一定常数开销，但要避免数量级差距
- 约束：默认 **`fastmath=False`**（避免改变数值语义）；如某些函数确有必要开启，必须单独标注并提供对齐证明。

## 2. API 设计（对外接口与对齐策略）

### 2.1 两层 API

1) **Raw / C-like API**（用于严格对齐与测试）
- 形态尽量贴近 TA-Lib：`(startIdx, endIdx, inputs..., opts...) -> (outBegIdx, outNBElement, outputs...)`
- 输出为“紧凑结果”（不做 `NaN` 前缀），以便与 C 逐元素对比。

2) **Pythonic API**（方便用户使用）
- 形态对齐 Python `talib` 常见用法：`FUNC(array, **params) -> np.ndarray | tuple[np.ndarray,...]`
- 输出与输入长度一致，`NaN` 前缀对齐（或提供 `outBegIdx` 以便用户对齐）。

### 2.2 dtype 策略（对齐 TA_ 与 TA_S_）

- 内核统一以 `float64` 计算与输出（TA-Lib 输出多为 double），但允许输入为：
  - `float64`（对应 `TA_XXX`）
  - `float32`（对应 `TA_S_XXX`）
- 整型输出（如 `CDL*`）统一 `int32`。

## 3. 代码组织（建议目录结构）

- `upstream/ta-lib/`：上游只读基线（用于比对/提取元数据）
- `numbatalib/`：Python 包
  - `_settings.py`：compatibility/unstablePeriod 等运行时设置（Python 层持有）
  - `_core/`：Numba 内核与共享工具（移动均线、滚动统计、hilbert 组件等）
  - `_func/`：按指标组织的实现（或按 group 组织）
  - `_generated/`：从上游元数据生成的清单、签名、docstring、参数范围等（禁止手写）
- `tools/`：辅助脚本（提取函数清单、生成 wrapper、构建上游 DLL、生成 golden data）
- `tests/`：一致性测试（对比上游）+ 基准测试（可选 asv/pytest-benchmark）
- `docs/`：开发文档（移植约定、对齐差异、性能指南、PORT_STATUS）

## 4. 元数据与自动化（降低 161 个函数的手工成本）

### 4.1 函数清单（Source of truth）

- 使用 `upstream/ta-lib/src/ta_abstract/ta_group_idx.c` 与/或 `include/ta_func.h` 生成：
  - `function_list.json`：161 个函数名、group、输入输出数量、参数列表与范围
  - `wrappers.py`：自动生成 Pythonic API 的参数签名与 docstring（只负责参数解包与 dtype 规范化，不包含算法）

### 4.2 迁移辅助工具

- 生成“移植骨架”：每个函数一个文件 + TODO 片段 + 对应上游 C 文件路径
- 生成“对齐用例”：为每个函数随机生成一组合法参数（覆盖边界：最小期、最大期、默认值、NaN/inf）

## 5. 对齐测试（Golden reference 体系）

### 5.1 参考实现来源（优先级）

1) **本仓库上游 TA-Lib 编译出的 DLL**（推荐，版本完全可控）
- 用 CMake/VS Build 把 `upstream/ta-lib` 编译成动态库
- 用 `ctypes` 或 `cffi` 调用（优先走 `ta_abstract` 接口以减少手写 wrapper）

2) **Python TA-Lib（`import talib`）**（可作为快速回归/对照）
- 若环境可用，作为第二套参考，主要用于 sanity check。

### 5.2 测试分层

- **Unit parity**：逐函数、逐输出对比（含 `outBegIdx/outNBElement`）
- **Property tests**：随机输入、极端输入、NaN/inf、不同 dtype
- **Cross-check**：对共享基础件（EMA/SMA/STDDEV）做额外强约束，因为大量函数依赖它们

## 6. 性能策略（Numba 侧约束）

- 所有核心计算使用 `@numba.njit(cache=True)`；默认 `fastmath=False`、`parallel=False`
- 避免在热循环里分配临时数组：优先外部分配/复用 buffer
- 明确 stride/contiguous 规则：入口统一转换为 `np.ascontiguousarray`
- 对“窗口滚动”类算法优先做 **O(n)** 单遍实现，避免每点重复求和
- 针对 `CDL*`（大量分支）：
  - 保持逻辑与上游一致
  - 必要时用局部 helper 函数减少重复计算（在不改变计算顺序的前提下）

## 7. 实施阶段（里程碑拆解）

### Phase 0：仓库脚手架（1–2 天）

- 建立 `pyproject.toml`、包目录 `numbatalib/`、最小可 import
- 加入 `tools/` 与 `tests/` 目录
- 固化上游版本信息（写入 `docs/UPSTREAM.md`）

验收：
- `python -c \"import numbatalib\"` 成功
- CI（如有）能跑基础测试

### Phase 1：函数清单与生成器（2–4 天）

- 解析上游生成：
  - 161 个函数名清单（含 group）
  - 每个函数的 inputs/outputs/opt 参数与范围（来自注释/ta_abstract）
- 自动生成 wrapper 骨架与 docstring

验收：
- 自动生成的清单数量 == 161
- wrapper 可被 import（即便实现仍是 TODO）

### Phase 2：参考实现与对齐框架（3–7 天）

- 编译上游 TA-Lib 并提供 Python 调用（推荐走 `ta_abstract`）
- 建立“单函数对齐”测试框架：可指定函数名、输入、参数，输出 diff 报告

验收：
- 对任意一个函数（如 `SMA`）可自动跑对齐对比并给出误差报告

### Phase 3：核心基础件（1–2 周）

优先实现“被大量复用”的 building blocks，并先通过对齐测试：
- MA 家族：SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA
- 滚动统计：SUM/MIN/MAX/STDDEV/VAR
- 常用变换：ROC/ROCP/LINEARREG 系列等（视依赖关系）

验收：
- 选定的 10–20 个高频函数全量通过对齐测试
- 性能基准达到阶段门槛（例如 <= 1.5× C）

### Phase 4：按 Group 批量迁移（滚动进行，4–8 周）

建议顺序（先简单、再复杂、最后大量分支）：
1) Math/Transform（ACOS/ASIN/ATAN/SQRT/LOG/EXP…）
2) Price Transform（AVGPRICE/TYPPRICE/WCLPRICE…）
3) Overlap Studies（BBANDS/SAR/SAREXT/MIDPOINT…）
4) Momentum（RSI/MACD/STOCH/ADX/CCI/MFI…）
5) Volatility/Volume（ATR/NATR/ADOSC/OBV…）
6) Statistic（BETA/CORREL/LINEARREG_*…）
7) Cycle / HT 系列（最难，最后单独攻关）
8) Pattern Recognition（`CDL*`，数量大，尽量用生成器+逐函数对齐）

验收：
- 每迁移完一个 group：该 group 下全部函数通过对齐测试
- 更新 `docs/PORT_STATUS.md`（完成/阻塞原因/差异说明）

### Phase 5：完善抽象层与高级能力（可选，2–4 周）

- 复刻 `ta_abstract`：提供函数发现、参数范围查询、doc 查询等（便于上层框架自动化）
- 提供“批量计算 pipeline”（一次输入，多个指标复用中间结果/缓存）

验收：
- 可以按函数名动态调用（类似 TA-Lib abstract API）

### Phase 6：发布与长期维护（持续）

- API 稳定性承诺（semver）
- 增加回归数据集（固定 seed + 固定用例）
- 跟踪上游 TA-Lib 新版本变更（仅在明确需要时升级基线）

## 8. 风险点与应对

- **Hilbert/HT 系列**：对浮点误差与实现细节敏感 → 单独做高强度对齐与逐行移植
- **全局状态（compatibility/unstablePeriod）**：Numba 不适合可变全局 → 用 Python 层持有状态、调用时显式传参进 njit 内核
- **CDL* 大量分支**：易出 off-by-one → 强制对齐测试覆盖（小样本 + 随机 + 边界）
- **性能与一致性的冲突**：以一致性为先；性能优化必须提供对齐证据与 benchmark 报告
