# numbatalib：用 Numba 彻底重写 TA-Lib（161 个指标），无需编译安装，兼容 talib API

![numbatalib 封面](assets/wechat/banner.png)

> 适合人群：量化/交易/研究里需要 TA 指标的人；被 TA-Lib 安装/编译折腾过的人；希望“结果一致 + 性能别掉”的人  
> 一句话：**TA-Lib Core 的 161 个函数，用 pure Python + Numba 重写，并做了逐函数对齐测试，还提供 talib 兼容层。**

---

## 1）为什么很多人“想用 TA-Lib，但又怕装 TA-Lib”？

TA-Lib 好用，但现实里经常遇到：

- 原生是 C 库：不同系统/编译链/依赖很容易踩坑
- Python 侧依赖 C 扩展：装 wheel 还好，源码安装就容易卡
- 你只想算个 `RSI/SMA/MACD`，不想为此研究半天环境

于是就有了一个常见需求：**能不能像装普通 Python 包一样安装 TA-Lib，但又别牺牲性能和一致性？**

---

## 2）numbatalib 是什么？一句话解释

**numbatalib = TA-Lib Core（161 函数）的 Python + Numba 版本。**  
目标是：

- 算法一致：和本地 `talib`（TA-Lib Python 包）逐函数对齐
- 性能不掉：用 Numba JIT 生成接近 C 的循环性能
- 安装简单：不依赖 `libta-lib`，不需要本地编译 C 扩展

![流程示意](assets/wechat/flow.png)

---

## 3）它的亮点是什么？（对实战特别友好）

### 亮点 A：161 个函数全部覆盖

- TA-Lib Core：**161 / 161** 已实现
- 本机 `talib==0.4.32` 只暴露 158 个函数；缺失的 `ACCBANDS/AVGDEV/IMI` 也已对齐，并用上游 C 参考实现做了校验

### 亮点 B：talib 兼容层（最小习惯成本）

如果你希望“导入方式、参数、stream/abstract、报错信息”都尽量像 `talib`，可以直接：

```python
import numpy as np
import numbatalib.talib as talib

x = np.random.default_rng(0).normal(size=1000).cumsum()

talib.SMA(x, timeperiod=20)
talib.stream.RSI(x, timeperiod=14)  # streaming: 返回最后一个标量
from numbatalib.talib import abstract
abstract.Function("MACD")(x, fastperiod=12, slowperiod=26, signalperiod=9)
```

### 亮点 C：对齐测试不是“抽样”，而是逐函数跑

仓库里有自动化脚本，会生成：

- `generated/parity_results.csv`：逐函数一致性（对比本地 talib）
- `generated/bench_results.csv`：逐函数速度对比（对比本地 talib）
- `port_checklist.csv`：checklist（覆盖/测试/速度/备注）

---

## 4）安装（像普通包一样）

```bash
pip install numbatalib
```

不需要你提前装 `libta-lib`，也不需要编译 C 扩展。

---

## 5）基础用法：直接当指标库用

```python
import numpy as np
import numbatalib as ta

x = np.random.default_rng(0).normal(size=20000).cumsum()

rsi = ta.RSI(x, timeperiod=14)
macd, signal, hist = ta.MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)
```

补充：`numbatalib` 用动态导出方式暴露函数名（`ta.SMA/ta.RSI/...`），你也可以用：

```python
ta.available_functions()
ta.implemented_functions()
```

---

## 6）一致性怎么保证？（你关心的“结果一模一样吗”）

### 对比本地 talib（158 个函数）

用 `pytest` 做逐函数回归（多种长度/随机种子）：

```bash
pytest -q
```

### 对比上游 TA-Lib C（补齐的 3 个函数）

仓库里提供了上游 C 的最小参考构建（Windows 下用 Zig 构建 DLL），用于对齐 `ACCBANDS/AVGDEV/IMI`：

```bash
python tools/compare_vs_upstream_c.py
```

结果保存为 `generated/parity_results_upstream_c.csv`。

---

## 7）速度如何？会更快吗？差多少？

结论先说：

- 在作者这台 Windows + conda 环境的基准测试里，**大部分函数与 talib 同量级**（中位数约 1.02x）
- 有一部分函数更快（例如 `SQRT/CEIL/FLOOR` 这类偏数学运算，Numba 对循环/调用开销更友好）
- 少数函数更慢（例如 `MOM/MAVP/MIDPOINT` 等），仍有优化空间

你可以在自己的机器上复现（会输出 CSV）：

```bash
python tools/compare_vs_talib.py --bench --write-checklist
```

建议 benchmark 时注意 Numba 的 JIT 预热（第一次调用会编译）。

---

## 8）常见问题（你可能会遇到）

### Q1：第一次跑怎么会慢一截？

Numba JIT 需要编译，首次调用会有编译成本；后续调用性能才是常态。

### Q2：和 talib 的 NaN/起始 padding 一致吗？

以“结果对齐”为优先目标：输出数组的 NaN padding/起始有效位与 talib 对齐（详见测试与 CSV）。

### Q3：pandas / polars Series 支持吗？

`numbatalib.talib` 兼容层会像 talib 一样：输入是 Series 时输出也尽量保持 Series（并保留 index）。

---

## 9）你会得到什么？（把它放进你的策略/研究里）

- 更“pip friendly”的 TA 指标库（尤其 Windows/CI 环境）
- 一个尽量兼容 `talib` 的 API（减少迁移成本）
- 可复现的逐函数一致性/性能基准（CSV 留痕）

---

## 10）项目地址 & 下载

- GitHub：`https://github.com/bluesHeart/numbatalib`
- PyPI：`https://pypi.org/project/numbatalib/`

---

## 关注公众号：市场逍遥游

![市场逍遥游 公众号二维码](assets/wechat/wechat_mp_card.png)

