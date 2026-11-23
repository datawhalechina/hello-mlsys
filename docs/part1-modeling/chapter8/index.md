---
title: "第 8 章 推理框架的系统取舍、Profile 与性能分析"
description: "Hello MLSys · 第 8 章 推理框架的系统取舍、Profile 与性能分析 · 同卡 vLLM vs llama.cpp、显存失败模式、留出预测、框架级 Profile 与选型决策"
---

# 第 8 章 推理框架的系统取舍、Profile 与性能分析

同一个模型、同一张显卡、同一组请求，只把推理框架从 vLLM 换成 llama.cpp，首 token 延迟就可能相差数倍。这个差异不能简单归因成“vLLM 的 GEMM 更快”：权重格式、精度、上下文预算、图执行、算子融合、批处理与 KV 池大小都同时变了。如果只贴一张吞吐排行榜，读者无法判断哪一部分是框架能力，哪一部分是配置和负载造成的。

本章是第 1 篇的收口章。我们先固定模型、硬件、输入输出长度与 SLA，再在同一张 RX 7900 XTX 上对比 vLLM 与 llama.cpp。显存账本先排除根本起不来的长上下文配置；主扫描给出 C=1/8/32/64 的延迟与有效吞吐；C=16/48 作为测前冻结的留出点，检验一条只用 C=1/8/32 校准的幂律；最后用框架级 kernel profile 与第 1、3、7 章的模型解释差异，并把所有不等价项逐条列出。

本章能给出的结论有明确边界：它完成的是 vLLM 与 llama.cpp 的单卡短上下文横评，不是全框架排名；模拟筛选器只对静态显存判定有效，它的延迟预测在 48 并发时严重失真，负结果会原样保留。

## 8.1 课程坐标与本章增量

Harvard MLSysBook 的 *ML Frameworks* 把框架问题拆成执行、微分与抽象三层，其中 eager dispatch、静态图、graph capture 与 kernel fusion 正好解释了“同一模型为什么换个引擎就变快”。*Benchmarking* 强调 benchmark 与生产负载之间的缝隙，*Model Serving* 则把请求生命周期、batching、runtime 选择与 KV 预算连在一起。Stanford CS336 的 *Inference*、CMU 15-442/15-642 的 LLM serving 两讲，与 Orca、PagedAttention/vLLM、SGLang、Sarathi-Serve、DistServe 等论文，分别覆盖 continuous batching、分页 KV、前缀复用、chunked prefill 与 goodput。



## 8.2 框架取舍与测量契约

框架级取舍至少有四条互相牵制的轴：

1. **执行路径**：逐算子 eager dispatch 的 launch 次数多；graph capture 与编译能把一串小 kernel 压成更少的提交。它主要影响 decode 的 launch 间隙与 GPU busy。
2. **算子融合**：融合 norm、RoPE、KV 写入等小算子，同时减少访存流量和 kernel 数。它不能只用 GEMM 占比解释。
3. **调度与批处理**：continuous batching 让完成的序列立刻让位；chunked prefill 控制大块 prefill 对 decode 的干扰；调度器也决定排队与尾延迟。
4. **KV 与上下文预算**：分页 KV 降低碎片，但池子大小、每序列上下文和最大并发共同决定“装得下多少条请求”。

这四轴无法靠单个 tok/s 数字分开，所以本章先冻结测量契约：

| 项 | 取值 |
|-|-|
| 硬件 | AMD Radeon RX 7900 XTX 23.98 GiB，`gfx1100`，宿主 GPU0，实验期独占 |
| 模型 | Qwen3-4B，4,022,468,096 参数，36 层，hidden=2560，32 q / 8 kv heads，head_dim=128 |
| 负载 | closed loop，input=256 / output=128，`ignore_eos`，temperature=0，top_p=1，seed=42 |
| 样本 | 200 prompts/点，warmup=2 |
| SLO | TTFT P99 ≤ 800 ms，TPOT P99 ≤ 100 ms，E2EL P99 ≤ 15,000 ms，单点失败率 ≤ 1% |
| 延迟口径 | TTFT = 首 token − 请求发出；TPOT = `(E2EL − TTFT) / (completion_tokens − 1)`；E2EL = 完成 − 请求发出 |
| 吞吐口径 | 裸输出 tok/s 与“同时满足三条 P99 的单请求 SLO”的 goodput 分开报告 |

TPOT 这一项必须特别说明。压测最早按 SSE chunk 间隔计算 TPOT，但 vLLM 和 llama.cpp 都可能把多个 completion token 合并进一个 chunk；chunk 数不等于 token 数，会把 TPOT 算高。本章全部采用 `results/framework/analysis.json` 里由 `requests.jsonl` 重算的 `tpot_corrected_ms`，原始 `summary.json` 与原始请求行保留不动。修正前后的差异可到 C=1 vLLM 的旧 TPOT P99 14.22 ms 与新值 10.19 ms 中看到；正文不再使用旧的 chunk 间隔数字。

## 8.3 先筛显存：131K 上下文在任何延迟比较之前就装不下

Qwen3-4B 每个 token 的 KV 字节数是

$$
m_{\mathrm{token}}=2\times 36\times 8\times 128\times 2=147{,}456\ \mathrm{B}=144\ \mathrm{KiB}.
$$

131,072 token 的 KV 需要 $131072\times147456\approx18.00$ GiB。再加上约 7.49 GiB 的 F16/BF16 权重，已经超过 23.98 GiB 的物理容量；还没算 graph、workspace 与运行时。

![四组配置的显存账本。两个 131K 上下文候选都越过物理容量，在启动阶段就被判为静态不可行。](./images/fig8-5-memory-first.png)

| 配置 | 权重 (GiB) | KV (GiB) | graph / workspace / runtime (GiB) | 合计 (GiB) | 账本判定 |
|-|-:|-:|-:|-:|-|
| vLLM，4K ctx，C48 | 7.49 | 10.61 | 5.05 | 23.15 | 物理容量内 |
| vLLM，128K ctx，C1 | 7.49 | 18.00 | 5.05 | 30.54 | 超容量 |
| llama.cpp，512 ctx/slot，C48 | 7.49 | 4.50 | 0.35 | 12.34 | 物理容量内 |
| llama.cpp，131K ctx，C1 | 7.49 | 18.00 | 0.35 | 25.84 | 超容量 |

这张表是“声明配置 + 实测容量 + 第 3 章 KV 几何”的账本，不是进程 RSS。真实运行中，vLLM 的显存采样恒定为 21,689,548,800 B（约 20.20 GiB），llama.cpp 在 13.5 GB 附近波动；它们与账本的差值来自 `gpu-memory-utilization` 预留、分配器粒度与运行时空闲，不能拿 23.15 GiB 的合计去冒充实测占用。

账本判死后，真机又做了一次失败探针：用同一个 F16 GGUF、`--ctx-size 131072` 启动 llama.cpp。它没有进入请求处理，而是在创建 KV 时失败：

```text
n_ctx_seq (131072) > n_ctx_train (40960) -- possible training context overflow
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 18432.00 MiB on device 0: cudaMalloc failed: out of memory
failed to allocate ROCm0 buffer of size 19327352832
```

18,432 MiB 正好是 18.00 GiB，与 KV 账本一致；失败发生在 `llama_init_from_model`，不是压测超时。日志在 `results/framework/llamacpp-memory-fail/probe.log`。这给出一个比“模拟器说装不下”更硬的失败模式：**长上下文候选的首次破坏约束是静态显存，延迟曲线根本没有机会出现。**

模拟筛选器也得到 4 个 FAIL，但两类失败必须分开看：

| 候选 | 模拟结论 | 首次破坏 |
|-|-|-|
| vLLM BF16 4K / C48 | FAIL | 解析延迟 P99 TTFT 66,029 ms > 800 ms |
| llama.cpp F16 512 / C48 | FAIL | 解析延迟 P99 TTFT 66,029 ms > 800 ms |
| vLLM BF16 131K / C1 | FAIL | 固定预算 32,794,685,234 B > 25,753,026,560 B |
| llama.cpp F16 131K / C1 | FAIL | 固定预算 27,748,098,662 B > 25,753,026,560 B |

两个 131K 候选的 `STATIC_MEMORY_INFEASIBLE` 与真机 OOM 方向一致，可以作为有效筛选证据。两个 48 并发候选都算出约 66 秒 TTFT，而真机留出只测得 1.15 秒（vLLM）与 2.42 秒（llama.cpp）。这不是“预测偏保守”，而是模型失真：当前 analytical backend 对 batch 内权重与 KV 读取的复用建模不足，把并发请求近似成一串独立全量扫描。**因此本章只采纳筛选器的静态显存判定，不采纳它的延迟数字。** 负结果保留在 `results/screen-qwen4b-7900xtx-v2/`。

## 8.4 主扫描：裸吞吐与有效容量不是一回事

两个服务都关闭了 prefix cache，负载没有共享前缀。vLLM 的 KV 池是 77,248 token（10.61 GiB），llama.cpp 是 64 slots × 512 token = 32,768 token（4.50 GiB）。主扫描结果如下：

| C | TTFT P99 (ms) | TPOT P99 (ms) | E2EL P99 (ms) | 裸输出 (tok/s) | SLO goodput (req/s) | 合格请求 | 判定 |
|-:|-:|-:|-:|-:|-:|-:|-|
| **vLLM** 1 | 40.24 | 10.19 | 1,330.00 | 96.43 | 0.753 | 200/200 | PASS |
| **vLLM** 8 | 206.05 | 14.25 | 1,877.16 | 546.79 | 4.272 | 200/200 | PASS |
| **vLLM** 32 | 780.35 | 20.74 | 2,920.09 | 1,406.22 | 10.986 | 200/200 | PASS |
| **vLLM** 64 | 1,564.51 | 30.30 | 4,948.88 | 1,834.59 | 11.323 | 158/200 | FAIL |
| **llama.cpp** 1 | 124.86 | 12.55 | 1,693.72 | 79.79 | 0.623 | 200/200 | PASS |
| **llama.cpp** 8 | 778.33 | 32.47 | 4,900.64 | 297.29 | 2.323 | 200/200 | PASS |
| **llama.cpp** 32 | 1,867.03 | 86.37 | 12,015.42 | 412.89 | 2.500 | 155/200 | FAIL |
| **llama.cpp** 64 | 3,859.00 | 112.41 | 17,146.44 | 553.48 | 0.822 | 38/200 | FAIL |

![TTFT P99 随并发上升。vLLM 在 C48 留出点越线，llama.cpp 在 C16 留出点已经越线。](./images/fig8-1-ttft-p99.png)

先看裸吞吐：vLLM 从 C1 的 96 tok/s 涨到 C64 的 1,835 tok/s，llama.cpp 从 80 tok/s 涨到 553 tok/s。如果只画这张曲线，会得到“并发越高越好”的结论。但 SLO goodput 在越线点立刻变脸：vLLM 从 C32 的 10.986 req/s 只微涨到 C64 的 11.323 req/s，并发翻倍只换来 3.1% 的有效产出；llama.cpp 在 C64 反而从 2.500 跌到 0.822 req/s。

![左图为裸输出吞吐，右图为 SLO goodput。裸吞吐继续上涨时，合格请求的交付速率可能停滞或塌陷。](./images/fig8-2-throughput-goodput.png)

判定不能只看三条 P99 阈值。主扫描中：

- vLLM C64 的 TPOT P99 30.30 ms、E2EL P99 4,948.88 ms 都远低于阈值，但 TTFT P99 1,564.51 ms 越线，且 200 条里有 42 条不满足单请求 SLO，失败率 21% > 1%，所以整体 FAIL。
- llama.cpp C32 的三条 P99 分别为 1,867.03 / 86.37 / 12,015.42 ms，TTFT 与 E2EL 已越线；合格请求只有 155/200，失败率 22.5%。
- llama.cpp C64 的 TTFT、TPOT、E2EL 三条 P99 同时越线；合格请求只剩 38/200，goodput 掉到 C8 的约三分之一。

工程上更该记住的是“最大已测通过点”和“首次违约点”：

| 框架 | 最大已测通过点 | 首次违约点 | 裸吞吐峰值所在点 | 有效容量结论 |
|-|-:|-:|-:|-|
| vLLM | C32 | C48（留出） | C64 | C32 是本章的工程安全水位 |
| llama.cpp | C8 | C16（留出，仅超 38.66 ms） | C64 | C8 是本章的工程安全水位 |

little 定律可以核对“多出来的并发去了哪里”。闭环稳态下，在途请求数约等于完成速率乘以平均逗留时间。vLLM C32 是 10.986 req/s，$32/10.986\approx2.91$ s，与 E2EL P50 2.71 s 同一量级；C64 是 14.333 req/s，$64/14.333\approx4.47$ s，与 E2EL P50 3.98 s、P99 4.95 s 对齐。llama.cpp C32 是 3.226 req/s，$32/3.226\approx9.92$ s，E2EL P50 9.39 s；C64 是 4.324 req/s，$64/4.324\approx14.80$ s，E2EL P50 14.47 s。并发翻倍后，新增在途请求主要变成更长的逗留时间，而不是更高的完成速率——这正是第 7 章排队拐点在闭环压测里的表现。

KV 抢占不是这两组 C64 尾延迟的主因。vLLM 全扫描的 `num_preemptions_total` 都是 0；llama.cpp 配置了 64 slots × 512 token，64 × (256+128) = 24,576 token 仍在 32,768 token 的总上下文内。llama.cpp 没有暴露与 vLLM 可直接比较的 preemption 计数器，因此本章不声称它发生了或没有发生 KV 驱逐；能说的是，**显存池容量本身足以容纳 64 条定长请求，观察到的尾延迟增长首先来自准入、调度与每步计算的服务时间。**

## 8.5 留出预测：只用 C=1/8/32 拟合，C=16/48 测前冻结

主扫描不是盲预测：C=1/8/32/64 先测量，C=64 在预测写盘前已经被看见。为了保留一个真正的“预测→实测”检验，只拿 C=1/8/32 的三点拟合两条幂律：

$$
\log(\mathrm{TTFT}_{99})=a_{\mathrm{T}}+b_{\mathrm{T}}\log C,\qquad
\log(\mathrm{TPOT}_{99})=a_{\mathrm{P}}+b_{\mathrm{P}}\log C,
$$

再用固定 OSL=128 得到 $\mathrm{E2EL}_{99}=\mathrm{TTFT}_{99}+127\times\mathrm{TPOT}_{99}$。C=64 明确排除出拟合与验收；C=16/48 在预测写盘后才测量。预测文件在 `2026-11-23T04:22:20Z` 冻结，测量分析文件在 04:30:53Z 生成，时间顺序可从 `prediction-holdout.json` 与 `holdout-analysis.json` 审计。

![四个留出点的预测与实测。柱上是误差百分比，红线是对应 SLO 阈值。](./images/fig8-3-holdout-prediction.png)

| 框架 / C | TTFT P99 预测→实测（误差） | TPOT P99 预测→实测（误差） | E2EL P99 预测→实测（误差） | SLO 预测→实测 |
|-|-|-|-|-|
| vLLM / 16 | 408.74 → 386.20（−5.5%） | 17.40 → 16.57（−4.8%） | 2,618.59 → 2,178.94（−16.8%） | PASS → PASS |
| vLLM / 48 | 1,039.92 → 1,147.30（+10.3%） | 21.72 → 26.57（+22.3%） | 3,798.02 → 4,033.74（+6.2%） | FAIL → FAIL |
| llama.cpp / 16 | 1,173.08 → 838.66（−28.5%） | 54.42 → 53.18（−2.3%） | 8,084.38 → 7,124.30（−11.9%） | FAIL → FAIL |
| llama.cpp / 48 | 2,789.04 → 2,421.73（−13.2%） | 99.43 → 109.23（+9.9%） | 15,416.86 → 14,858.43（−3.6%） | FAIL → FAIL |

四个 SLO 判定全部命中，平均绝对百分比误差（MAPE）是 TTFT 14.38%、TPOT 9.816%、E2EL 9.623%。但这里必须克制：

- 三个校准点拟合两个参数，幂律没有多余自由度做统计检验；预测文件里的区间只是 $e^{\pm1.96\sigma}$ 启发式带，不是置信区间。
- llama.cpp C16 实测 TTFT P99 838.66 ms，只比 800 ms 阈值高 38.66 ms，且 196/200 条请求合格。把它称为“稳健的违约边界”不诚实；它更适合被记成“边缘 FAIL”。
- vLLM C48 的 TPOT 误差达到 +22.3%，说明在接近拐点时，幂律对每 token 服务时间的增长估计不足。
- 预测的是四个留出点，不是主扫描 C=1/8/32/64 的完整盲预测，也不是容量曲线的全区间外推。

尽管有这些限制，留出检验仍然完成了一个有用的判断：在 C16，预测认为 vLLM 通过、llama.cpp 失败；在 C48，预测认为两者都失败。四个判定全中，说明“用低并发三点估计有效容量边界”在这个窄区间里可行；误差大小则说明它只适合做粗筛，不能替代实测。

## 8.6 框架级 kernel profile：少 dispatch 与高 GPU busy

延迟数字回答“谁先违约”，profile 回答“时间花在哪里”。两个框架都在 C=8、20 条请求、ISL=256/OSL=128 下采集 trace，vLLM 用 `/start_profile` + `/stop_profile` 的 torch profiler，llama.cpp 用 `rocprofv3 1.4.1` attach。原始 trace 分别约 209 MB 与 249 MB，只留在 `/tmp`，仓库提交的是按 trace 哈希绑定的摘要 `results/framework/profile-summary.json`。

![C=8 下两个框架的 kernel dispatch 数与 GPU busy。llama.cpp 每输出 token 的 dispatch 数约为 vLLM 的 1.84 倍。](./images/fig8-4-kernel-profile.png)

| 指标 | vLLM | llama.cpp |
|-|-:|-:|
| kernel dispatches | 269,892 | 497,906 |
| trace 跨度 | 8.50 s | 13.95 s |
| kernel 时间合计 | 7,456.09 ms | 9,871.50 ms |
| GPU busy（kernel / 跨度） | 87.73% | 70.77% |
| dispatches / 输出 token | 105.4 | 194.5 |
| GEMM 时间占比 | 89.18% | 89.32% |
| attention 时间占比 | 6.25% | 4.83% |
| norm / RoPE 时间占比 | 1.45% | 3.25% |

vLLM 的 trace 还能按 GPU annotation 切开：

| 阶段 | 出现次数 | kernel 数 | kernel 时间 | 注解跨度 | busy |
|-|-:|-:|-:|-:|-:|
| prefill | 5 | 2,255 | 158.40 ms | 233.60 ms | 67.81% |
| decode | 635 | 258,998 | 6,295.36 ms | 7,188.32 ms | 87.58% |
| mixed chunked prefill | 3 | 1,353 | 385.45 ms | 397.91 ms | 96.87% |

这组数字支持三个解释，但都不能越过 trace 的边界：

1. vLLM 的 HIP graph / torch.compile 与融合路径显著减少了每个输出 token 的 dispatch 数。105.4 对 194.5，不是“GEMM 快 1.84 倍”，而是提交与小算子数量少。
2. 两个框架的 kernel 时间都以 GEMM 为主，约 89%；attention 只占 5%–6%。这说明短上下文、decode 主导的负载里，注意力不是第一瓶颈，不能把差异都推给 attention backend。
3. vLLM 在 decode 阶段的 GPU busy 是 87.58%，llama.cpp 全程 70.77%。busy 高不等于延迟低，但在同一 C=8 下，它解释了 vLLM 为什么每输出 token 需要更少的墙钟时间。

profile 本身有扰动：trace 采集会拉长墙钟，attach 与 torch profiler 也不是同一条 instrumentation 路径。因此本章的 SLO 判定全部来自低开销压测，profile 只做相对归因；不能用 8.50 s 或 13.95 s 的 trace 跨度去覆盖第 7 章的正式延迟。

## 8.7 用第 1、3、7 章模型解释差异

**第 1 章 Roofline：单流 decode 先撞权重带宽。** C=1 时每次生成一个 token 都要扫过约 8.04 GB 的权重。若按 960 GB/s 的有效带宽估算，下界是 $8.04/960\approx8.38$ ms/token。实测 vLLM C1 TPOT P99 10.19 ms，相当于 790 GB/s、82% 的带宽效率；llama.cpp 12.55 ms，相当于 641 GB/s、67%。这不是精确的硬件计数器结论，但它把“低并发下 vLLM 更快”落到了权重扫描效率，而不是泛泛的“实现更好”。

**第 3 章 KV 账本：两个框架的容量分母不同。** vLLM 的 KV 池能放 77,248 token，llama.cpp 只有 32,768 token。按每条请求 384 token 计算，账本容量分别是 201 条与 85 条；前者还被 `max-num-seqs=64` 截住，后者被 64 个 slot 截住。表面上两者都能装下 C=64，但余量不同：vLLM 的池子有约 3 倍余量，llama.cpp 只剩约 1.33 倍。这解释了为什么本章不能把 llama.cpp 的高并发退化简单归因成 KV 驱逐——账本显示它还没装不下，但余量更小，调度与上下文增长更早侵蚀尾延迟。

**第 7 章排队：并发翻倍会把时间花在逗留上。** 前面用 Little 定律核对过，C32→C64 时 vLLM 的完成速率只从 10.99 涨到 14.33 req/s，llama.cpp 从 3.23 涨到 4.32 req/s；新增并发主要变成 E2EL。P99 比 P50 涨得更快，说明尾部请求承担了准入等待与 prefill/decode 干扰。vLLM 的 chunked prefill 让 mixed 阶段 busy 达到 96.87%，但它并没有消除 C64 的 TTFT 越线，只是把更多 prefill 工作塞进了同一条 decode 时间线。

**第 4 章的量化视角只作为边界出现。** 本章没有跑 INT4/FP8 的框架横评，无法回答“量化后排序是否反转”。131K 的失败探针说明，在 24 GiB 卡上，单靠 F16 权重与 KV 已经装不下；量化权重或 KV 可能改变可行性，但本章没有测，不能写成结论。

## 8.8 不等价项账本

框架横评最容易被误读的地方，是把所有差异都归给“调度器”。本章的比较至少同时改变了以下变量：

| 不等价项 | vLLM | llama.cpp | 对本实验的影响 |
|-|-|-|-|
| checkpoint / 权重格式 | BF16 Hugging Face | F16 GGUF | 权重字节接近，但转换、布局与算子选择不同 |
| 对外接口 | vLLM OpenAI server | llama.cpp server 的 OpenAI 兼容层 | 请求解析、流式封装与采样路径不同 |
| 上下文预算 | `max-model-len=4096`，64 seqs | 64 slots × 512 ctx，总 32,768 token | 每序列可用上下文与批处理上限不同 |
| KV 池 | 77,248 token / 10.61 GiB | 32,768 token / 4.50 GiB | 容量分母与余量不同 |
| 执行路径 | torch.compile + HIP/CUDA graph + chunked prefill | 融合 kernel 路径，未见同等 graph capture | dispatch 数与 GPU busy 差异的主要来源之一 |
| prefix cache | 关闭 | `--no-cache-prompt` | 本章负载无共享前缀，排序不因前缀复用反转 |
| attention / flash-attn | 默认后端 | `--flash-attn on` | 短上下文下 attention 占比小，但不是零 |
| 采样与流式 | vLLM 采样器 | llama.cpp 采样器 | 温度 0 降低随机性，仍不能视为同一实现 |

因此正确的句子是“在这组配置下，vLLM 的有效容量高于 llama.cpp”，而不是“vLLM 的调度器单独带来了三倍容量”。若要分离调度器贡献，需要固定权重格式、后端与上下文预算，再只换调度策略；本章没有做这个受控实验。

## 8.9 决策矩阵与边界

把测量结果压成选型建议时，先写清楚场景：

| 场景 | 建议 | 依据 |
|-|-|-|
| 单卡 24 GiB、Qwen3-4B、256/128、无共享前缀、SLO 800/100/15,000 | 选 vLLM，工程水位 C32 | C32 是最大已测通过点；C48 首次违约；goodput 10.986 req/s |
| 同上，但只要求 C≤8 | 两者都能过，vLLM 延迟余量更大 | vLLM C8 TTFT 206 ms；llama.cpp 778 ms，已贴近 800 ms |
| 需要 131K 上下文 | 本章配置不可行 | 真机 KV 分配 18.00 GiB 直接 OOM；需减 KV 精度、减上下文或换更大显存 |
| 需要最大化裸 tok/s、可接受违约 | 两者都会继续涨到 C64 | vLLM 1,835 tok/s；llama.cpp 553 tok/s；但 goodput 已不代表有效服务 |
| 需要共享长前缀、多轮复用 | 本章不能回答 | prefix cache 被关闭；SGLang/RadixAttention 场景未测 |

本章没有完成 SGLang、Ollama、MIGraphX 或 P/D 分离的横评，也不覆盖长上下文成功路径、共享前缀、多轮对话、多卡、多副本、功耗与温升。`llama.cpp` 在 C16 的边缘 FAIL 说明 800 ms 阈值附近的结论对样本波动敏感；`P99` 来自 200 条请求，不能当成生产 SLA 的保证。所有结论仅限 RX 7900 XTX、Qwen3-4B、ISL=256/OSL=128 与本章冻结的 SLO。

## 8.10 实验设计与复现

| 实验 | 目的 | 口径 | 产物 |
|-|-|-|-|
| 课程与论文核对 | 确认本章增量与引用边界 | 19 条来源逐一访问核实 | `code/part1-modeling/chapter8/course-research.md` |
| 显存账本 + 失败探针 | 先判静态可行性 | 第 3 章 KV 几何 + 真机启动日志 | `results/framework/llamacpp-memory-fail/probe.log`、fig8-5 |
| 模拟筛选 | 验证失败模式分类 | analytical backend；只采信静态显存 | `results/screen-qwen4b-7900xtx-v2/` |
| 主扫描 | 找最大已测通过点与首次违约点 | 200 prompts/点，低开销压测 | `results/framework/{vllm,llamacpp}/`、fig8-1/2 |
| 留出预测 | 检验低并发校准能否预测新并发 | C=1/8/32 拟合，C=16/48 测前冻结 | `prediction-holdout.json`、`holdout-analysis.json`、fig8-3 |
| 框架级 profile | 归因 dispatch、GPU busy 与 kernel 类别 | C=8、20 请求；trace 只做相对归因 | `profile-summary.json`、fig8-4 |
| 不等价项审计 | 防止把配置差异写成框架能力 | 逐项列出格式、精度、接口、上下文、KV、执行路径 | 本章 8.8 表 |

复现顺序是：先修好 TPOT 口径，再跑主扫描；然后只用 C=1/8/32 拟合、冻结预测、再测 C=16/48；profile 与失败探针最后做，且不参与 SLO 判定。脚本包括 `bench_framework.py`、`analyze_framework.py`、`predict_holdout.py`、`analyze_holdout.py`、`summarize_profiles.py` 与 `figs/make_figs.py`。原始请求行、原始 summary 与 trace 哈希均保留，正文数字以对应 JSON 为准。

## 本章小结

- 在 RX 7900 XTX、Qwen3-4B、ISL=256/OSL=128 与 800/100/15,000 ms SLO 下，vLLM 的最大已测通过点是 C32，llama.cpp 是 C8；vLLM 的首次违约在 C48，llama.cpp 在 C16。
- 裸吞吐会一直涨到 C64，但有效 goodput 不会：vLLM C64 只比 C32 高 3.1%，llama.cpp C64 反而从 2.500 跌到 0.822 req/s。容量判断必须同时看 P99、失败率与 goodput。
- 131K 上下文在 24 GiB 卡上先撞显存：KV 账本 18.00 GiB，真机分配 18,432 MiB 时 OOM。模拟筛选器的静态显存判定有效，48 并发的 66 秒延迟预测严重失真，已作为负结果保留。
- 测前冻结的 C=16/48 留出预测命中 4/4 个 SLO 判定，MAPE 为 TTFT 14.38%、TPOT 9.816%、E2EL 9.623%；但 llama.cpp C16 只超阈值 38.66 ms，是边缘 FAIL。
- C=8 profile 显示 vLLM 每输出 token 105.4 次 dispatch、GPU busy 87.73%，llama.cpp 是 194.5 次、70.77%；GEMM 都占约 89%，短上下文下 attention 只占 5%–6%。框架差异主要是执行路径、调度与 KV 预算的组合，不能归成单一算子。
- 本章完成的是 vLLM 与 llama.cpp 的真机横评；SGLang、Ollama、MIGraphX、共享前缀、长上下文成功路径与多卡部署仍未覆盖。

下一章把视角从单卡服务推向端侧与资源受限场景：当显存、功耗和离线约束同时收紧时，前面这些容量账本与延迟模型需要重新排序。

## 延伸阅读

- Harvard MLSysBook, *ML Frameworks*：<https://mlsysbook.ai/vol1/07_frameworks/07_frameworks.html>
- Harvard MLSysBook, *Benchmarking*：<https://mlsysbook.ai/vol1/12_benchmarking/12_benchmarking.html>
- Harvard MLSysBook, *Model Serving*：<https://mlsysbook.ai/vol1/13_model_serving/13_model_serving.html>
- Stanford CS336, *Inference*：<https://cs336.stanford.edu/>
- CMU 15-442/15-642, *LLM Serving*：<https://mlsyscourse.org/schedule>
- Orca, OSDI'22：<https://www.usenix.org/conference/osdi22/presentation/yu>
- PagedAttention / vLLM, SOSP'23：<https://arxiv.org/abs/2309.06180>
- SGLang, NeurIPS'24：<https://arxiv.org/abs/2312.07104>
- Sarathi-Serve, OSDI'24：<https://www.usenix.org/conference/osdi24/presentation/agrawal>
- DistServe, OSDI'24：<https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf>
- vLLM profiling：<https://docs.vllm.ai/en/latest/contributing/profiling/>
- llama.cpp server：<https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md>
- ROCprofiler-SDK `rocprofv3`：<https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html>
