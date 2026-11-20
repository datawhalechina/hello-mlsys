# 第 7 章课程调研（在线服务、排队与 SLA）

调研完成：2026-11-18（subagent 报告）。下列 URL 均在调研时逐条访问核实；
正文「延伸阅读」只从本清单取值。标注“不覆盖”表示公开材料中未核实到，
不是推断未公开内容。

## 一、覆盖矩阵

| 课程/材料 | 覆盖位置 | 深度 | 本章判断 |
|---|---|---|---|
| Stanford CS336 Sp25 | L10 `Inference` | 单请求推理、KV、continuous batching、PagedAttention 很强 | **不覆盖** Little/M/M/1/P99/goodput/容量实验 |
| CMU 15-442/15-642 Sp26 | Week 9–10 serving | 调度、PagedAttention、RadixAttention 很强 | **不覆盖** 正式排队论、SLA 容量实验 |
| CMU 11-868 Sp25 | 04/07 vLLM/LMCache；04/14 DistServe/SGLang | vLLM 内部、KV 抢占、goodput 最强 | **不覆盖** 公开课堂的 Little/M/M/1/P99 推导与 1→64 扫描 |
| Berkeley CS294-262 Fa25 | 09/22、09/26、10/31、11/03 | serving 论文阅读 | **不覆盖** 可核实的公式推导、P99 实验、benchmark 作业 |
| Harvard mlsysbook.ai | Vol I Ch13；Vol II Ch10 | Little、M/M/1、P99、TTFT/TPOT、goodput、KV 都有 | 最接近，但不是可执行 DES + 真机校准课 |
| MIT 6.S184 | — | diffusion/flow matching | **不可引**，不是 ML systems |
| MIT 6.5940 Fa24 | L13 `LLM Deployment Techniques` | TTFT/TPOT、KV、vLLM、continuous batching | **不覆盖** 在线容量规划与排队实验 |
| GT CS8803 Sp24 | 02/26–04/01 | paper reading | **不覆盖** queueing/SLA 实验 |
| UIUC CS498 MLSys Sp25 | 03/25–04/17 | SLA/TTFT/TPOT/continuous batching 机制 | **不覆盖** Little/M/M/1/P99/DES |

## 二、公式与现场算例（直接可引）

### 指标

- `E2E_i = TTFT_i + Σ ITL_i`（UIUC、MIT 6.5940）。
- `TPOT_i = (E2E_i − TTFT_i)/(N_out_i − 1)`，`N_out>1`；课程常用近似
  `E2E = TTFT + TPOT × N_out`，实验应保留逐请求 ITL 分布。
- Harvard `13.9.1` 给出一组显式 SLO 示例：TTFT < 500 ms、TPOT < 50 ms、
  吞吐 > 1000 tok/s。

### Little's Law

- `N = λW`。
- Harvard `13.4.1`：1000 QPS × 50 ms = 50 个 in-system 请求；GPU batch 最多
  32 个请求时，其余在 host queue 等待。这是 **batch size ≠ 在线用户数**
  的最佳公开来源。

### M/M/1 与 P99

- `E[T] = S/(1−ρ)`，`ρ = λS`，要求 `ρ<1`。
- Harvard：`P99 ≈ 4.6 × S/(1−ρ)`（仅指数响应假设）。
- 70% 利用率：mean `3.3S`，P99 `15.3S`。
- ResNet-50 教学算例：50 ms p99 预算、5000 QPS、batch-1 `3.5 ms` →
  `ρ≤0.68` → `5000/0.68 = 7402.9 req/s` → 单卡 283 req/s → 27 张，
  加 30% headroom 为 36 张。

### Batching 与 goodput

- `μ(B)=B/T(B)`，`throughput = min(λ, μ(B))`（Harvard `13.5`）。
- Harvard ResNet-50：`T(B)=3.50 ms + 0.033 ms × B`；平均 formation wait
  为 window 的一半。
- Continuous batching waste：`W = 1 − mean_output/max_output`；输出
  `[50,200,100,150]` → `1−125/200=37.5%`（Harvard `10.6.5`）。
- Goodput `G = (# 完成且满足全部 SLO 的请求数) / 时间窗口`。
  Harvard `10.3`：X 完成 1200 rps、70% 达标 → goodput 840；
  Y 完成 1000 rps、95% 达标 → goodput 950。
- CMU 11-868 DistServe slide：raw throughput 10 rps、goodput 3 rps。

### KV 容量与抢占

- `KV_bytes/token = 2 × L × H_kv × d_head × bytes_per_element`。
- `B_KV ≈ (VRAM − weights − workspace − graph_reserve) /
  (KV_bytes/token × S_avg)`。
- Harvard `13.9.5` Llama3-8B：`2×32×8×128×2 = 131,072 B/token ≈ 131 KB/token`；
  8192 token ≈ 1.07 GB；80 GB H100 约 65 条对话。
- Harvard `13.10.3` Llama3-8B AWQ 4-bit/H100：KV ≈ 0.131 MB/token；
  77.3 GB reserve ≈ 0.6M tokens，batch ≈ 469；prefill 10,000 tok/s ≈ 10 req/s。
- CMU 15-442：GPT-3 175B 单请求 KV 约 3 GB；静态 KV 只有 20–40% 真正存
  token 状态。
- CMU 11-868 vLLM slide：KV 利用率 20–40%；request-level preemption；
  `swap` 与 `recompute`；slide 明确倾向 recompute + FCFS。
- DistServe 论文：prefill 单机、FCFS、Poisson、`RD<1` 时
  `Avg_TTFT = D + R D²/[2(1−RD)]`；13B OPT/A100-80GB、512-in/64-out，
  7.4× 请求量或 12.6× 更紧 SLO，>90% attainment。

## 三、课程没做什么（我们的差异化）

在核实范围内，没有课程/教材同时完成：

1. **DES + 真实 vLLM 校准**：Harvard 有解析 queueing，DistServe 有 simulator；
   但没有“真实 vLLM 服务时间校准 DES，再做 holdout”的教学闭环。
2. **open-loop λ、closed-loop C、engine batch B 三概念统一**：Harvard 最接近，
   但没有把三种负载语义放进同一实验矩阵。
3. **1→64 并发扫描**：CS336 只比较 B=1/64/256 的解析值；15-442 用 batch=3
   做示意；没有课程公开做 vLLM `bench serve` 的 1→64 阶梯。
4. **SLA 水位、最大承载、goodput、失败/超时同表交付**。
5. **拐点留出验证**：没有课程公开“冻结校准集与 holdout → 预测拐点 → 未见负载验证”。
6. **消费级 AMD 单卡/ROCm**：公开数字基本是 A100/H100 或解析模型。
7. **Kernel/调度归因**：课程讲机制，但缺少把 P99 陡升拆成排队、prefill、
   decode、KV recompute、算力饱和的实测归因表。
8. **DES 驱动动画**：没有课程提供排队/continuous batching/KV 水位/SLA 突破动画。

## 四、可引 URL 清单

### Stanford CS336
- https://stanford-cs336.github.io/spring2025/ — 课程主页，L10 `Inference`。
- https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py — 可执行讲义源码。
- https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/var/traces/lecture_10.json — 讲义 trace。

### CMU
- https://mlsyscourse.org/schedule — 15-442/15-642 schedule。
- https://mlsyscourse.org/slides/14-LLM-serving-part1.pdf — continuous batching / PagedAttention / RadixAttention。
- https://mlsyscourse.org/slides/15-LLM-serving-part2.pdf — speculative decoding。
- https://llmsystem.github.io/llmsystem2025spring/ — 11-868 主页。
- https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf — vLLM/PagedAttention、KV 20–40%、preemption/recompute。
- https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-24-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf — TTFT/TPOT、goodput、DistServe。
- https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf — DistServe OSDI'24。
- https://arxiv.org/abs/2401.09670 — DistServe arXiv。

### Berkeley / Harvard / MIT / GT / UIUC
- https://ucb-cs294-262.github.io/fa25/ — Berkeley CS294-262 Fall 2025。
- https://mlsysbook.ai/vol1/13_model_serving/13_model_serving.html — Model Serving：Little、M/M/1、P99、batching、TTFT/TPOT、KV。
- https://mlsysbook.ai/vol2/10_inference/10_inference.html — Serving at Scale：SLO goodput、batching waste、preemption、KV budget（web edition，可能更新）。
- https://diffusion.csail.mit.edu/2026/index.html — 6.S184 正式内容；**不是 ML systems**。
- https://hanlab.mit.edu/courses/2024-fall-65940 — MIT 6.5940。
- https://web.archive.org/web/20241109140337/https://cs8803-spring2024.anand-iyer.com/schedule/ — GT CS8803 2024 paper list。
- https://minjiazhang.github.io/courses/cs-mlsys-498-2025spring.html — UIUC CS498 MLSys。
- https://minjiazhang.github.io/courses/sp25-resource/week8.1_inference_continusous_batching.pdf — TTFT/TPOT/E2E/continuous batching。
- https://arxiv.org/abs/2309.06180 — vLLM/PagedAttention。
- https://www.usenix.org/system/files/osdi22-yu.pdf — Orca。
- https://arxiv.org/abs/2308.16369 — Sarathi-Serve。

## 五、不可引 / 需谨慎

- MIT 6.S184 = diffusion/flow matching，**不可引为 serving 课**。
- CMU 11-868 不能说“vLLM 作者开设并主讲整门课”；Woosuk Kwon 是 guest lecturer。
- CMU 15-442 公开 schedule 未列出 serving 专项 lab，不能声称有。
- Berkeley CS294-262 的 Google Drive slides 需登录，具体内容未核实，不能引用。
- CS336 B=1/64/256 数字是解析模型，不是 H100 实测。
- Harvard ResNet-50 36 卡算例是 batch-1 M/M/1 教学算例，不是 LLM/vLLM 结果。

## 六、建议叙事顺序

1. 单请求时间线：TTFT/ITL/TPOT/E2E/失败超时口径（CS336 L10 + Harvard 13.9.1 + UIUC）。
2. 从单请求到系统：`λ`、`C`、`B` 三个不同量；Harvard 1000 QPS × 50 ms 算例直接打“batch ≠ 用户数”。
3. 排队与尾延迟：Little → M/M/1 → P99；Harvard 70% → 15.3×service、36 卡算例。
4. LLM 特有机制：static/continuous batching → PagedAttention → KV → preemption（CMU 15-442、11-868、Harvard 13.9.5）。
5. Goodput 与拐点：Harvard 840 vs 950、DistServe 10 vs 3 rps。
6. 本章 DES：到达分布 + ISL/OSL + 校准 prefill/decode + 动态 KV；1→64 输出 P99、SLA 水位、goodput、queue/exec 占比。
7. 真机：`vllm bench serve` 阶梯，导出 TTFT/TPOT/ITL 分位与 E2E；预测拐点对齐。
8. 误差归因与 holdout：排队、prefill、decode、KV block 抢占/recompute、调度、算力饱和；冻结校准集与未见 trace。
9. 动画与交付：DES 逐请求时间线驱动“到达→排队→continuous batching→KV 水位→SLA 突破”。
10. 结论口径：所有数字绑定模型、精度、ISL/OSL、arrival trace、GPU/ROCm/vLLM 版本；没有实测就写“待测”。

一句话：Harvard 给公式骨架，CS336/CMU/UIUC/MIT 给 serving 机制，
Berkeley/GT 给论文入口；但“DES + 真机校准 + 1→64 扫描 + 拐点 holdout +
消费级 AMD 单卡”这套闭环，在核实范围内没有课程做过。
