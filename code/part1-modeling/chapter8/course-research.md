# 第 8 章课程调研（推理框架、Profile 与系统取舍）

调研完成：2026-11-23。下列 URL 均在调研时访问核实；正文「延伸阅读」
只从本清单取值。标注“不覆盖”表示公开材料中未核实到，不是推断未公开内容。

## 一、覆盖矩阵

| 课程 / 材料 | 覆盖位置 | 深度 | 本章判断 |
|---|---|---|---|
| Harvard MLSysBook Vol I Ch7 | `ML Frameworks` | eager dispatch、静态图、graph capture、kernel fusion 很强 | 解释“同模型同卡为何引擎不同”；**不覆盖** Qwen3/ROCm 真机横评 |
| Harvard MLSysBook Vol I Ch12 | `Benchmarking` | benchmark–production gap、微/宏/端到端粒度、推理指标 | 本章测量契约与不等价项的直接依据 |
| Harvard MLSysBook Vol I Ch13 | `Model Serving` | request lifecycle、batching、runtime selection、KV 预算 | 把框架差异落到时延、吞吐与容量约束 |
| Stanford CS336 Sp26 | L10 `Inference` | KV cache、continuous batching、PagedAttention、推理性能 | 机制讲解充分；**不覆盖**多框架真机横评 |
| CMU 15-442/15-642 Sp26 | Week 9–10 `LLM Serving` | continuous batching、PagedAttention、RadixAttention、speculative decoding | 课程坐标与机制核对；**不覆盖**本章的框架筛选实验 |
| Orca (OSDI'22) | 全文 | iteration-level scheduling / continuous batching 的源头 | 解释调度收益与调度开销 |
| vLLM / PagedAttention (SOSP'23) | 全文 | KV 分页、碎片、共享与吞吐 | 解释 vLLM 的显存侧优势与配置约束 |
| SGLang (NeurIPS'24) | 全文 | RadixAttention、前缀复用、结构化程序执行 | 解释“工作负载有共享前缀时”框架排序可能反转 |
| Sarathi-Serve (OSDI'24) | 全文 | chunked prefill、stall-free batching | 解释 prefill/decode 干扰与尾延迟 |
| DistServe (OSDI'24) | 全文 | goodput、P/D 干扰、资源解耦 | 解释 raw throughput 与 SLO goodput 不能混用 |
| vLLM 文档 | optimization / benchmark / metrics / profiling | 配置旋钮、内置压测、Prometheus 指标、profiler 入口 | 本章 vLLM 侧的可复现口径 |
| llama.cpp server 文档 | server README / OpenAI API | slots、continuous batching、`ignore_eos`、timings、metrics | 本章 llama.cpp 侧的可复现口径 |
| ROCprofiler-SDK 文档 | `rocprofv3` trace / PMC | kernel trace、marker、硬件计数器与稳定性能档 | 本章 kernel 时间归因与 PMC 的使用边界 |

## 二、可直接支撑本章的结论

### 2.1 框架不是“同一算子的不同壳”

- Harvard Ch7 把框架问题拆成执行、微分、抽象三层；其中 eager 执行的逐算子
  dispatch 与静态图/Graph capture 的差异，直接对应本章观察到的
  **kernel 数、GPU busy 与 decode 尾延迟**。
- kernel fusion 同时减少访存流量和 launch 次数；因此“框架更快”不能只说成
  GEMM 更快，必须拆成数据通路、融合、调度与 KV 管理四类原因。
- 这要求横评记录**模型格式、权重精度、量化、attention backend、graph/compile、
  调度器、KV block 与采样参数**，否则差异无法归因。

### 2.2 批处理与 KV 管理共同决定容量

- Orca 的 iteration-level scheduling 使完成的序列可以立刻让位，避免静态 batch
  等待最慢请求。
- PagedAttention 把 KV cache 分页，降低外部碎片与重复复制；vLLM 的
  `gpu-memory-utilization`、`max-model-len`、`max-num-seqs` 因此不是互不相干的
  旋钮，而是共同决定 KV 池大小与可接纳并发。
- SGLang 的 RadixAttention 说明：当请求共享长前缀时，缓存复用会改变框架排序；
  本章必须关闭或显式记录 prefix cache，避免把负载特性误当成框架能力。

### 2.3 Prefill 与 Decode 的干扰必须进入解释

- Sarathi-Serve 指出 prefill 是计算密集的大块工作，decode 是逐 token 的低利用率
  工作；二者混批会造成 decode stall。
- DistServe 用 goodput 而不是裸吞吐评价服务，因为超过 SLO 的 token 不应继续
  计入有效产出。
- 因此本章同时报告 P50/P99 TTFT、TPOT、E2EL、请求吞吐、token 吞吐与
  **SLO goodput**；不能只报一个“最大 tok/s”。

### 2.4 Profile 是归因层，不是性能结论层

- ROCprofiler-SDK 支持 `rocprofv3 --kernel-trace` 与 `--pmc`；PMC 需要稳定性能档，
  且 trace 会引入可观测扰动。
- vLLM 提供 Prometheus 指标和 profiler 入口；llama.cpp server 提供
  `/metrics`、`timings` 与 `timings_per_token`。
- 因此本章把 **低干扰的端到端延迟** 与 **同一 trace 内的 kernel 相对归因**
  分开报告；profile 墙钟不能替代正式压测数字。

## 三、本章差异化

在核实范围内，课程和论文分别覆盖机制或单框架使用，但没有同时给出：

1. 同一张 AMD 消费级 GPU、同一 Qwen3-4B、同一 ISL/OSL 下的 vLLM 与
   llama.cpp 闭环横评；
2. 同一套 P50/P99、吞吐、SLO goodput 与显存证据表；
3. 显式“不等价项账本”：BF16 vs F16 GGUF、OpenAI 接口 vs 原生 server、
   graph/compile vs 融合 kernel、prefix cache 开关；
4. 把模拟筛选器的失败模式（显存先爆 / 延迟先爆）与真机观测分开验证；
5. 用 Ch1 Roofline、Ch3 KV 账本、Ch7 排队与 Ch8 kernel profile 共同解释
   框架差异，而不是只贴一张 benchmark 表。

本章的增量是第 5 点，而不是重新发明 continuous batching 或 PagedAttention。

## 四、来源清单（已核验）

### 课程

- https://mlsysbook.ai/vol1/07_frameworks/07_frameworks.html — ML Frameworks。
- https://mlsysbook.ai/vol1/12_benchmarking/12_benchmarking.html — Benchmarking。
- https://mlsysbook.ai/vol1/13_model_serving/13_model_serving.html — Model Serving。
- https://cs336.stanford.edu/ — Stanford CS336 Spring 2026；L10 `Inference`。
- https://mlsyscourse.org/schedule — CMU 15-442/15-642 Spring 2026；Week 9–10 LLM Serving。
- https://mlsyscourse.org/slides/14-LLM-serving-part1.pdf — continuous batching、PagedAttention、RadixAttention。
- https://mlsyscourse.org/slides/15-LLM-serving-part2.pdf — speculative decoding。

### 论文

- https://www.usenix.org/conference/osdi22/presentation/yu — Orca: A Distributed Serving System for Transformer-Based Generative Models。
- https://www.usenix.org/system/files/osdi22-yu.pdf — Orca 论文 PDF。
- https://arxiv.org/abs/2309.06180 — Efficient Memory Management for Large Language Model Serving with PagedAttention。
- https://doi.org/10.1145/3600006.3613165 — PagedAttention / vLLM，SOSP'23。
- https://arxiv.org/abs/2312.07104 — SGLang: Efficient Execution of Structured Language Model Programs。
- https://www.usenix.org/conference/osdi24/presentation/agrawal — Sarathi-Serve，OSDI'24。
- https://arxiv.org/abs/2403.02310 — Sarathi-Serve arXiv 版本。
- https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf — DistServe，OSDI'24。
- https://arxiv.org/abs/2401.09670 — DistServe arXiv 版本。

### 框架与工具文档

- https://docs.vllm.ai/en/latest/configuration/optimization/ — Optimization and Tuning。
- https://docs.vllm.ai/en/latest/benchmarking/cli/ — Benchmark CLI。
- https://docs.vllm.ai/en/latest/usage/metrics/ — Production Metrics。
- https://docs.vllm.ai/en/latest/contributing/profiling/ — Profiling vLLM。
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md — llama.cpp server API、slots、continuous batching、metrics 与 timings。
- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html — rocprofv3 trace、PMC 与性能档说明。

## 五、不能外推的边界

- Harvard、CS336、CMU 的机制讲解不等于本章硬件上的框架排名。
- Orca/PagedAttention/SGLang/Sarathi-Serve/DistServe 的论文数字来自各自模型、
  硬件与负载；不能直接当成 Qwen3-4B + RX 7900 XTX 的预期值。
- vLLM 的 CUDA/HIP Graph、`torch.compile`、chunked prefill、prefix cache 都是
  配置变量；论文或文档中的默认值不保证与本实验相同。
- llama.cpp 与 vLLM 不是“只换调度器”的等价实现：GGUF 权重格式、采样路径、
  融合 kernel、OpenAI 兼容层与 batch 管理都不同。
- 本章只覆盖单卡、短上下文、固定长度负载；不能把结论外推到长上下文、
  多轮共享前缀、P/D 分离或多副本部署。
