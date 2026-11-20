# 第 1 章：18 点 nominal 实验报告

协议 `hello_mlsys_ch1_nominal_v1`；创建于 `2026-11-20T05:46:14.271Z`。

**解析预测完成；不代表课程的历史实测、先预测后测量纪律或 W7900D 硬件验收已完成。**

- 解析 nominal 下界；未执行 GPU、未拟合效率、hardware_validated=false。
- 本次 stored_at 是当前运行时间，不证明早于课程已有实测；未来测量须另行关联 prediction_id。
- 本协议使用 full-square attention；默认服务模拟器和既有 Profile 仍使用原口径。
- GEMM 激活及 elementwise 是 plan 中公开的流量假设，并非从课程近似时间反推的效率。
- W7900D 算力/带宽待确认；AI 示意不是该硬件的 Roofline 或时间预测。
- 本章时间不包含排队、launch、通信或容量/OOM 判定；带宽受限不等于显存装不下。

## 计数与瓶颈

![RX9070XT 双屋顶与全部18点组件](roofline-rx9070xt.svg)

图中保留全部54个 nominal 组件点；精确重合的 marker 会覆盖，编号与配置在右表，完整数值见 [components.csv](components.csv)。
每个组件先选矩阵/向量峰值，取 max(compute, memory)，step 再求和；全局 AI 不用于判定。

| 模型 | Phase | B | S / ctx | GEMM | Attention | Elementwise | Step (ms) | 聚合 token/s |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| Qwen3-0.6B | prefill | 1 | 128 | memory | memory | memory | 1.628570 | 78596.58 |
| Qwen3-0.6B | prefill | 1 | 512 | compute | compute | memory | 2.712782 | 188736.16 |
| Qwen3-0.6B | prefill | 1 | 2048 | compute | compute | memory | 14.551406 | 140742.41 |
| Qwen3-0.6B | decode | 1 | 1024 | memory | memory | memory | 2.046131 | 488.73 |
| Qwen3-0.6B | decode | 4 | 1024 | memory | memory | memory | 2.597171 | 1540.14 |
| Qwen3-0.6B | decode | 16 | 1024 | memory | memory | memory | 4.801331 | 3332.41 |
| Qwen3-1.7B | prefill | 1 | 128 | memory | memory | memory | 4.839834 | 26447.19 |
| Qwen3-1.7B | prefill | 1 | 512 | compute | compute | memory | 7.892416 | 64872.40 |
| Qwen3-1.7B | prefill | 1 | 2048 | compute | compute | memory | 35.269945 | 58066.44 |
| Qwen3-1.7B | decode | 1 | 1024 | memory | memory | memory | 5.560269 | 179.85 |
| Qwen3-1.7B | decode | 4 | 1024 | memory | memory | memory | 6.111846 | 654.47 |
| Qwen3-1.7B | decode | 16 | 1024 | memory | memory | memory | 8.318157 | 1923.50 |
| Qwen3-4B | prefill | 1 | 128 | memory | memory | memory | 12.091392 | 10586.04 |
| Qwen3-4B | prefill | 1 | 512 | compute | compute | memory | 20.167395 | 25387.51 |
| Qwen3-4B | prefill | 1 | 2048 | compute | compute | memory | 90.184586 | 22708.98 |
| Qwen3-4B | decode | 1 | 1024 | memory | memory | memory | 12.806106 | 78.09 |
| Qwen3-4B | decode | 4 | 1024 | memory | memory | memory | 13.515622 | 295.95 |
| Qwen3-4B | decode | 16 | 1024 | memory | memory | memory | 16.353690 | 978.37 |

Prefill token/s 是输入处理率；decode token/s 是生成速率，二者不可混作同一服务吞吐。

![Batching 曲线](batching.svg)

固定 ctx 后，batching 复用 GEMM 权重，但 KV 扫描仍随 B 增长。

![Prefill 曲线](prefill.svg)

固定 B=1；长序列的 full-square attention 计算项增长。

## W7900D：待确认参数

![W7900D 待确认，仅 AI 索引](roofline-w7900d-pending.svg)

参数未知，无法画有数值意义的该硬件屋顶线；此图只展示同一 18 点的 AI，纵轴不是吞吐。

## 手算与近似表对账

| 0.6B 配置 | 课程约数 ms | 本协议精确 ms | 差值 ms | 一位小数相同 |
| --- | ---: | ---: | ---: | --- |
| qwen3_0_6b_prefill_s128 | 1.5 | 1.628570 | +0.128570 | 否 |
| qwen3_0_6b_prefill_s512 | 2.7 | 2.712782 | +0.012782 | 是 |
| qwen3_0_6b_prefill_s2048 | 14.0 | 14.551406 | +0.551406 | 否 |
| qwen3_0_6b_decode_b1 | 2.1 | 2.046131 | -0.053869 | 否 |
| qwen3_0_6b_decode_b4 | 2.6 | 2.597171 | -0.002829 | 是 |
| qwen3_0_6b_decode_b16 | 4.9 | 4.801331 | -0.098669 | 否 |

这些差异属于公式、流量假设与近似表口径，**不标作硬件测量误差或校准通过**。

- 时间对账使用课程算例 B/C 的约数，未将其当作标定目标或测量数据。
- 0.6B 精确 P_linear=440401920、P_linear+P_head=595984384；课程写约440M/596M。
- prefill 激活流量系数与 elementwise 是明确假设；各组件求和可能无法复现近似表的单步舍入。
- prefill LM head=omitted；课程主项省略末 token head，任务表泛称包含 head。
- 不对 FlashAttention 统一乘1.13：原文13%是外部算子测例比值，非本模型冻结公式。
- 原文 F=4*L*q_heads*head_dim*S²、额外 D=4*L*q_heads*S²，故极限为 head_dim=128；不能同时得到0.5。

## Primitive 直觉练习

| 算子 | FLOPs | Bytes | AI | 判定 | t_compute ms | t_memory ms | t ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| matmul_4096_bf16 | 137438953472 | 100663296 | 1365.333333 | compute | 0.704815 | 0.157286 | 0.704815 |
| layernorm_2048_1024_bf16 | 8388608 | 8388608 | 1.000000 | memory | 0.000172 | 0.013107 | 0.013107 |
| attention_flash_s512_28_layers | 60129542144 | 176160768 | 341.333333 | compute | 0.308357 | 0.275251 | 0.308357 |
| attention_naive | 60129542144 | 645922816 | 93.090909 | memory | 0.308357 | 1.009254 | 1.009254 |

## 可复现底稿

- report data SHA256：`46d48034bf3455f710ebb30919e72e43f653259a7b67aa1646162990c4b4907f`
- plan SHA256：`d3378bc69b3af2fe25a47bc0a4cd652812f7d6fda586a700bc9e13993ce4923a`
- formulas SHA256：`1bad6c090de9fe4dbd6ca0de19bc4a51e3ca733a86e7bd85411087e3674504ff`
- 课程原文：[7df2dae493eb / 第 1 章](https://github.com/amdjiahangpan/hello-mlsys/blob/7df2dae493eb30c0f7e845b6d587b905288c0d7a/docs/part1-modeling/chapter1/index.md)；SHA256 `0c0c01a90d35b1b107cc88ccd2c48e776de141bb698356d4b8696baa401ec231`。
- [report.json](report.json) 保存全部公式、plan、逐点 prediction_id / stored_at、模型输入 hash。
- 以上图、表和 CSV 都从该 report 对象生成；未读取或重标课程已有测量为本次实验。
