# 第 6 章课程调研（模型并行策略与显存/算力拆解，模拟器验证）

调研完成：2026-11-16（subagent，报告 a28880cbf81d908f2）。
19 条 URL 全部在调研时逐条核实可达；正文延伸阅读只引下列清单内条目。
引用前若发现 URL 变动，以 web 复核为准（不复用旧链接）。

## 一、各课程覆盖与公式密度

### CS336（Stanford，spring 2025）—— 主对标
- 课程主页 cs336.stanford.edu/spring2025/（Lecture 7 "Parallelism basics" 为并行专讲）。
- lecture_01.py / lecture_02.py（raw 代码可运行）：L2 含"16B 参数 × 8×H100 →
  可放 40B"的现场算账（16 B/param 口径：权重+梯度+优化器态）。
- Lecture 7 讲义（2025 版 PDF，只读不可运行）：全库公式最密的一讲——
  - DP：每卡 ≈ 2×#params（训练态，bf16 权重 + fp32 优化器态简化）；
  - ZeRO 1/2/3：通信量级 2× / 2× / 3×（相对 DP）；
  - 8×A100（12 B/param BF16-Kahan 口径）最大模型：ZeRO0/1/2/3 =
    6.66B / 16B / 24.62B / 53.33B（真实系统对照表）；
  - TP 每层通信 ≈ 8b·s·h·(n−1)/n（n=并行度）；
  - 激活显存 5a·s/h + 10s·b·h 量级式（h=头数相关项）。
- assignment2-systems（raw PDF）：FSDP 完整实现作业 + §8 "communication vs
  compute inequality" 开放任务（带宽-only 无 α 时延项；gloo/CPU-only 行为验证
  先例——与我们本章 gloo 口径同思路）。

### CMU 11-868（Spring 2026，llmsystem.github.io）
- L14 分布式训练总览；L15 DDP；
- L16 模型并行：**气泡 (K−1)/(M+K−1)**、"M > 4×K 时气泡可忽略"经验阈值、
  **"TP 必须限制在节点内"结论的原话出处**、torch.distributed.pipelining 代码；
- L18 ZeRO：训练态 20N+激活 口径、LLaMA-3 8B→160GB、GPT-3 175B→3500GB、
  ZeRO 族参数化 4N+M·N → (4+M)·N/K、Z3 通信 3×。

### CMU 15-442/642（mlsyscourse.org 幻灯片）
- slide 08：DP + ZeRO（20GB/GPU @1B 参数 实例）；
- slide 09：模型并行 + 流水线——TP 通信量 O(C_out·C_in) vs PP 激活
  O(B·C_in)/O(B·C_out)、气泡 (p−1)/m、1F1B / interleaved (1/v)·(p−1)/m、
  Megatron 512-GPU 规模案例；
- slide 11：显存优化——激活 checkpointing O(N/K+K)。

### mlsysim（开源模拟器，tutorials/ + math/）
- tutorials/distributed.html：Llama-3.1-70B / 256×H100 算例——纯 DP 效率
  42.5%，最优 DP8+TP8+PP4（M=16）→ 57.8%（3D 组合实证）；
- math.html：**η_scale = T_compute / (T_compute + T_dp + T_tp + T_ep + T_bubble)**
  （"切分收益被通信吃掉"的判定公式，与我们本章判定条件同构）、
  气泡 (P−1)/(V·M+P−1)（V=virtual stage）、**M ≥ 19×(P−1) 时气泡 <5%**、
  ring/tree 选择 + crossover M_cross ≈ N·α·β（含 α 时延项！）；
  **口径瑕疵（正文须标注）**：其 4D 并行表把 TP 标为 point-to-point 通信，
  与 TP 实为集合通信（AR/allgather）的事实不符。

### MIT 6.5940（hanlab.mit.edu）
- 课程页 + 目录：仅概念级覆盖（并行列为阅读主题）；分布式专讲未公开 →
  深度不可核实，不作公式对标。

### GT CS8803（Spr 2024，web.archive.org 存档课表）
- Alpa + Megatron-LM 为阅读材料；论文课性质，无课程级公式。

## 二、气泡公式四变体辨析（正文必做，避免口径混用）

| 出处 | 公式 | 口径 |
|---|---|---|
| 我们 / CMU 11-868 L16 | (P−1)/(M+P−1) | 气泡 / 总时间 |
| mlsysim | (P−1)/(V·M+P−1) + M ≥ 19(P−1) | 气泡 / 总时间（virtual stage 计入） |
| CS336 L7 | (n−1)/n_micro | 气泡 / 计算时间 |
| CMU 15-442 slide 09 | (p−1)/m | 气泡 / microbatch 数 |

## 三、本章超越空间（调研结论）

- **没有任何课程给出 α/时延-based 的 TP/PP crossover 判定**（mlsysim 的
  M_cross≈N·α·β 是 ring vs tree 拓扑选择，不是 TP vs PP 方案选择）；
- **没有任何课程做消费级硬件（PCIe 消费卡）per-GPU 通信占比模拟**——
  全部对标数据中心 NVLink/IB；
- "TP 组限制在节点内"只有 CMU 11-868 给结论+经验阈值（M>4×K），
  无模拟器留出验证 → 我们用 32B holdout 补；
- 因此：本章模拟器（公式口径 × Ch5 拓扑档案 × α/β 实测档）在"判定条件
  可算 + 留出验证 + 消费级拓扑"三点上无课程竞品。

## 四、不可引清单（勿再尝试，调研已核实不成立/不存在）

- MIT 6.S056（= "Hack Yourself"，非 ML 系统课）；MIT 6.8850（查无此课）；
  MIT 6.S184 / 6.852 / 6.S965（CH5 已核：6.S965 = 6.5940 前身，均不作并行对标）；
- Berkeley CS294-262 / CS250（无分布式并行覆盖）；Stanford CS250（编码理论）；
- UW CSE 5712（查无此课）；GT CS8803 Spr 2026 reading list（SharePoint 未核实）；
- mlsysbook.ai Vol II 章节（preview 级，只可引 mlsysim 在线页面）；
- MIT-Efficient-AI 镜像的分布式讲义（未公开 → 深度不可核实）。

## 五、建议叙事顺序（正文骨架参考）

装不下（16/20/12 B-per-param 显存账本）→ DP/ZeRO 族（训练态 20N 口径）→
TP（层内、逐层 2×AR、GQA 切分）→ PP（层间、气泡四变体辨析）→
3D/4D 组合 + 真实系统（mlsysim 70B 算例）→ 切分收益被通信吃掉
（η_scale + CS336 A2 留白 → 我们补 α/β 实测判定）→ 真机 gloo 行为验证
（行为 ≠ 性能）→ 延伸阅读（19 URL）。
