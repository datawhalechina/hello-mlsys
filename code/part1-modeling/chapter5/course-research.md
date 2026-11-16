# CH5 通信主题课程调研报告（通信成本模型 α–β / 集合通信原语 / Scale-in vs Scale-up / 节点内外互连）

调研方式：WebFetch/WebSearch + 官方课程页/syllabus/slides PDF（curl + pdftotext，关键 slide 渲染成 PNG 逐张读取），所有引用 URL 均于本次会话逐一 HTTP 200 核实。

---

## 一、逐课程覆盖与深度评估

### 1. Stanford CS336（Spr 2025）—— 主对标，工具导向 + 定性规则，**无 α–β 公式**

- **L7 "Parallelism basics"（Tatsu）**：定性 + 经验规则级。集合原语族（Reduce/AllReduce/AllGather/Broadcast/ReduceScatter）；"AllReduce = ReduceScatter + AllGather，bandwidth-limited regime 下这就是最优"；TPU 环面 vs GPU all-to-all（≤256）；naive DP "每 batch 传 2× #params"；ZeRO-1/2/3 通信成本 2/2/3 × #params；PP bubble = (n_stages−1)/n_micro；"Tensor parallel within a node (up to 8 GPUs) due to high-speed interconnects"；TP vs PP 每层通信量（bsh 点对点 vs 8bsh(n−1)/n all-reduce）；3D 并行 rules of thumb；TP=8 最优（64 卡 8×8 配置）；Narayanan 2021 缩放表；DeepSeek V3 / Llama3-405B / Yi / Gemma 2 真实配置。**没有任何 α–β 模型、没有 ring 时间公式、没有数值通信算例、没有 crossover 讨论。**
- **L8 可执行讲义（Percy，现场跑 4-GPU 节点）**：工具导向，是全课程库中唯一"真机测量通信带宽"的：`nvidia-smi topo -m`；互连具体数字——PCIe "v7.0, 16 lanes => 242 GB/s"（节点内经典路径）、跨节点 "Ethernet (~200 MB/s)"、"NVLink intra-node / NVSwitch inter-node"、"Each H100 has 18 NVLink 4.0 links, total 900 GB/s"、"HBM 3.9 TB/s"；NCCL 拓扑感知；gloo(CPU)/nccl(GPU) 后端；all-reduce / reduce-scatter / all-gather 语义演示；**现场 all-reduce 带宽测量**（world_size=4，100×1024² 元素 = 400MB）：
  ```python
  sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
  total_duration = world_size * duration
  bandwidth = sent_bytes / total_duration
  ```
  这个 2(P−1) 因子正是 ring all-reduce 的系统总发送量，除以 world_size×duration 后得到的就是 NCCL 约定的 per-rank bus bandwidth，与 ring 公式 2(P−1)/P 因子自洽（课程没把这层明说，只作为测量惯例给出）。reduce-scatter 用 (world_size−1) 因子。并给出 NVIDIA nccl-tests PERFORMANCE.md 与 stas00 all_reduce_bench.py 两个延伸阅读。
- **深度定位**：公式层面为零（不写 T=α+β·Data），但"硬件层级具体数字 + 真机 NCCL 测量"是本课程库中最强，且测量对象是数据中心 NV18 节点，**不是消费级 PCIe**。

### 2. CMU 11-868 LLM Systems（Spr 2026）—— 带宽-only 推导，**无 α**

- **L14 Distributed Training**：NCCL "supports various of interconnect technologies: **PCIe, NVLink, InfiniBand, IP sockets**"（全课程库中唯一列出 IP sockets 的）；AllReduce = ReduceScatter & AllGather；"NCCL uses rings"；逐帧 slide 推导 "Broadcast with unidirectional ring"：单步 t = N/B，总时间 (K−1)N/B；再 "break data into S messages"：每步 t = N/SB，总时间 (K−2+S)N/SB ≈ N/B。**纯带宽推导，无延迟项、无 crossover。**
- **L15 DDP**："Efficient Ring AllReduce (ScatterReduce+AllGather)"；小消息 batching 降延迟；gradient bucketing；async all-reduce；"DDP reduces latency by overlapping backward and AllReduce"。
- **L16 Model Parallelism**：GPipe / interleaved 1F1B；"GPUs **without high-speed interconnect**"（PP 适合慢链路）；TP 逐算子 all-reduce 必要性分析（Megatron column/row parallel）。
- **L17 MoE**：DeepSeek EP 的 "**hierarchical all-to-all** communication pattern: reduces the latency"（A100 上 EP 规模 vs kernel 延迟对比）——全课程库中唯一提 hierarchical all-to-all 的。
- **深度定位**：推导深度介于 CS336（无公式）与 UIUC（完整 α–β）之间——有逐步推导但只有 β 没有 α；互连只列名不参数化。

### 3. UIUC CS498 Machine Learning System（SP26 / SP25，Minjia Zhang & Hoa La）—— **公式级完全对标，符号体系与我们一致**

SP26 week3.2 与 SP25 week3.1 为同一 deck（均 50+ 页，标题 "Data Parallelism, Allreduce-based DP, Communication terminologies"），**2025 与 2026 两版均已渲染 slide 核实**：

- 符号定义（slide 逐字）：**"α: Round-trip time latency; β: Network bandwidth; γ: Compute overhead"**
- **T_AllReduce = T_ReduceScatter + T_AllGather = 2((P−1)α + (P−1)/P · Data/β) = 2(P−1)α + 2(P−1)/P · Data/β**（与我们章节公式逐项一致）
- T_AllGather = (P−1)α + (P−1)/P·Data/β；T_ReduceScatter 同；含计算时 +γ
- **Tree AllReduce：T = T_Reduce + T_Broadcast = 2(P·α + Data/β) + γ**
- 引用 Centauri（ASPLOS'24 Best Paper，communication partitioning 做 comm-compute overlap）
- ring 两阶段逐步图（4 rank，reduce-scatter 4 轮 + all-gather 4 轮逐帧）；ring vs tree 两种范式；"Collectives are implemented using many P2P; Collectives are much more expensive than P2P"；NCCL/RCCL/MSCCL；NCCL GPU-direct ring backend
- SP25 week3.1 deck：PS 历史、"Data Parallelism with AllReduce — Fast connections: **NVLink, NVSwitch**"
- **没有的**：无 α/β 数值代入、无 crossover/knee、无互连参数表、无测量校准；实验室硬件是 NCSA Delta（quad A100/A40，Slurm）。
- **深度定位**：唯一与我们章节同级的公式级教学（α/β/γ + ring 分解 + tree），止于公式，不做数值化和测量。

### 4. Harvard mlsysbook.ai / MLSys·im（CS249r "TinyTorch" 项目配套）—— **唯一给出完整 α–β 公式族 + crossover 公式 + 数值算例的来源**

- **mlsysim physics.communication API**（逐字核实）：符号约定 "Nonnegative message size M (bytes) / Participant count N / Positive unidirectional bandwidth B (bytes/second) / Nonnegative startup latency alpha (seconds)"；
  - P2P：**T = α + M/B**
  - Ring AllReduce：**T = 2(N−1)M/(NB) + 2(N−1)α**
  - ReduceScatter/AllGather：每 rank 移动 (N−1)/N 个消息当量，f = (N−1)/N
  - Tree reduce：**T = 2⌈log2 N⌉(M/B + α)**
  - **Hierarchical（节点内/跨节点分离）：T = 2[(g−1)M/(g·B_local) + (g−1)α_local] + T_ring(M/g, n, B_network, α_network)** —— 即 "scale-up 域用本地参数、scale-out 域用网络参数" 的参数化表达
  - **Crossover（延迟 vs 带宽主导）：M_cross = α·B**；**ring vs tree 交叉点：M_est = N·α·B/log2(N)**
  - All-to-All：T = (N−1)M/(NB) + (N−1)α
- **physics.networking API**：fat-tree / torus、bisection bandwidth、oversubscription、hop delay
- **Tutorial "Distributed Training: 3D Parallelism and Scaling Efficiency"**（数值算例，逐字核实）：**Llama-3.1-70B FP16，32 DGX H100 节点 × 8 = 256 H100，每 step 141.2 GB FP16 梯度**；DP all-reduce overhead：**100GbE (12.5 GB/s) → 3287.97 ms，效率 42.5%；IB HDR (25 GB/s) → 1917.43 ms，55.9%；IB NDR (50 GB/s) → 1233.37 ms，66.3%**；"uses intra-node bandwidth (NVLink)"（TP 走 NVLink）
- **empirical-calibration 页**（重要反证）：明确声明 mlsysim 的结果是 "analytical outputs, not hardware measurements"，且跨设备通信 "**require separate modeling or measurement**"——**没有任何课程给出从 nccl-tests 类测量反解 α/β 的校准流程**。
- 书本体（Vol II "Scaling ML Systems"，覆盖 collective communication）为 **Preview / "in active revision"**，建议只引 mlsysim 页面，不引书章节。
- **深度定位**：公式深度天花板（α–β 全原语 + 两个 crossover 公式 + 分层互连模型 + LLM 数值算例），但是软件/分析工具项目，**不是课堂教学**，且算例用数据中心互连（IB/100GbE），无消费级 PCIe。

### 5. CMU 15-442/15-642 "Machine Learning Systems"（Spr 2026，mlsyscourse.org，Zhihao Jia 等）—— 意外收获，per-worker 通信量对比表

- **Slide 08 "ML Parallelization (Data Parallelism and Zero Redundancy)"**：naive all-to-all 总量 N(N−1)M vs ring 2NM（每 worker 2M，**与 N 无关**）vs tree 2NM vs butterfly NM·logN；对比表（PS 2×N×M、naive N²×M、ring 2×N×M、tree 2×N×M、butterfly N×M×logN）；时间 = 数据量/带宽（ring "M/N × 2N / bandwidth"、tree "M × 2 × logN / bandwidth"）——**只有 β，无 α**
- **Slide 09 "Model and Pipeline Parallelism"**：bubble 分析、"reduce bubble time at the cost of increased communication"
- **深度定位**：带宽-only 的 per-worker 通信量比较，补充了 ring/tree/butterfly/naive 四种方案的横向对比（UIUC 只对比 ring vs tree）。

### 6. GT CS8803（Spr 2024，Anand Iyer）—— 论文阅读课，**不教 α–β 模型**

Wayback 存档 schedule（2024-11-09 抓取）逐条核实，与本主题相关的 week：
- Jan 31 "Training Approaches for Large Models"：FSDP（必读）、PaLM、Pathways
- Feb 5 "Automating Parallelization"：**Alpa（必读）、Megatron-LM（必读）**、GSPMD
- Feb 7 "Extreme Scale Training"：GShard、"Overlap Communication with Dependent Computation via Decomposition"（必读）、ZeRO-Infinity（必读）
- Mar 6 "Optimizations for Large Model Inference"：**FlexGen（必读，含单 GPU 下 GPU/CPU/SSD 三级带宽的 offloading 线性规划——与我们的 α–β 思想同构但是 inference 侧）**、FlashAttention、Efficiently Scaling Transformer Inference
- Mar 25 MoE week：DeepSpeed-MoE、MegaBlocks、Switch、Tutel、Janus
- 定位：通过论文间接暴露通信话题（Megatron 的 TP all-reduce、FlexGen 的带宽参数化），但课程本身零公式教学。**Spr 2026 的 reading list 在 SharePoint 访客表格里，未核实。**

### 7. 明确不覆盖 / 不适用的课程

| 课程 | 结论 |
|---|---|
| MIT 6.S184 | 非 ML systems 课（扩散模型方向），**不可引**（CH3 结论，维持） |
| MIT 6.852 (Distributed Algorithms, Lynch) | 通用分布式算法课，无 ML 通信建模，不适用 |
| MIT 6.S965 (TinyML/efficient DL) | 搜索中发现；边缘高效 ML，非 LLM 训练通信，不适用 |
| **MIT 其他** | 未找到公开覆盖本主题的 MIT ML systems 课 |
| Berkeley CS294-262 (FA25) | 通用计算机系统性能课，非 LLM 训练通信，不适用（CH3 已核实） |
| Berkeley CS294-192 (SP25) | "Deep Unsupervised Learning"（Abbeel/Beal），非 systems 课，不适用 |
| UW "CSE 5712" | **未找到此课程**（UW ML 课为 CSE 446/546），不引 |

---

## 二、公式符号体系与现场算例对照

| 来源 | 符号 | Ring AllReduce 推导 | 数值算例 | Crossover/knee |
|---|---|---|---|---|
| **我们 CH5** | α=延迟, β=带宽 | T = 2(P−1)α + 2(P−1)/P·Data/β | 计划：消费级 PCIe 实测 | 计划：定量 knee |
| **UIUC CS498** | α=RTT latency, β=bandwidth, γ=compute | **完全一致**：2(P−1)α + 2(P−1)/P·Data/β；tree: 2(Pα + Data/β)+γ | **无**（纯符号） | **无** |
| **mlsysim** | M=bytes, N=participants, B=B/s, α=s | T = 2(N−1)M/(NB) + 2(N−1)α；tree: 2⌈log2N⌉(M/B+α) | **有**：Llama-3.1-70B/256 H100/141.2GB，100GbE/IB HDR/IB NDR 三档 3287.97/1917.43/1233.37 ms（42.5%/55.9%/66.3%） | **有**：M_cross = αB；ring vs tree M_est = NαB/log2N |
| **CMU 11-868** | N=bytes, B=bandwidth（无 α） | 逐步：每步 t=N/B → 总 (K−1)N/B → 切 S 片后 ≈N/B | 无 | 无 |
| **CMU 15-442** | M=bytes, N=workers, bandwidth（无 α） | per-worker 通信量：ring 2M（与 N 无关）/ tree 2M / butterfly M·logN / naive N·M | 无 | 无 |
| **CS336 L7/L8** | 无符号体系 | 无公式；L8 测量惯例 sent = 2(P−1)×size，bandwidth = sent/(N×t)（NCCL bus bandwidth 惯例） | **有**：4 GPU NV18 节点现场 all-reduce 100M 元素带宽（数据在 lecture trace 里，源码给方法） | 无 |

关键判断：**我们的 T = 2(P−1)α + 2(P−1)/P·β·Data 与 UIUC（教学场景）和 mlsysim（软件场景）逐项等价**，符号习惯（α/β vs α/B）两种写法课程里都有先例，无需独创。crossover 公式只有 mlsysim 有，且它没有把 knee 算到具体 GPU 上——"消费级互连上的定量 knee"是空白。

## 三、"Scale-in / Scale-up" 词本身的用法

**没有任何课程使用字面 "scale-up / scale-in" 这对术语**（全部文本 grep 确认；CMU 11-868 仅一处动词用法 "scale up to larger models"）。课程实际用语：

- CS336 L7（slide 双栏）："**Intra-node parallelism** via high-speed interconnects" / "**High-speed inter-node parallelism**"
- CS336 L8："Within a node: NVLink / Across nodes: NVSwitch"，并给量级：PCIe 242 GB/s vs 跨节点 Ethernet **~200 MB/s**（≈1000× 差距，课程未明说比值）；NVLink H100 900 GB/s vs HBM 3.9 TB/s
- CMU 11-868：NCCL 支持 "PCIe, NVLink, InfiniBand, IP sockets"（只列名，无量级）
- mlsysim：参数化表达——B_local/α_local（节点内）vs B_network/α_network（节点间）+ 100GbE/IB HDR/IB NDR 三档表（12.5/25/50 GB/s）
- UIUC："Fast connections: NVLink, NVSwitch"

结论：我们章节用 "scale-up（节点内互连）vs scale-out（跨节点互连）" 并配量级差，课程先例是 intra-node/inter-node 用语，**不冲突，且"scale-up/scale-out + 千倍量级差 + 四层参数 profile"的打包表述没有现成课程竞品**。xGMI 和 RoCEv2 在**所有已查课程中零出现**（InfiniBand 仅 CMU 11-868 列名 + mlsysim 参数化；Ethernet 仅 CS336 L8 给 ~200 MB/s）。

## 四、课程明确没做的（我们的差异化空间）

1. **消费级 PCIe 路径的 α–β 刻画**：CS336 只引 PCIe Gen5 x16 = 242 GB/s 理论值（数据中心语境），无单卡/双卡消费级硬件的任何测量；2-hop（GPU→CPU→GPU）与 P2P 的差异无课程讨论。
2. **单 GPU / 单进程行为验证**：所有多卡演示都要求 ≥2–4 GPU（CS336 L8 要 4 卡）；1-device 下 collective 的 no-op 语义、gloo CPU 后端行为无课程覆盖。
3. **从测量反解 α 与 β**：CS336 L8 只测聚合带宽（且是 4 卡 NVLink 节点）；mlsysim 明确把跨设备通信推给 "separate modeling or measurement"；**没有任何课程扫消息大小画 log-log 带宽曲线、标出 knee、从两段直线拟合 α/β**。
4. **定量 crossover**：只有 mlsym 给 M_cross = αB 公式，没算过任何具体 GPU 对的 knee 数值。
5. **xGMI / RoCEv2**：零覆盖（AMD 节点内互连与以太网 RDMA 全缺）。
6. **TP=2/4 单层通信占比数值**：CS336 L7 只给 8 卡情形 "8bsh(n−1)/n per layer" 定性式，无课程对具体 GPU 算 TP=2/4 的 comm/flops 比。
7. **双 GPU PCIe 2-hop α–β 校准实验**：最接近的竞品是 CS336 L8 的 in-class demo（数据中心硬件、只测带宽不拟合模型、不是学生作业）——我们把它作为进阶实验没有任何课程对标。

## 五、可引用 URL 清单（本次会话全部 200 核实）

**Stanford CS336 Spr 2025**
1. `https://stanford-cs336.github.io/spring2025/` — 课程主页（讲义表）
2. `https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf` — L7 幻灯片（定性并行原语 + 3D 规则）
3. `https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_08.py` — L8 可执行讲义源码（互连数字 + NCCL all-reduce 带宽测量）
4. `https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_08.json` — L8 现场运行 trace

**CMU 11-868 Spr 2026**
5. `https://llmsystem.github.io/llmsystem2026spring/docs/Syllabus` — 课程表（含全部讲义 PDF 链接）
6. `https://llmsystem.github.io/llmsystem2026spring/assets/files/llmsys-14-distributed-training-b27c3d1dc185e680c6f5cc924e9ec9d7.pdf` — ring broadcast 逐步推导（带宽-only）+ NCCL 互连列表
7. `https://llmsystem.github.io/llmsystem2026spring/assets/files/llmsys-15-ddp-165dbe3873fac21eb8b339e64bcfee28.pdf` — ring all-reduce + DDP bucketing/overlap
8. `https://llmsystem.github.io/llmsystem2026spring/assets/files/llmsys-16-model-parallel-83b41612547620ee0e172caa1ee448ed.pdf` — GPipe/interleaved + TP all-reduce 分析
9. `https://llmsystem.github.io/llmsystem2026spring/assets/files/llmsys-17-MoE-3aa3125f9ccdd4bb7109ef077fbe9260.pdf` — hierarchical all-to-all（DeepSeek EP）

**UIUC CS498（Minjia Zhang / Hoa La）**
10. `https://minjiazhang.github.io/courses/26sp-schedule.html` — SP26 课程表
11. `https://minjiazhang.github.io/courses/sp26-resource/week3.2_training_dp_allreduce.pdf` — **α/β/γ 定义 + ring/tree 全公式 deck**
12. `https://minjiazhang.github.io/courses/25sp-schedule.html` — SP25 课程表
13. `https://minjiazhang.github.io/courses/sp25-resource/week3.1_training.pdf` — 2025 版同 deck（slide 34–35 渲染核实，公式相同）

**Harvard mlsysbook.ai / MLSys·im**
14. `https://mlsysbook.ai/mlsysim/` — mlsysim 入口
15. `https://mlsysbook.ai/mlsysim/api/physics.communication.html` — **完整 α–β 公式族 + M_cross = αB + ring/tree 交叉 + hierarchical 模型**
16. `https://mlsysbook.ai/mlsysim/api/physics.networking.html` — fat-tree/torus、bisection、oversubscription
17. `https://mlsysbook.ai/mlsysim/tutorials/distributed.html` — **Llama-3.1-70B/256-H100 数值算例 + 100GbE/IB 对比**
18. `https://mlsysbook.ai/mlsysim/tutorials/06_scaling_1000_gpus.html` — 千卡扩展 tutorial（仅核实可达，未读正文）
19. `https://mlsysbook.ai/mlsysim/empirical-calibration.html` — 明确声明跨设备通信需单独建模/测量（反证用）
20. `https://github.com/harvard-edge/cs249r_book` — 书仓库（Vol II preview，勿直接引书章节）

**CMU 15-442 Spr 2026（mlsyscourse.org）**
21. `https://mlsyscourse.org/` — 课程主页
22. `https://mlsyscourse.org/slides/08-ML-parallelization-part1.pdf` — ring/tree/butterfly/naive per-worker 通信量对比
23. `https://mlsyscourse.org/slides/09-ML-parallelization-part2.pdf` — 模型/流水并行 + bubble

**GT CS8803 Spr 2024（存档）**
24. `https://web.archive.org/web/20241109140337/https://cs8803-spring2024.anand-iyer.com/schedule/` — 2024 论文阅读列表（FSDP/Alpa/Megatron/FlexGen 必读）

**课程内转引的第三方（均 200）**
25. `https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md` — NCCL 测试与带宽计算惯例（CS336 L8 所引）
26. `https://siboehm.com/articles/22/data-parallel-training` — DDP 数据依赖图经典讲解（UIUC slide 所引）

**不可引清单**：MIT 6.S184（非 ML systems 课）、MIT 6.852（通用分布式算法）、MIT 6.S965（TinyML/边缘）、Berkeley CS294-262（通用系统性能）、Berkeley CS294-192（无监督学习）、UW "CSE 5712"（查无此课）、GT CS8803 **Spr 2026 reading list**（仅 SharePoint 访客链接，**未核实**）、mlsysbook.ai Vol II 书章节（preview/in revision，只引 mlsysim 页面）。

## 六、建议叙事顺序（对标 CS336"公式 → 具体数值 → 装不下/慢在哪 → 手段"）

1. **单卡为什么不够**（compute/memory 双上限，CS336 L7 开篇 + UIUC week3.1 的 GPT-3 2800GB 表可引）→ 引出多卡通信问题
2. **一阶模型 T ≈ α + β×Data**：先定义 α/β（借 UIUC 的 α=RTT latency、β=bandwidth 习惯，注明 mlsysim 用 α/B 记法同源），给出 P2P 基线 T = α + M/B（mlsysim）——这一步课程里有现成对标，我们加的是消费级硬件语境
3. **集合原语族**：broadcast/scatter/gather/reduce/all-gather/reduce-scatter/all-reduce/all-to-all（CS336 L7 + mlsysim API 双来源；UIUC 的 "collectives 由 P2P 拼出、比 P2P 贵" 作直觉锚点）
4. **Ring all-reduce 推导**：T = 2(P−1)α + 2(P−1)/P·Data/β（UIUC 教学版；CMU 11-868 的 (K−1)N/B → ≈N/B 作"带宽极限"对照；CMU 15-442 的 per-worker 2M 表作"通信量与 P 无关"的旁证）；tree 备选 2(Pα + Data/β)
5. **Regime 分析：延迟主导 vs 带宽主导**：M_cross = αB（mlsysim，唯一公式来源，明确标注"课程中仅此一例"）→ 落到我们的消费级互连上算 knee = 本章节实测层，课程无竞品
6. **具体数值：互连层级**：先引 CS336 L8 数据中心数字（PCIe 242 GB/s / Eth ~200 MB/s / NVLink 900 GB/s / HBM 3.9 TB/s）建立量级感，再引 mlsysim 的 100GbE/IB HDR/IB NDR 表，然后给出**我们的四层 profile（PCIe 4.0 x16 / xGMI / 100G RoCEv2 / NVLink 300 GB/s）并明示 xGMI 与 RoCEv2 无课程先例**
7. **数值算例**：mlsysim 的 Llama-3.1-70B/256-H100（141.2 GB、三档 fabric、42.5%–66.3% 效率）做"大模型 + 慢网络"锚点；TP=2/4 单层通信占比做"单节点 + 快网络"锚点（CS336 L7 的 TP 通信式 + 我们补数值）
8. **测一测**：CS336 L8 的 NCCL all-reduce 带宽测量方法（bus bandwidth 惯例 + nccl-tests 链接）作为方法学对标 → 我们的单 GPU gloo 行为验证 + PCIe 路径校准（附双 GPU 2-hop 进阶实验）——明示这是课程未做的部分
9. **装不下 / 慢在哪 + 手段**：DP 2×#params（CS336 L7）、bubble (n−1)/m（CS336 L7）、TP per-layer all-reduce（CS336 L7 + CMU 11-868 L16）→ 手段：micro-batch（CS336 L8）、bucketing+overlap（CMU 11-868 L15）、hierarchical all-to-all（CMU 11-868 L17）、ZeRO 2/2/3×（CS336 L7）、3D 并行 rules of thumb（CS336 L7）——收尾呼应 CS336 的 "no single solution"。

一句话定位：公式层我们完全复刻 UIUC（α/β/γ + ring 分解），测量层复刻 CS336 L8（NCCL 带宽测量）但把硬件换成消费级 PCIe，crossover 与分层互连公式引 mlsysim 并注明其局限（纯分析、无 α/β 校准流程）——这三者拼起来在课程库里没有完整先例，差异化点即"消费级互连上的定量 α–β"。
