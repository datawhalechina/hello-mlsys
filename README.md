<div align="center">
  <img src="./docs/public/images/head.png" alt="Hello MLSys：连接设备，搭建平台，让每一份 AI 算力更高效" width="100%">
  <p><strong>连接设备，搭建平台，让每一份 AI 算力更高效 🚀</strong></p>
</div>

<div align="center">

*单机部署 · 服务评测 · 单机优化 · 多机服务 · 进阶协同*

<p align="center">
  <a href="./README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
</p>

</div>

<div align="center">

<a href="https://datawhalechina.github.io/hello-mlsys/"><img alt="在线阅读" src="https://img.shields.io/badge/%F0%9F%9A%80_%E5%9C%A8%E7%BA%BF%E9%98%85%E8%AF%BB-%E5%AE%8C%E6%95%B4%E6%95%99%E7%A8%8B-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://datawhalechina.github.io/hello-mlsys/curriculum/"><img alt="全书目录" src="https://img.shields.io/badge/%F0%9F%93%9A_%E5%85%A8%E4%B9%A6%E7%9B%AE%E5%BD%95-%E7%AB%A0%E8%8A%82%E5%AF%BC%E8%88%AA-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://github.com/datawhalechina/hello-rocm"><img alt="hello-rocm AI 入门实践" src="https://img.shields.io/badge/%F0%9F%9B%A0%EF%B8%8F_hello--rocm-AI_%E5%85%A5%E9%97%A8%E5%AE%9E%E8%B7%B5-1a73e8?logo=github&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://github.com/datawhalechina/hello-gpu"><img alt="hello-gpu GPU 计算优化" src="https://img.shields.io/badge/%E2%9A%A1_hello--gpu-GPU_%E8%AE%A1%E7%AE%97%E4%BC%98%E5%8C%96-1a73e8?logo=github&logoColor=white&labelColor=1a1a1a" height="28"></a>

</div>

<p align="center">
  <img src="./docs/public/aup-logo.png" alt="AMD University Program" height="44">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./docs/public/rocm-logo.png" alt="AMD ROCm" height="44">
</p>

> [!CAUTION]
> ⚠️ Alpha 内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提 Issue 反馈问题或建议。

&emsp;&emsp;🎓 **和新加坡国立大学（NUS）的同学一起学 MLSys！** 本项目由 NUS 的同学参与共建，也会和更多 NUS 同学一起边学边做。欢迎你加入我们，在 AMD 设备上从一台机器的模型服务，走到多人共享的 LLM 服务平台。

&emsp;&emsp;模型能跑起来之后，下一步往往是把它变成大家都能用的服务：一台机器能撑住多少并发？首 Token 延迟和生成速度能否满足应用？精度与量化、批处理、KV Cache 和框架配置该怎么选？实验室里的几台 AMD 设备，又该怎样连成一个多人共享、稳定可用的 LLM 服务？

&emsp;&emsp;然而，**能把模型跑起来，并不等于能把服务做好**。这些问题牵涉模型、硬件、框架配置、负载与服务目标之间的取舍：只靠反复试错，既慢又贵；只看框架文档，又说不清某个配置为什么有效、换一台设备是否依然成立。

&emsp;&emsp;**hello-mlsys** 正是为这条路径而生。本教程基于 AMD 硬件，以 **vLLM、llama.cpp** 等开源推理框架为实践载体（vLLM 在 Linux 上走通主线，llama.cpp 做对照实验并覆盖 Ryzen AI MAX+ 395 等端侧设备），带你从**单机部署**走到**多机 LLM 服务**：通过性能评测、精度与量化选择、框架调优和资源分配，搭建自己的高效模型服务平台。全书重点回答一个问题：

> **给定设备、任务和服务目标，怎样选择模型精度、推理框架与资源配置，让整体服务更高效？**

&emsp;&emsp;为了让每个选择都有依据，本教程坚持**先预测、再实测**：动手调参之前，先用模拟器 **[mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator)** 估算性能、显存与容量边界，再挑选少量有信息价值的配置上 AMD 真机验证，解释偏差、校准模型，最后做出配置决策。

&emsp;&emsp;**本教程不是推理框架的使用手册，而是希望帮助读者建立 LLM 服务系统的工程能力：能部署、会评测、懂优化、可扩展。任何人都可以提出 issue 或是提交 PR，共同构建维护这个项目。**

### AMD Hello 系列

&emsp;&emsp;从 AI 入门与应用实践，到 GPU 计算优化，再到 LLM 服务平台构建，为行业应用、科研实践与人才培养提供共同基础。三个项目可以独立学习，也能通过同一个模型串联实践：在 hello-rocm 中训练、微调和使用模型，在 hello-gpu 中理解并优化计算，在 hello-mlsys 中将模型部署为高效的共享服务。

| 项目 | 标语 | 核心定位 |
| ---- | ---- | ---- |
| [**hello-rocm**](https://github.com/datawhalechina/hello-rocm) | 每个 AI 小白，从零开始，动手实践的第一站。 | 基于 AMD 平台的 AI 与 Agentic AI 入门实践手册 |
| [**hello-gpu**](https://github.com/datawhalechina/hello-gpu) | 看懂 GPU，亲手优化每一次计算。 | 从自己训练的小模型数据类型、GEMM 优化出发，学习 GPU 编程与 Agent 辅助优化 |
| **hello-mlsys**（本项目） | 连接设备，搭建平台，让每一份 AI 算力更高效。 | 以 vLLM 等开源推理框架为案例，学习 LLM 服务部署、性能优化与多机服务构建 |

&emsp;&emsp;环境安装、AI 入门与模型微调请参考 hello-rocm，GPU 编程与算子优化请参考 hello-gpu，本教程不重复这两部分内容。

> &emsp;&emsp;***学习建议：章节顺序就是推荐的学习顺序。建议先读[第 0 章 协议与纸面演练](https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/)，统一 token、GB/GiB、延迟等测量口径；第 1、2 章建立 Roofline 与 Profile 的性能分析和校准方法；第 3–8 章依次学习显存、精度与量化、通信、并行、服务评测与推理框架取舍；最后在第 2 篇的综合实战中，把这些方法用于端侧、单机与集群设计。手边暂时没有 AMD 显卡，第 1 章的 Roofline 预测与第 3 章的显存账本都可以纯 CPU 完成。***

## 项目意义

&emsp;&emsp;什么是 MLSys？

> MLSys（Machine Learning Systems）研究如何把机器学习模型高效、可靠、经济地运行在真实硬件与真实负载上。对 LLM 来说，它关心的不只是单个请求跑多快，而是一套服务在给定设备与服务目标下，能稳定交付多少有效工作。

&emsp;&emsp;模型越来越大，推理越来越贵，“跑得动”之后紧接着就是“撑得住、划得来、大家都能用”。精度与量化、批处理、KV Cache、框架配置、服务副本与请求分配，每一个决定都会牵动延迟、吞吐、内存与成本。然而，大多数学习资料要么停留在“某个框架怎么启动”，要么停留在论文公式：前者告诉你怎么跑，却不告诉你为什么这样配；后者给出模型，却不告诉你它离真机有多远。

&emsp;&emsp;本项目覆盖 AMD 的多类硬件：**Radeon RX 9070 XT、RX 7900 XTX** 等桌面显卡价格亲民，学生和小型实验室也能完成真机实验；**Ryzen AI MAX+ 395** 等端侧设备把大容量统一内存带到了个人电脑上，适合端侧部署与资源受限场景（适配中）；**W7900D 48 GB** 多卡计划用于多卡与集群实验。服务评测方法、显存账本、通信模型与排队论在任何平台上都同样成立，换硬件只需换参数表。

&emsp;&emsp;围绕这条主线，本教程坚持三个原则：

| 原则 | 含义 | 你会怎么练 |
| ---- | ---- | ---- |
| **以服务为目标** | 终点是一个能用的 LLM 服务，而不是一次推理 | 从单机 API 服务做到多人共享的服务平台 |
| **以评测为依据** | 每一项优化都要有统一口径的实测数据 | 固定模型与任务，测输入输出长度、并发、延迟、吞吐与内存 |
| **先预测再实测** | 用模拟器缩小搜索范围，用真机裁决 | 预测先留痕，只测信息量最大的少数配置，解释误差并校准 |

&emsp;&emsp;本教程聚焦 **LLM 推理服务**；AI 入门、训练与微调请见 hello-rocm，GPU 编程与算子优化请见 hello-gpu。

&emsp;&emsp;***我们希望让每一位学习者都能连接手边的设备、搭建自己的服务平台，让每一份 AI 算力都用得更高效。***

## 项目受众

&emsp;&emsp;本项目适合以下学习者：

* 已经能在本地跑通模型，希望把它部署成稳定服务、供应用或团队调用的开发者；
* 实验室或团队里有几台 AMD 设备，想搭建多人共享 LLM 服务平台的同学与工程师；
* 想进入 MLSys / AI Infra / 推理优化方向，希望掌握服务评测、性能优化与多机部署方法的学习者；
* 需要为推理服务做容量规划、量化选型或框架选型，却缺少实测依据的开发者；
* 希望了解 AMD GPU / ROCm 生态下 LLM 服务系统的学习者。

> &emsp;&emsp;你不需要有算子开发经验，但最好具备 Python、PyTorch、Linux 基础与英文文档阅读能力；模型使用与微调的基础可以先在 hello-rocm 中补齐。

## 项目规划及进展

### 章节与实验覆盖

> ✅ 表示该章正文已包含对应类型的结果（模拟器预测或真机实测），🚧 表示规划中或适配中。表中只统计已完成的实验；第 4、5 章另含 RX 7900 XTX 实测。

| 篇 / 章 | 核心实验 / 主题 | 模拟器 | RX 9070 XT | W7900D | Ryzen AI MAX+ 395 | 状态 |
| ---- | ---- | :----: | :----: | :----: | :----: | :----: |
| [Part 0 入门](https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/) | 定位、硬件图谱、统一测量协议、纸面手算 | | | | | 🚧 |
| [Ch 1](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter1/) | 使用模拟器估算时间、Roofline 与瓶颈 | ✅ | ✅ | 🚧 | | 🚧 |
| [Ch 2](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter2/) | 第一次跑通 Profile，并使用实际样例校准 | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 3](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter3/) | 显存占用估算：模型权重与 KV Cache | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 4](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter4/) | 精度、量化方法与误差预算 | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 5](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter5/) | 通信代价模型与 Scale-in / Scale-up | ✅ | 🚧 | 🚧 | | 🚧 |
| [Ch 6](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter6/) | 模型并行策略与显存 / 算力拆解 | ✅ | ✅（行为验证） | 🚧 | 🚧 | 🚧 |
| [Ch 7](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter7/) | 服务、排队与 SLA，负载生成与压测 | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 8](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter8/) | 推理框架的系统取舍、Profile 方式与性能分析 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [Ch 9.1](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-1/) | 与 AI 一起实战：端侧部署助手 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [Ch 9.2](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-2/) | 与 AI 一起实战：单机服务满足 SLA | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [Ch 9.3](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-3/) | 与 AI 一起实战：多卡长上下文、多 Subagent 请求处理 | 🚧 | | 🚧 | | 🚧 |
| [Part 3](https://datawhalechina.github.io/hello-mlsys/part3-electives/) | 选修与案例库：优先选修实验与扩展案例卡 | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [附录](https://datawhalechina.github.io/hello-mlsys/curriculum/) | 硬件参数、模型卡、术语表、参考资源、社区共建、常见失败模式与排查清单 | | | | | 🚧 |

### 项目结构

```
hello-mlsys/
├── docs/                      # VitePress 教程站点源文件
│   ├── part0-intro/           # 第 0 篇：入门
│   ├── part1-modeling/        # 第 1 篇：建模与校准
│   ├── part2-ai-design/       # 第 2 篇：与 AI 一起设计
│   ├── part3-electives/       # 第 3 篇：选修与案例库
│   ├── appendix/              # 附录 A–F
│   ├── curriculum/            # 全书目录（由大纲生成）
│   └── .vitepress/            # 站点配置与全书大纲（outline.mjs）
├── code/                      # 各章实验代码、测量记录与图表脚本
├── notebooks/                 # Notebook 动手学习
└── scripts/                   # 环境配置、构建与检查脚本
```

## 目录

> 章节顺序就是推荐的学习顺序。全书内容覆盖「单机部署、服务评测、单机优化、多机服务、进阶协同」五个模块，其中「单机部署」与「多机服务」正在编写，完成后插入相应位置。vLLM 走通主线，llama.cpp 做对照与端侧；涉及 Radeon 多卡张量并行的结论（第 5、6 章）以本书实测为准。

| 章节名 | 简介 | 所属模块 | 状态 |
| ---- | ---- | ---- | ---- |
| **第 0 篇：入门** | | | |
| [第 0 章 协议与纸面演练](./docs/part0-intro/chapter0/index.md) | 教程定位、硬件图谱、统一测量协议、模拟器工具链与能力边界 | 性能分析基础 | 🚧 |
| **第 1 篇：建模与校准** | | | |
| [第 1 章 模拟器估算、Roofline 与瓶颈](./docs/part1-modeling/chapter1/index.md) · [代码](./code/part1-modeling/chapter1) | Roofline、算术强度、Compute-bound / Memory-bound 判定、计算时间量级估算 | 性能分析基础 | 🚧 |
| [第 2 章 第一次真机 Profile 与性能校准](./docs/part1-modeling/chapter2/index.md) · [代码](./code/part1-modeling/chapter2) | 真机准入检查、最小 Profile 跑通、误差归因、校准集与留出验证 | 性能分析基础 | 🚧 |
| [第 3 章 显存占用估算：模型权重与 KV Cache](./docs/part1-modeling/chapter3/index.md) · [代码](./code/part1-modeling/chapter3) | 权重显存、KV Cache、激活与碎片开销、最大可服务上下文与并发 | 单机优化 | 🚧 |
| [第 4 章 精度、量化方法与误差预算](./docs/part1-modeling/chapter4/index.md) · [代码](./code/part1-modeling/chapter4) | 数值格式、精度 / 延迟 / 显存三维权衡、误差预算分配 | 单机优化 | 🚧 |
| [第 5 章 通信代价模型与 Scale-in / Scale-up](./docs/part1-modeling/chapter5/index.md) · [代码](./code/part1-modeling/chapter5) | 节点内与节点间互联、集合通信原语、α–β 代价模型 | 进阶协同 | 🚧 |
| [第 6 章 模型并行策略与显存 / 算力拆解](./docs/part1-modeling/chapter6/index.md) · [代码](./code/part1-modeling/chapter6) | DP / FSDP、TP、PP、切分收益被通信吃掉的判定条件 | 进阶协同 | 🚧 |
| [第 7 章 服务性能与服务质量](./docs/part1-modeling/chapter7/index.md) · [代码](./code/part1-modeling/chapter7) | 衡量推理服务的延迟与有效吞吐，按 SLO 判断达标容量和服务余量。比较开环与闭环压测，用排队论分析等待，讲解连续批处理中的请求调度与缓存约束，并通过离散事件模拟预测服务表现。 | 服务评测 | 🚧 |
| [第 8 章 推理框架的系统取舍、Profile 与性能分析](./docs/part1-modeling/chapter8/index.md) · [代码](./code/part1-modeling/chapter8) | 同卡 vLLM vs llama.cpp、显存失败模式、留出预测、框架级 Profile 与选型决策 | 单机优化 | 🚧 |
| **第 2 篇：与 AI 一起设计** | | | |
| [第 9.1 章 端侧部署助手](./docs/part2-ai-design/chapter9-1/index.md) | 资源受限场景下的模型、精度、上下文与 KV 配置选型 | 单机优化 | 🚧 |
| [第 9.2 章 单机服务满足 SLA](./docs/part2-ai-design/chapter9-2/index.md) | 单卡容量规划：显存上界与排队上界、实测拐点与留出验证 | 单机优化 | 🚧 |
| [第 9.3 章 负载启发的推理集群设计](./docs/part2-ai-design/chapter9-3/index.md) | 冻结的 deep-research 轨迹上以总墙钟最短为目标：32B/7B 两档模型协作与多种并行策略，先模拟筛选配置，再在 W7900D 48 G 多卡集群实测 | 进阶协同 | 🚧 |
| **第 3 篇：选修与案例库** | | | |
| [选修与案例库](./docs/part3-electives/index.md) | 优先选修实验与扩展案例卡 | 选修 | 🚧 |
| **附录** | | | |
| [附录 A 硬件参数](./docs/appendix/appendix-a-hardware/index.md) | 待补充 | —— | 🚧 |
| [附录 B 模型卡](./docs/appendix/appendix-b-model-cards/index.md) | 待补充 | —— | 🚧 |
| [附录 C 术语表](./docs/appendix/appendix-c-glossary/index.md) | 待补充 | —— | 🚧 |
| [附录 D 参考资源](./docs/appendix/appendix-d-references/index.md) | 待补充 | —— | 🚧 |
| [附录 E 社区共建](./docs/appendix/appendix-e-community/index.md) | 待补充 | —— | 🚧 |
| [附录 F 常见失败模式与排查清单](./docs/appendix/appendix-f-troubleshooting/index.md) | 含 rocprofv3 PMC 修复记录 | —— | 🚧 |

## 贡献者名单

### 核心贡献者

- 陆遥-指导教师-项目负责人（新加坡国立大学计算机学院助理教授）
- [yuzhou-项目负责人](https://github.com/youyoulyz)（上海科技大学博士在读）
- [秦博文-项目成员](https://github.com/eyuansu62)（新加坡国立大学博士在读，[mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator) 模拟器搭建与校准）
- [潘嘉航-项目成员](https://github.com/amdjiahangpan)（AMD 软件工程师）
- [刘伟鸿-项目成员](https://github.com/Weihong-Liu)（Datawhale 成员）
- yanhui-项目成员（新加坡国立大学 MComp 硕士）
- [郑纪言-项目成员](https://github.com/Keyan0412)（新加坡国立大学 MComp 硕士）

> 注：欢迎更多贡献者加入！

### 其他

- 如果有任何想法可以联系我们，也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！
- 感谢 AMD University Program 对本项目的支持！

<div align="center">
  <a href="https://github.com/datawhalechina/hello-mlsys/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-mlsys" alt="Hello MLSys 项目贡献者" />
  </a>
</div>

## 参与贡献

&emsp;&emsp;我们欢迎所有形式的贡献！无论是：

* 完善或新增教程章节
* 修复错误与 Bug
* 在其他 AMD 设备上复跑实验、补充实测与校准数据
* 分享你在 AMD 设备上部署与优化 LLM 服务的实践经验
* 提出建议与想法

&emsp;&emsp;参与前请先阅读 **[贡献指南 CONTRIBUTING.md](./CONTRIBUTING.md)**，其中说明了内容放置位置、章节与实验记录规范、本地预览方式与 PR 流程。

- 如果你发现了一些问题，可以提 Issue 进行反馈；
- 如果你想参与贡献本项目，可以提 Pull Request，我们会尽快 review。

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

---

<div align="center">

**让我们一起连接设备、搭建平台，让每一份 AI 算力更高效！** 💪

Made with ❤️ by the hello-mlsys community

</div>
