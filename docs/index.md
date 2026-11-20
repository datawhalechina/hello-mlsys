---
layout: home
title: Hello MLSys
description: 连接设备，搭建平台，让每一份 AI 算力更高效。基于 AMD 硬件，以 vLLM 等开源推理框架为实践载体，学习从单机部署到多机 LLM 服务。
hero:
  name: "Hello MLSys"
  text: "连接设备，搭建平台，让每一份 AI 算力更高效"
  tagline: 基于 AMD 硬件，以 vLLM 等开源推理框架为实践载体，从单机部署走到多机 LLM 服务
  actions:
    - theme: brand
      text: 开始学习
      link: /part0-intro/chapter0/
    - theme: alt
      text: 全书目录
      link: /curriculum/
features:
  - title: 单机部署
    details: 模型加载、API 服务与应用调用，得到一个可运行的 LLM 服务。🚧 规划中
  - title: 服务评测
    details: 输入输出长度、并发、延迟、吞吐与内存，形成标准化性能报告。
    link: /part1-modeling/chapter7/
    linkText: 第 7 章 服务、排队与 SLA
  - title: 单机优化
    details: 精度与量化、批处理、KV Cache、框架配置，给出有实测依据的优化方案。
    link: /part1-modeling/chapter3/
    linkText: 从第 3 章显存与 KV Cache 开始
  - title: 多机服务
    details: 服务副本、统一入口、请求分发与健康检查，搭建多人共享的 LLM 服务平台。🚧 规划中
  - title: 进阶协同
    details: 模型并行、分布式推理与跨节点性能分析，明确多设备协同的适用条件。
    link: /part1-modeling/chapter5/
    linkText: 从第 5 章通信代价模型开始
---

::: info Alpha 内测版
教程正在持续完善，部分章节仍为大纲或待补充内容。阅读时请以各章现有正文与验证记录为准，欢迎通过 GitHub Issue 反馈问题。
:::

## 这本教程学什么

模型能跑起来之后，下一步往往是把它变成大家都能用的服务：一台机器能撑住多少并发？精度与量化、批处理、KV Cache 和框架配置该怎么选？实验室里的几台 AMD 设备，又该怎样连成一个多人共享的 LLM 服务？

Hello MLSys 以 vLLM 在 Linux 上走通主线、llama.cpp 做对照实验并覆盖 Ryzen AI MAX+ 395 等端侧设备，带你从单机部署走到多机 LLM 服务，重点回答一个问题：

**给定设备、任务和服务目标，怎样选择模型精度、推理框架与资源配置，让整体服务更高效？**

为了让每个选择都有依据，本教程坚持**先预测、再实测**：先用模拟器 [mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator) 估算性能、显存与容量边界，再挑选少量配置上 AMD 真机验证，解释偏差、校准模型，最后做出配置决策。

## 建议学习顺序

章节顺序就是推荐的学习顺序：

1. [第 0 章 协议与纸面演练](/part0-intro/chapter0/)：统一 token、GB/GiB、延迟等测量口径。
2. **第 1–2 章 · 性能分析基础**：[Roofline 与瓶颈](/part1-modeling/chapter1/)、[Profile 与性能校准](/part1-modeling/chapter2/)。
3. **第 3–4 章 · 单机优化**：[显存与 KV Cache](/part1-modeling/chapter3/)、[精度与量化](/part1-modeling/chapter4/)。
4. **第 5–6 章 · 进阶协同**：[通信代价模型](/part1-modeling/chapter5/)、[模型并行策略](/part1-modeling/chapter6/)。
5. **第 7–8 章 · 服务评测与框架取舍**：[服务、排队与 SLA](/part1-modeling/chapter7/)，以及推理框架的系统取舍。
6. **第 2 篇 · 与 AI 一起设计**：把前面的方法用于端侧、单机与集群设计。

单机部署与多机服务两个模块正在编写中，完成后插入全书相应位置；完整章节见[全书目录](/curriculum/)。

## AMD Hello 系列

三个项目可以独立学习，也能通过同一个模型串联实践：在 hello-rocm 中训练、微调和使用模型，在 hello-gpu 中理解并优化计算，在 hello-mlsys 中将模型部署为高效的共享服务。

| 项目 | 标语 | 核心定位 |
| ---- | ---- | ---- |
| [hello-rocm](https://github.com/datawhalechina/hello-rocm) | 每个 AI 小白，从零开始，动手实践的第一站。 | 基于 AMD 平台的 AI 与 Agentic AI 入门实践手册 |
| [hello-gpu](https://github.com/datawhalechina/hello-gpu) | 看懂 GPU，亲手优化每一次计算。 | 从自己训练的小模型数据类型、GEMM 优化出发，学习 GPU 编程与 Agent 辅助优化 |
| **hello-mlsys** | 连接设备，搭建平台，让每一份 AI 算力更高效。 | 以 vLLM 等开源推理框架为案例，学习 LLM 服务部署、性能优化与多机服务构建 |

## 开始之前

建议具备 Python、PyTorch 和 Linux 基础。环境安装、AI 入门与模型微调可参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm)，GPU 编程与算子优化可参考 [hello-gpu](https://github.com/datawhalechina/hello-gpu)。
