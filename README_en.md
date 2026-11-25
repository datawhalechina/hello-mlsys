<div align="center">
  <img src="./docs/public/images/head_en.png" alt="Hello MLSys: Connect devices, build platforms, make every unit of AI compute more efficient" width="100%">
  <p><strong>Connect devices, build platforms, make every unit of AI compute more efficient 🚀</strong></p>
</div>

<div align="center">

*Single-machine deployment · Serving benchmarks · Single-machine optimization · Multi-machine serving · Advanced collaboration*

<p align="center">
  <a href="./README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="./README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
</p>

</div>

<div align="center">

<a href="https://datawhalechina.github.io/hello-mlsys/"><img alt="Read Online" src="https://img.shields.io/badge/%F0%9F%9A%80_Read_Online-Full_Tutorial-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://datawhalechina.github.io/hello-mlsys/curriculum/"><img alt="Curriculum" src="https://img.shields.io/badge/%F0%9F%93%9A_Curriculum-Chapter_Guide-ed1941?logo=amd&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://github.com/datawhalechina/hello-rocm"><img alt="hello-rocm: AI Hands-on Start" src="https://img.shields.io/badge/%F0%9F%9B%A0%EF%B8%8F_hello--rocm-AI_Hands--on_Start-1a73e8?logo=github&logoColor=white&labelColor=1a1a1a" height="28"></a> | <a href="https://github.com/datawhalechina/hello-gpu"><img alt="hello-gpu: GPU Optimization" src="https://img.shields.io/badge/%E2%9A%A1_hello--gpu-GPU_Optimization-1a73e8?logo=github&logoColor=white&labelColor=1a1a1a" height="28"></a>

</div>

<p align="center">
  <img src="./docs/public/aup-logo.png" alt="AMD University Program" height="44">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./docs/public/rocm-logo.png" alt="AMD ROCm" height="44">
</p>

> [!CAUTION]
> ⚠️ Alpha preview: this is an early internal build. It is incomplete and may contain errors. Issues with bug reports or suggestions are welcome.
>
> The tutorial body is currently written in Chinese; this English README gives an overview of the project.

&emsp;&emsp;🎓 **Learn MLSys together with students at the National University of Singapore (NUS)!** This project is co-built by NUS students, and more NUS students will join in to learn and build along the way. Join us and go from serving a model on one machine to a multi-user LLM serving platform on AMD devices.

&emsp;&emsp;Once a model runs, the next job is usually turning it into a service other people can use: how many concurrent requests one machine can handle, how long the first token takes and whether generation is fast enough, how to set quantization, batching, KV cache, and framework parameters, and how to connect a few machines into one shared service.

&emsp;&emsp;All of these involve trade-offs among model, hardware, framework configuration, workload, and service targets, and trial and error is slow and burns machine time.

&emsp;&emsp;hello-mlsys is meant to spare you those detours. It is built on AMD hardware: the main line uses vLLM on Linux, and llama.cpp is used for comparison experiments and on-device deployment. The whole book revolves around one question:

> Given the devices, tasks, and service targets, how should you choose model precision, inference framework, and resource configuration so the whole service runs more efficiently?

&emsp;&emsp;In practice, we predict first and measure second. Before tuning anything, use [mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator) to estimate rough limits on performance, memory, and capacity, and write the predictions down; then take only the few configurations that tell you the most to real AMD hardware, explain the gaps, calibrate the model, and make the configuration decision.

&emsp;&emsp;Part 2, "Designing with AI", has a companion agent-based learning workbench, [hello-mlsys-lab](https://github.com/taowangcheng/hello-mlsys-lab), which is still in development and testing. Learners talk with an AI assistant in a local course page or the terminal to discuss design tasks, run simulations with [mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator), plan experiments, and compare the simulated results with measurements. Until the AMD sandbox is connected, hardware measurements use stand-in data.

&emsp;&emsp;**This is not a user manual for inference frameworks. By the end, you should be able to deploy a model as a service yourself, benchmark it under consistent conventions, find the bottleneck and tune it, and know when and how to add machines. Anyone is welcome to open an issue or submit a PR to help build and maintain this project.**

### The AMD Hello Series

&emsp;&emsp;hello-mlsys is one of three projects in the AMD Hello series; the other two are hello-rocm and hello-gpu. You can study them separately or connect them through the same model: train, fine-tune, and use it in hello-rocm, understand and optimize its computation in hello-gpu, then deploy it as a shared service in hello-mlsys.

| Project | Slogan | Focus |
| ---- | ---- | ---- |
| [**hello-rocm**](https://github.com/datawhalechina/hello-rocm) | The first hands-on stop for every AI beginner, starting from zero. | A hands-on guide to AI and agentic AI on AMD platforms |
| [**hello-gpu**](https://github.com/datawhalechina/hello-gpu) | Understand the GPU and optimize every computation yourself. | Starting from the data types and GEMM optimization of a small model you trained, learn GPU programming and agent-assisted optimization |
| **hello-mlsys** (this project) | Connect devices, build platforms, make every unit of AI compute more efficient. | Using vLLM and other open-source inference frameworks as case studies, learn LLM serving deployment, performance optimization, and multi-machine serving |

> &emsp;&emsp;***Study tip: start with [Chapter 0: Protocol and Paper Drill](https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/) to align on measurement conventions such as tokens, GB/GiB, and latency; then read Chapters 1 and 2 for Roofline, profiling, and calibration; work through Chapters 3–8 in order, and finally put these methods to use in the three design problems of Part 2. No AMD GPU at hand? The Roofline predictions in Chapter 1 and the memory ledger in Chapter 3 run on a CPU alone.***

## Why This Project

&emsp;&emsp;What is MLSys?

> MLSys (Machine Learning Systems) studies how to run machine learning models efficiently and reliably on real hardware under real workloads while keeping costs under control. For LLM inference, what counts is how much useful work the whole service can reliably deliver; how fast a single request runs is only part of it.

&emsp;&emsp;Models keep growing and inference keeps getting more expensive, so a wrong configuration costs more as well. Most material covers only half of the story: framework tutorials show you how to launch a server but rarely explain why it is configured that way or whether the setting still holds on another machine, and the models in papers are seldom checked against real hardware.

&emsp;&emsp;The experiments cover three kinds of AMD hardware: desktop GPUs (Radeon RX 9070 XT, RX 7900 XTX), affordable enough for students and small labs to run real-hardware experiments; edge devices (Ryzen AI MAX+ 395), whose large unified memory suits on-device deployment and resource-constrained scenarios; and workstation GPUs (Radeon PRO W7900D 48 GB), combined into multi-GPU setups for parallelism and cluster experiments. The benchmarking methods, memory ledgers, communication models, and queueing theory in this book carry over to other platforms; switching hardware only means switching the parameters.

&emsp;&emsp;***We hope every learner can put the devices at hand to work, build a serving platform of their own, and be able to say where every unit of compute goes.***

## Who This Is For

&emsp;&emsp;This project is for:

* Developers who can already run a model locally and want to deploy it as a stable service for applications or their team;
* Students and engineers whose lab or team has a few AMD machines and wants to build a shared LLM serving platform;
* Learners who want to enter MLSys / AI Infra / inference optimization and master serving benchmarks, performance optimization, and multi-machine deployment;
* Developers who need to do capacity planning, quantization selection, or framework selection for inference services but lack measured evidence;
* Learners who want to understand LLM serving systems in the AMD GPU / ROCm ecosystem.

> &emsp;&emsp;You do not need kernel development experience, but you should be comfortable with Python, PyTorch, Linux, and reading English documentation; environment setup and the basics of using and fine-tuning models can be picked up in hello-rocm first.

## Roadmap and Progress

### Chapters and Experiment Coverage

> ✅ means the chapter text already includes that type of result (simulator prediction or real-hardware measurement); 🚧 means planned or in progress. Chapters 4, 5, and 8 also include RX 7900 XTX measurements (Chapter 8 is measured on the RX 7900 XTX only).

| Part / Chapter | Core experiment / topic | Simulator | RX 9070 XT | W7900D | Ryzen AI MAX+ 395 | Status |
| ---- | ---- | :----: | :----: | :----: | :----: | :----: |
| [Part 0 Introduction](https://datawhalechina.github.io/hello-mlsys/part0-intro/chapter0/) | Positioning, hardware map, unified measurement protocol, paper calculation | | | | | 🚧 |
| [Ch 1](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter1/) | Estimating time with a simulator, Roofline, and bottlenecks | ✅ | ✅ | 🚧 | | 🚧 |
| [Ch 2](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter2/) | First end-to-end profile and calibration with real samples | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 3](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter3/) | Memory estimation: model weights and KV cache | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 4](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter4/) | Precision, quantization, and error budgets | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 5](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter5/) | Communication cost models and scale-in / scale-up | ✅ | 🚧 | 🚧 | | 🚧 |
| [Ch 6](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter6/) | Model parallelism and memory / compute partitioning | ✅ | ✅ (behavior check) | 🚧 | 🚧 | 🚧 |
| [Ch 7](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter7/) | Serving performance and quality: latency, goodput, and SLO capacity | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| [Ch 8](https://datawhalechina.github.io/hello-mlsys/part1-modeling/chapter8/) | Inference framework trade-offs, profiling, and performance analysis | ✅ | 🚧 | 🚧 | 🚧 | 🚧 |
| [Ch 9.1](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-1/) | Designing with AI: on-device deployment assistant | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [Ch 9.2](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-2/) | Designing with AI: meeting an SLA on a single machine | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [Ch 9.3](https://datawhalechina.github.io/hello-mlsys/part2-ai-design/chapter9-3/) | Designing with AI: multi-GPU long-context, multi-subagent request handling | 🚧 | | 🚧 | | 🚧 |
| [Part 3](https://datawhalechina.github.io/hello-mlsys/part3-electives/) | Electives and case library: priority electives and extended case cards | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| [Appendices](https://datawhalechina.github.io/hello-mlsys/curriculum/) | Hardware parameters, model cards, glossary, references, community, common failure modes and troubleshooting | | | | | 🚧 |

### Repository Layout

```
hello-mlsys/
├── docs/                      # VitePress tutorial site sources
│   ├── part0-intro/           # Part 0: Introduction
│   ├── part1-modeling/        # Part 1: Modeling and Calibration
│   ├── part2-ai-design/       # Part 2: Designing with AI
│   ├── part3-electives/       # Part 3: Electives and Case Library
│   ├── appendix/              # Appendices A–F
│   ├── curriculum/            # Full curriculum (generated from the outline)
│   └── .vitepress/            # Site config and book outline (outline.mjs)
├── code/                      # Per-chapter experiment code, measurement records, and plotting scripts
├── notebooks/                 # Hands-on notebooks
└── scripts/                   # Environment setup, build, and check scripts
```

## Contents

> The chapter order is the recommended learning order. The book covers five modules: single-machine deployment, serving benchmarks, single-machine optimization, multi-machine serving, and advanced collaboration; "single-machine deployment" and "multi-machine serving" are being written and will be inserted at the appropriate places. Conclusions about multi-GPU tensor parallelism on Radeon (Chapters 5 and 6) rely on our own measurements. Chapter links point to the Chinese text.

| Chapter | Summary | Module | Status |
| ---- | ---- | ---- | ---- |
| **Part 0: Introduction** | | | |
| [Chapter 0 Protocol and Paper Drill](./docs/part0-intro/chapter0/index.md) | Tutorial positioning, hardware map, unified measurement protocol, simulator toolchain and its limits | Performance-analysis foundations | 🚧 |
| **Part 1: Modeling and Calibration** | | | |
| [Chapter 1 Simulator Estimates, Roofline, and Bottlenecks](./docs/part1-modeling/chapter1/index.md) · [code](./code/part1-modeling/chapter1) | Roofline, arithmetic intensity, compute-bound / memory-bound classification, order-of-magnitude time estimates | Performance-analysis foundations | 🚧 |
| [Chapter 2 First Real-Hardware Profile and Performance Calibration](./docs/part1-modeling/chapter2/index.md) · [code](./code/part1-modeling/chapter2) | Hardware readiness checks, minimal profiling, error attribution, calibration and holdout sets | Performance-analysis foundations | 🚧 |
| [Chapter 3 Memory Estimation: Model Weights and KV Cache](./docs/part1-modeling/chapter3/index.md) · [code](./code/part1-modeling/chapter3) | Weight memory, KV cache, activation and fragmentation overhead, maximum servable context and concurrency | Single-machine optimization | 🚧 |
| [Chapter 4 Precision, Quantization, and Error Budgets](./docs/part1-modeling/chapter4/index.md) · [code](./code/part1-modeling/chapter4) | Numeric formats, precision / latency / memory trade-offs, error budget allocation | Single-machine optimization | 🚧 |
| [Chapter 5 Communication Cost Models and Scale-in / Scale-up](./docs/part1-modeling/chapter5/index.md) · [code](./code/part1-modeling/chapter5) | Intra-node and inter-node interconnects, collective communication primitives, α–β cost model | Advanced collaboration | 🚧 |
| [Chapter 6 Model Parallelism and Memory / Compute Partitioning](./docs/part1-modeling/chapter6/index.md) · [code](./code/part1-modeling/chapter6) | DP / FSDP, TP, PP, and when communication eats the gains from partitioning | Advanced collaboration | 🚧 |
| [Chapter 7 Serving Performance and Quality of Service](./docs/part1-modeling/chapter7/index.md) · [code](./code/part1-modeling/chapter7) | Measure latency and goodput of an inference service and judge SLO-compliant capacity and headroom; compare open-loop and closed-loop load tests, analyze waiting with queueing theory, explain request scheduling and cache constraints in continuous batching, and predict service behavior with discrete-event simulation | Serving benchmarks | 🚧 |
| [Chapter 8 Inference Framework Trade-offs, Profiling, and Performance Analysis](./docs/part1-modeling/chapter8/index.md) · [code](./code/part1-modeling/chapter8) | Same-card vLLM vs llama.cpp, memory failure modes, holdout prediction, framework-level profiling, and the selection decision | Single-machine optimization | 🚧 |
| **Part 2: Designing with AI** | | | |
| [Chapter 9.1 On-Device Deployment Assistant](./docs/part2-ai-design/chapter9-1/index.md) | Choosing model, precision, context, and KV configuration under tight resource limits | Single-machine optimization | 🚧 |
| [Chapter 9.2 Meeting an SLA on a Single Machine](./docs/part2-ai-design/chapter9-2/index.md) | Single-GPU capacity planning: memory and queueing upper bounds, measured knee, holdout validation | Single-machine optimization | 🚧 |
| [Chapter 9.3 Workload-Driven Inference Cluster Design](./docs/part2-ai-design/chapter9-3/index.md) | Minimizing total wall-clock time on a frozen deep-research trace: 32B/7B model collaboration and multiple parallel strategies, screened in simulation and then measured on a multi-GPU W7900D 48 GB cluster | Advanced collaboration | 🚧 |
| **Part 3: Electives and Case Library** | | | |
| [Electives and Case Library](./docs/part3-electives/index.md) | Priority electives and extended case cards | Electives | 🚧 |
| **Appendices** | | | |
| [Appendix A Hardware Parameters](./docs/appendix/appendix-a-hardware/index.md) | Coming soon | —— | 🚧 |
| [Appendix B Model Cards](./docs/appendix/appendix-b-model-cards/index.md) | Coming soon | —— | 🚧 |
| [Appendix C Glossary](./docs/appendix/appendix-c-glossary/index.md) | Coming soon | —— | 🚧 |
| [Appendix D References](./docs/appendix/appendix-d-references/index.md) | Coming soon | —— | 🚧 |
| [Appendix E Community](./docs/appendix/appendix-e-community/index.md) | Coming soon | —— | 🚧 |
| [Appendix F Common Failure Modes and Troubleshooting Checklist](./docs/appendix/appendix-f-troubleshooting/index.md) | Includes the rocprofv3 PMC fix record | —— | 🚧 |

## Contributors

### Core Contributors

- Yao Lu - Faculty Advisor & Project Lead (Assistant Professor, School of Computing, National University of Singapore)
- [yuzhou - Project Lead](https://github.com/youyoulyz) (PhD student, ShanghaiTech University)
- [Bowen Qin - Project Member](https://github.com/eyuansu62) (PhD student, National University of Singapore; builds and calibrates [mlsys-simulator](https://github.com/eyuansu62/mlsys-simulator))
- [Tao Wangcheng - Project Member](https://github.com/taowangcheng) (PhD student, National University of Singapore; develops and tests the companion agent [hello-mlsys-lab](https://github.com/taowangcheng/hello-mlsys-lab))
- [Jiahang Pan - Project Member](https://github.com/amdjiahangpan) (Software Engineer, AMD)
- [Weihong Liu - Project Member](https://github.com/Weihong-Liu) (Datawhale member)
- yanhui - Project Member (MComp student, National University of Singapore)
- [Jiyan Zheng - Project Member](https://github.com/Keyan0412) (MComp student, National University of Singapore)

> Note: more contributors are welcome!

### Others

- If you have any ideas, feel free to contact us, and please open issues
- Special thanks to everyone who has contributed to the tutorial!
- Thanks to the AMD University Program for supporting this project!

<div align="center">
  <a href="https://github.com/datawhalechina/hello-mlsys/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/hello-mlsys" alt="Hello MLSys contributors" />
  </a>
</div>

## Contributing

&emsp;&emsp;All kinds of contributions are welcome, for example:

* Improving existing chapters or adding new ones
* Fixing errors and bugs
* Re-running experiments on other AMD devices and adding measurement and calibration data
* Sharing your experience deploying and optimizing LLM services on AMD devices
* Sharing suggestions and ideas

&emsp;&emsp;Before contributing, please read the [Contributing Guide CONTRIBUTING_en.md](./CONTRIBUTING_en.md). It covers where content belongs, how to write chapters and measurement records, how to preview locally, and the PR workflow.

- If you find a problem, please open an issue;
- If you would like to contribute, please open a pull request and we will review it as soon as possible.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

---

<div align="center">

**Let's connect devices, build platforms, and make every unit of AI compute more efficient!** 💪

Made with ❤️ by the hello-mlsys community

</div>
