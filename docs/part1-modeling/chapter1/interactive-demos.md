---
title: 第 1 章交互演示维护说明
---

# 第 1 章交互演示

- [公式动画整章阅读版](https://datawhalechina.github.io/hello-mlsys/demos/chapter1/reader.html)
- [三个综合实验](https://datawhalechina.github.io/hello-mlsys/demos/chapter1/index.html)
- [返回第 1 章](./index.md)

## 文件与更新

`index.md` 是章节正文来源。`FormulaDemo` 是 VitePress 全局组件，每个公式下使用 `<FormulaDemo scene="max" />` 一类标签；GitHub 读者可以展开该位置的预览并进入独立演示。

浏览器资源位于 `docs/public/demos/chapter1/`，会被现有 VitePress 构建原样复制到网站。组件使用 `withBase` 适配项目子路径，iframe 通过同源消息同步高度；每个公式的动画状态独立。

修改章节文字后，更新独立阅读版：

```sh
python -m pip install -r scripts/requirements-chapter1.txt
python scripts/build-chapter1-reader.py
node tests/chapter1-formulas.mjs
npm run docs:build
```

生成后的 `reader.html` 需要一并提交。原始四张配图分别保留在章节目录，并在演示目录保存对应副本，供独立阅读版使用；更新配图时两处同步。

## 发布

复用 `.github/workflows/deploy.yml`：合并到 `main` 后，现有工作流构建并发布 GitHub Pages。无需新开网站或运行服务器。上线状态以仓库 Actions 和实际网页访问结果为准。

## 计数口径

动画展示解析模型或机制示意，不将模型结果标成真机实测。沿用已确认的原型计数校正：naive attention 按本文矩阵写一次、读一次口径，AI 大 S 极限为 `d_q/n_heads=128`；BF16 单次读取和写回共 4 bytes/元素。FA 的线性 IO 为本章理想输入输出口径，不涵盖所有 tile 重读、缓存和实现细节。
