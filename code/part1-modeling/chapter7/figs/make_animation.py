#!/usr/bin/env python3
"""Build the chapter-7 animation from a DES trace, not hand-authored numbers.

Input : results/des_animation_trace.json
Output: docs/public/part1-modeling/chapter7/animation.html

Run: cd code/part1-modeling/chapter7 && python3 figs/make_animation.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE.parents[3] / "docs" / "public" / "part1-modeling" / "chapter7" / "animation.html"


TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>第 7 章 · 排队雪崩与 SLA 突破（DES 驱动）</title>
<style>
  :root {
    --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --surface:#fcfcfb; --card:#fff; --grid:#e1e0d9; --baseline:#c3c2b7;
    --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --red:#e34948;
    --neutral:#d8d7d0; --wash:#f0efea;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--surface); color:var(--ink); font:14px/1.5 -apple-system,
    "Segoe UI","Noto Sans CJK SC","PingFang SC","Microsoft YaHei",sans-serif;
    padding:28px 18px 44px; }
  .wrap { max-width:1100px; margin:0 auto; }
  header h1 { font-size:21px; letter-spacing:.01em; }
  header .sub { color:var(--ink2); font-size:13px; margin-top:5px; }
  header .sub code { font:12px ui-monospace,Consolas,monospace; background:var(--wash);
    padding:1px 5px; border-radius:3px; }
  .card { background:var(--card); border:1px solid var(--grid); border-radius:6px;
    padding:16px 18px; margin-top:18px; }
  .controls { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .btn { font:inherit; font-size:12.5px; background:var(--wash); border:1px solid var(--grid);
    border-radius:4px; padding:4px 12px; cursor:pointer; }
  .btn:hover { background:#e8e7e1; }
  .ctrl-label { color:var(--ink2); font-size:12px; }
  input[type=range] { width:150px; accent-color:var(--blue); }
  .stats { display:grid; grid-template-columns:repeat(8,1fr); gap:8px; margin-top:14px; }
  .stat { background:var(--wash); border:1px solid var(--grid); border-radius:4px;
    padding:7px 9px; }
  .stat .k { color:var(--muted); font-size:11px; }
  .stat .v { font-size:18px; font-weight:700; font-variant-numeric:tabular-nums; }
  .stat .v.small { font-size:14px; }
  .stage { display:grid; grid-template-columns:1fr 120px; gap:18px; margin-top:16px; }
  .engine { display:flex; flex-direction:column; gap:12px; }
  .lane { border:1px solid var(--grid); border-radius:5px; padding:9px 10px 10px;
    background:#fff; }
  .lane .head { display:flex; justify-content:space-between; align-items:baseline;
    margin-bottom:6px; }
  .lane .head b { font-size:12.5px; }
  .lane .head span { color:var(--muted); font-size:11.5px; font-variant-numeric:tabular-nums; }
  .dots { display:flex; flex-wrap:wrap; gap:3px; min-height:20px; align-items:center; }
  .dot { width:9px; height:9px; border-radius:2px; background:var(--neutral); opacity:.28; }
  .dot.on { opacity:1; }
  .waiting .dot.on { background:var(--orange); }
  .prefill .dot.on { background:var(--blue); }
  .running .dot.on { background:var(--aqua); }
  .overflow { color:var(--orange); font-size:12px; font-weight:700; margin-left:4px; }
  .kv { border:1px solid var(--grid); border-radius:5px; padding:9px; background:#fff;
    display:flex; flex-direction:column; align-items:center; }
  .kv .title { font-size:12px; font-weight:700; }
  .kv .bar { width:46px; height:224px; border-radius:4px; background:var(--wash);
    position:relative; margin:10px 0 6px; overflow:hidden; }
  .kv .fill { position:absolute; left:0; right:0; bottom:0; height:0%;
    background:linear-gradient(to top,var(--aqua),var(--blue)); transition:height .08s linear; }
  .kv .mark { position:absolute; left:0; right:0; bottom:95%; border-top:2px dashed var(--red);
    pointer-events:none; }
  .kv .cap { color:var(--muted); font-size:11px; text-align:center; }
  .kv .pct { font-size:18px; font-weight:700; font-variant-numeric:tabular-nums; }
  .kv .pct.hot { color:var(--red); }
  .sla { display:flex; justify-content:space-between; align-items:center; gap:12px;
    border:1px solid var(--grid); border-radius:6px; padding:11px 14px; background:#fff; }
  .sla .badge { font-weight:700; padding:5px 10px; border-radius:4px;
    background:#e8f5ef; color:#116b4d; font-size:13px; }
  .sla .badge.bad { background:#fbe9e7; color:var(--red); }
  .sla .detail { color:var(--ink2); font-size:12.5px; }
  .bottom { display:grid; grid-template-columns:1.05fr .95fr; gap:16px; margin-top:14px; }
  .panel { border:1px solid var(--grid); border-radius:5px; padding:10px 12px; background:#fff; }
  .panel h3 { font-size:12.5px; margin-bottom:6px; }
  table { width:100%; border-collapse:collapse; font-size:11.5px;
    font-variant-numeric:tabular-nums; }
  th,td { border-bottom:1px solid var(--grid); padding:4px 5px; text-align:right; }
  th:first-child,td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; }
  .ok { color:#116b4d; font-weight:700; }
  .bad { color:var(--red); font-weight:700; }
  .progress { height:7px; border-radius:99px; background:var(--wash); overflow:hidden; margin-top:12px; }
  .progress .inner { height:100%; width:0%; background:var(--blue); }
  .foot { color:var(--muted); font-size:12px; margin-top:16px; line-height:1.7; }
  .foot code { font:11.5px ui-monospace,Consolas,monospace; }
  @media (max-width:800px) {
    .stats { grid-template-columns:repeat(4,1fr); }
    .stage { grid-template-columns:1fr; }
    .kv { flex-direction:row; gap:18px; }
    .bottom { grid-template-columns:1fr; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>第 7 章 · 排队雪崩：到达 → 等待 → Continuous Batching → KV 水位 → SLA 突破</h1>
    <div class="sub" id="subtitle"></div>
  </header>

  <section class="card">
    <div class="controls">
      <button class="btn" id="play">暂停</button>
      <button class="btn" id="replay">↻ 重播</button>
      <span class="ctrl-label">播放速度</span>
      <input type="range" id="speed" min="0.25" max="4" step="0.25" value="1">
      <span class="ctrl-label" id="speedLabel">1.0×</span>
      <span class="ctrl-label" style="margin-left:auto">t = <b id="clock">0.0</b> s</span>
    </div>
    <div class="stats">
      <div class="stat"><div class="k">Waiting</div><div class="v" id="s-wait">0</div></div>
      <div class="stat"><div class="k">Prefill</div><div class="v" id="s-prefill">0</div></div>
      <div class="stat"><div class="k">Running</div><div class="v" id="s-run">0</div></div>
      <div class="stat"><div class="k">Engine batch</div><div class="v" id="s-batch">0</div></div>
      <div class="stat"><div class="k">KV 水位</div><div class="v" id="s-kv">0%</div></div>
      <div class="stat"><div class="k">Preemptions</div><div class="v" id="s-pre">0</div></div>
      <div class="stat"><div class="k">Completed</div><div class="v" id="s-done">0</div></div>
      <div class="stat"><div class="k">Goodput</div><div class="v small" id="s-good">0.00/s</div></div>
    </div>
    <div class="progress"><div class="inner" id="progress"></div></div>

    <div class="stage">
      <div class="engine">
        <div class="lane waiting"><div class="head"><b>① Waiting queue</b><span>请求已到达，尚未进入 Prefill</span></div><div class="dots" id="waiting"></div></div>
        <div class="lane prefill"><div class="head"><b>② Prefill</b><span>输入 token 正在建立 KV</span></div><div class="dots" id="prefill"></div></div>
        <div class="lane running"><div class="head"><b>③ Running batch</b><span>每一步真正上 GPU 的序列数 B</span></div><div class="dots" id="running"></div></div>
      </div>
      <aside class="kv">
        <div class="title">KV cache</div>
        <div class="bar"><div class="fill" id="kvfill"></div><div class="mark" title="95% 预警线"></div></div>
        <div class="pct" id="kvpct">0%</div>
        <div class="cap">24,464 tokens<br>3.36 GiB</div>
      </aside>
    </div>

    <div class="sla">
      <div class="badge" id="sla-badge">SLA 正常</div>
      <div class="detail" id="sla-detail">等待首个请求完成……</div>
      <div class="detail" id="sla-count"></div>
    </div>

    <div class="bottom">
      <div class="panel">
        <h3>最近完成（DES 逐请求结果）</h3>
        <table><thead><tr><th>#</th><th>TTFT</th><th>TPOT</th><th>E2E</th><th>判定</th></tr></thead>
          <tbody id="recent"></tbody></table>
      </div>
      <div class="panel">
        <h3>这条 run 的口径</h3>
        <table><tbody>
          <tr><td>数据源</td><td>DES run（不是真机曲线）</td></tr>
          <tr><td>到达</td><td>Poisson λ = <b id="m-rate"></b> req/s</td></tr>
          <tr><td>请求数</td><td id="m-n"></td></tr>
          <tr><td>模型/硬件</td><td>Qwen3-4B · RX 9070 XT</td></tr>
          <tr><td>ISL / OSL</td><td>256 / 128 tokens</td></tr>
          <tr><td>服务时间</td><td id="m-fit"></td></tr>
          <tr><td>KV 容量</td><td>24,464 tokens · 16-token pages</td></tr>
        </tbody></table>
      </div>
    </div>
    <div class="foot">
      <b>怎么读：</b>Waiting 是引擎外部的排队；Running 才是 GPU 每一步的 batch size。
      λ 超过系统容量后，等待队列持续变长，KV 水位逼近上限，SLA 先从尾部破裂。
      动画的每一个数字都来自随章 DES 的逐请求时间线；真机 vLLM 的 P99 对齐表见正文
      <code>7.6</code> 节与 <code>results/analysis.json</code>。
    </div>
  </section>
</div>

<script>
const DATA = __DATA__;
const trace = DATA.trace;
const reqs = DATA.requests;
const SLO = DATA.slo;
const CAP = DATA.kv_capacity_tokens;
const tEnd = trace.length ? trace[trace.length - 1].t : 0;

const el = (id) => document.getElementById(id);
const fmt = (x, d=0) => (x == null || Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const pass = (r) => r.ttft_ms <= SLO.ttft_p99_ms && r.e2e_ms <= SLO.e2el_p99_ms &&
                    (r.tpot_ms == null || r.tpot_ms <= SLO.tpot_p99_ms);
const firstFail = reqs.find(r => r.completion != null && !pass(r));
const firstFailT = firstFail ? firstFail.completion : Infinity;

function lowerBound(t) {
  let lo = 0, hi = trace.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (trace[mid].t < t) lo = mid + 1; else hi = mid;
  }
  return Math.max(0, Math.min(lo, trace.length - 1));
}

function makeDots(container, n, max) {
  container.innerHTML = "";
  for (let i = 0; i < max; i++) {
    const d = document.createElement("span");
    d.className = "dot";
    container.appendChild(d);
  }
  container._max = max;
}
makeDots(el("waiting"), 0, 60);
makeDots(el("prefill"), 0, 12);
makeDots(el("running"), 0, 64);

function paintDots(container, n) {
  const dots = container.children;
  for (let i = 0; i < dots.length; i++) dots[i].classList.toggle("on", i < n);
  let over = container.querySelector(".overflow");
  if (!over) {
    over = document.createElement("span");
    over.className = "overflow";
    container.appendChild(over);
  }
  over.textContent = n > container._max ? `+${n - container._max}` : "";
}

function recentRows(t) {
  const done = reqs.filter(r => r.completion != null && r.completion <= t);
  const rows = done.slice(-6).reverse();
  el("recent").innerHTML = rows.map(r => {
    const ok = pass(r);
    return `<tr><td>${r.rid}</td><td>${fmt(r.ttft_ms)}</td><td>${fmt(r.tpot_ms,1)}</td>` +
           `<td>${fmt(r.e2e_ms)}</td><td class="${ok ? "ok" : "bad"}">${ok ? "PASS" : "FAIL"}</td></tr>`;
  }).join("") || `<tr><td colspan="5" style="color:var(--muted)">尚无完成请求</td></tr>`;
  return done;
}

function render(t) {
  const i = lowerBound(t);
  const s = trace[i] || {waiting:0,prefill:0,running:0,batch:0,kv_tokens:0,preemptions:0,t:0};
  el("clock").textContent = fmt(t, 1);
  el("s-wait").textContent = s.waiting;
  el("s-prefill").textContent = s.prefill;
  el("s-run").textContent = s.running;
  el("s-batch").textContent = s.batch;
  const kv = Math.min(100, 100 * s.kv_tokens / CAP);
  el("s-kv").textContent = fmt(kv, 0) + "%";
  el("s-pre").textContent = s.preemptions;
  el("kvfill").style.height = kv + "%";
  el("kvpct").textContent = fmt(kv, 0) + "%";
  el("kvpct").classList.toggle("hot", kv >= 95);
  paintDots(el("waiting"), s.waiting);
  paintDots(el("prefill"), s.prefill);
  paintDots(el("running"), s.running);
  el("progress").style.width = (100 * t / Math.max(tEnd, 0.001)) + "%";

  const done = recentRows(t);
  const failed = done.filter(r => !pass(r)).length;
  el("s-done").textContent = done.length;
  el("s-good").textContent = fmt((done.length - failed) / Math.max(t, 0.001), 2) + "/s";
  el("sla-count").textContent = `已完成 ${done.length} · 失败 ${failed}`;
  if (t >= firstFailT) {
    el("sla-badge").textContent = "SLA 已突破";
    el("sla-badge").classList.add("bad");
    el("sla-detail").textContent = `首次失守 t=${fmt(firstFailT,1)} s：请求 #${firstFail.rid} ` +
      `TTFT ${fmt(firstFail.ttft_ms)} ms / E2E ${fmt(firstFail.e2e_ms)} ms`;
  } else {
    el("sla-badge").textContent = "SLA 正常";
    el("sla-badge").classList.remove("bad");
    el("sla-detail").textContent = done.length ? "目前完成的请求全部满足 P99 阈值。" : "等待首个请求完成……";
  }
}

let playing = true;
let speed = 1.0;
let t0 = performance.now();
let base = 0;
let raf = null;

function frame(now) {
  if (playing) {
    const t = base + (now - t0) / 1000 * speed;
    if (t >= tEnd) { render(tEnd); playing = false; el("play").textContent = "播放"; return; }
    render(t);
  }
  raf = requestAnimationFrame(frame);
}

function start() {
  base = 0; t0 = performance.now(); playing = true;
  el("play").textContent = "暂停";
  if (raf) cancelAnimationFrame(raf);
  raf = requestAnimationFrame(frame);
}

el("play").onclick = () => {
  if (playing) { playing = false; el("play").textContent = "播放"; }
  else { playing = true; el("play").textContent = "暂停"; t0 = performance.now(); }
};
el("replay").onclick = start;
el("speed").oninput = (e) => {
  speed = Number(e.target.value);
  el("speedLabel").textContent = speed.toFixed(2).replace(/0$/, "") + "×";
  t0 = performance.now();
};

el("subtitle").innerHTML = `数据源：<code>results/des_animation_trace.json</code> · ` +
  `DES run λ=${DATA.summary.offered_rps} req/s · ${DATA.requests.length} requests · ` +
  `Qwen3-4B / RX 9070 XT · ISL 256 / OSL 128`;
el("m-rate").textContent = DATA.summary.offered_rps;
el("m-n").textContent = DATA.requests.length;
el("m-fit").textContent = DATA.fit;
render(0);
start();
</script>
</body>
</html>
'''


def main() -> None:
    payload = json.loads((RESULTS / "des_animation_trace.json").read_text(encoding="utf-8"))
    model = json.loads((RESULTS / "service_model.json").read_text(encoding="utf-8"))
    data = {
        "summary": payload["summary"],
        "trace": payload["trace"],
        "requests": payload["requests"],
        "slo": {"ttft_p99_ms": 800, "tpot_p99_ms": 100, "e2el_p99_ms": 15000},
        "kv_capacity_tokens": 24464,
        "fit": (f"t_prefill={model['prefill']['p0_ms']:.1f}+"
                f"{model['prefill']['p1_ms_per_seq']:.1f}*B ms; "
                f"t_decode={model['decode']['t1_ms']:.2f}*B^{model['decode']['gamma']:.4f} ms"),
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")
    print(f"trace steps={len(data['trace'])}, requests={len(data['requests'])}, "
          f"first_fail_summary={payload['summary']['slo_pass_rate']}")


if __name__ == "__main__":
    main()
