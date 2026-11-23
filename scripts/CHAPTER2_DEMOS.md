# Chapter 2 interactive demonstrations

Extend the accepted Chapter 1 workflow: inline animations in VitePress, linked previews in GitHub Markdown, and an offline whole-chapter reader. Preserve the existing Chapter 1 label edit. No commit, push or deployment is included in this task.

Eight scenes: calibrated resource model; profiler overhead and warmup; weighted fitting with selectable calibration points; cache traffic and apparent beta; actual clock/power traces; frozen holdout errors; clean versus profiled timing; hardware and engine comparisons.

Use existing results JSON as the source for recorded observations. Mark synthetic controls and animations explicitly. Do not tune against holdouts, turn sums of self times into measured idle time, or infer unmeasured CPU torch performance by multiplying speedups. Keep a Chapter 1 white/blue interface with mint/orange chart accents with nearby controls, readable units and keyboard operation.

Verification: coefficient anchors, no-selection fit, holdout arithmetic, trace endpoints, 8 scenes in Markdown and reader, VitePress build, iframe interaction and responsive UI. Regenerate with scripts/build-chapter2-demos.py; run node tests/chapter2-demos.mjs.

Verified: source-data/model checks and complete VitePress build passed. 8 scene controls checked, including all/none calibration selection, frozen out-of-domain error, mixed timing conditions and missing engine baseline. 26 range endpoints and 17 select cases produced valid SVGs. Playback completed at 100%. 390px viewport had 375px content/client width and no horizontal overflow. Browser error log empty. Built Chapter 1/2 iframe counts and base paths checked. Existing Chapter 1 label edit preserved; no commit or push performed.
