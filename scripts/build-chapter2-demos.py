from pathlib import Path
import json,re,html,shutil
import markdown
ROOT=Path(__file__).resolve().parent.parent
PUBLIC=ROOT/'docs/public/demos/chapter2'
SOURCE=ROOT/'docs/part1-modeling/chapter2/index.md'
LOCAL=ROOT.parent/'hello-mlsys/demos/chapter2'
def snapshot():
 r=ROOT/'code/part1-modeling/chapter2/results'
 load=lambda name:json.loads((r/name).read_text(encoding='utf-8'))
 cal=load('cal_9070xt.json');clocks=load('clock_drift_9070xt.json');e2e=load('e2e_1p7b_9070xt.json');cpu=load('cpu_refline_9070xt.json');minimal=load('minimal_profile_9070xt.json')
 points=[]
 for group,rows in cal['calibration'].items():
  for shape,row in rows.items():points.append({'id':group+'_'+shape,'name':{'gemm':'GEMM','ln':'LayerNorm','softmax':'Softmax'}[group]+' '+shape.replace('^3','³').replace('x','×'),'side':'eta' if group=='gemm' else 'beta','nominal':row['t_comp_ms'] if group=='gemm' else row['t_mem_ms'],'measured':row['median_ms'],'selected':row['fit'],'coefficient':row.get('eta_point',row.get('beta_point'))})
 holdout=[{'id':key,'name':key.replace('_',' '),'predicted':row['t_pred_cal_ms'],'measured':row['median_ms'],'domain':'域外' if key.startswith('attn') else '域内'} for key,row in cal['holdout_measured'].items()]
 measured={key:{'prediction':e2e['prediction'][key]['total_ms'],'clean':val['wall_ms'],'profile':val['profile']['wall_inside_profile_ms'],'self':val['profile']['gpu_self_ms'],'kernels':val['profile']['kernels']} for key,val in e2e['measured'].items()}
 data={'points':points,'fit':{'eta':cal['fit']['eta'],'beta':cal['fit']['beta'],'stored_at':cal['fit']['stored_at']},'holdout':holdout,'clocks':clocks['clock_series'],'clockBlocks':clocks['block_medians'],'fresh':clocks['fresh'],'hot':clocks['hot'],'e2e':measured,'cpu':cpu['summary'],'minimal':{'clean':minimal['A_event_sync_median_ms'],'v3':minimal['C_rocprofv3']['event_ms_inside'],'torch':minimal['D_torch_profiler']['event_ms_inside_profile']},'sources':['cal_9070xt.json','clock_drift_9070xt.json','e2e_1p7b_9070xt.json','cpu_refline_9070xt.json','minimal_profile_9070xt.json']}
 (PUBLIC/'data.js').write_text('globalThis.Ch2Data = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
def build():
 snapshot();scenes=json.loads((PUBLIC/'scenes.json').read_text(encoding='utf-8'))
 source=SOURCE.read_text(encoding='utf-8')
 source=re.sub(r'\A---\n.*?\n---\n','',source,flags=re.S)
 source=re.sub(r'::: danger (.*?)\n(.*?)\n:::',lambda m:'> **'+m[1]+'**\n> '+m[2],source,flags=re.S)
 source=re.sub(r'<!-- ch2-intro:start -->.*?<!-- ch2-intro:end -->','',source,flags=re.S)
 def scene(m):
  key=m[1]
  assert key in {e['id'] for e in scenes}
  return f'\n<section class="lab" id="scene-{key}" data-scene="{key}"></section>\n'
 source=re.sub(r'<!-- ch2-demo:start:([\w-]+) -->.*?<!-- ch2-demo:end:\1 -->',scene,source,flags=re.S)
 renderer=markdown.Markdown(extensions=['tables','fenced_code','toc','sane_lists'])
 content=renderer.convert(source)
 content=re.sub(r'href="/(?!/)([^"]*)"', lambda m: 'href="https://datawhalechina.github.io/hello-mlsys/'+m[1]+'"', content)
 nav='<h3>CHAPTER 02 / 阅读导航</h3>'+''.join('<a href="#'+i+'">'+html.unescape(re.sub('<[^>]+>','',t))+'</a>' for i,t in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',content))
 nav+='<h3>INTERACTIVE / 交互实验</h3>'+''.join(f'<a href="#scene-{s["id"]}">{s["title"]}</a>' for s in scenes)
 banner='<div class="chapter-intro"><small>CALIBRATION STUDIO / CHAPTER 02</small><b>把预测带上真机，再把证据带回来。</b><p>八组交互解释测量、校准与验证。实测数据与受控示意分别标注；不自动播放，不需要 GPU。</p></div>'
 head='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 2 章 · 校准实验室 | Hello MLSys</title><link rel="stylesheet" href="style.css"><script src="data.js" defer></script><script src="models.js" defer></script><script src="app.js" defer></script></head><body id="top"><header><a href="#top">HELLO MLSYS / 校准实验室</a><small>CHAPTER 02 · 8 组交互 · 离线可用</small><a href="../chapter1/reader.html">第 1 章演示 ↗</a></header><div class="layout"><aside>'
 foot='<footer>交互设计参考：<a href="https://datawhalechina.github.io/hello-gpu/atlas/amd/">GPU Atlas</a> · <a href="https://datawhalechina.github.io/dive-into-embodied-ai/cs123/pd-playground">PD Playground</a> · <a href="https://datawhalechina.github.io/hello-rocm/zh/">hello-rocm</a>。图形和代码为本课件独立实现。</footer></article></div><a href="#top" class="backtop">回到顶部 ↑</a></body></html>'
 page=head+nav+'</aside><article>'+banner+content+foot
 (PUBLIC/'index.html').write_text(page,encoding='utf-8')
 shutil.copytree(SOURCE.parent/'images',PUBLIC/'images',dirs_exist_ok=True)
 # Copy only the demonstration resources into the user's original material folder.
 shutil.copytree(PUBLIC,LOCAL,dirs_exist_ok=True)
 (LOCAL/'打开第2章动画.cmd').write_bytes(b'@echo off\r\nstart "" "%~dp0index.html"\r\n')
 print('Generated Chapter 2 reader, source data and local preview:',len(scenes),'scenes.')
if __name__=='__main__':build()
