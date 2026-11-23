from pathlib import Path
import re,json,html
import markdown
ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'docs/public/demos/chapter1'
SOURCE=ROOT/'docs/part1-modeling/chapter1/index.md'
def build():
 source=SOURCE.read_text(encoding='utf-8')
 scenes=json.loads((HERE/'formula-scenes.json').read_text(encoding='utf-8'))
 by_id={s['id']:s for s in scenes}
 source=re.sub(r'\A---\n.*?\n---\n','',source,flags=re.S)
 source=re.sub(r'::: danger (.*?)\n(.*?)\n:::',lambda m:'> **'+m[1]+'**\n> '+m[2],source,flags=re.S)
 def embed(m):
  key=m[1];scene=by_id[key]
  return f'\n<section class="formula-lab" id="formula-{key}" data-formula="{scene["kind"]}" aria-label="{html.escape(scene["title"])}"></section>\n'
 source=re.sub(r'<!-- formula-demo:start:([\w-]+) -->.*?<!-- formula-demo:end:\1 -->',embed,source,flags=re.S)
 source=re.sub(r'\*\*\[打开整章公式动画阅读版\].*?\n\n','',source,flags=re.S)
 source=source.replace('https://datawhalechina.github.io/hello-mlsys/demos/chapter1/', './')
 md=markdown.Markdown(extensions=['tables','fenced_code','toc','sane_lists'])
 content=md.convert(source).replace('src="./demos/chapter1/','src="./').replace('href="./demos/chapter1/','href="./')
 content=re.sub(r'href="/(?!/)([^"]*)"', lambda m: 'href="https://datawhalechina.github.io/hello-mlsys/'+m[1]+'"', content)
 def image(m):
  attrs=m[0];rel=m[1]
  if not (SOURCE.parent/rel).exists():
   alt=re.search(r'alt="([^"]*)"',attrs);label=alt[1] if alt else rel
   return '<div class="missing-figure">原课件配图在当前材料目录中缺失：'+html.escape(label)+'。保留正文说明与实测表；此处不重建实测图。</div>'
  return attrs
 content=re.sub(r'<img[^>]*src="(\./images/[^"]+)"[^>]*>',image,content)
 nav='<h3>CHAPTER 01 / 阅读导航</h3>'+''.join(f'<a href="#{i}">{html.unescape(re.sub("<[^>]+>","",name))}</a>' for i,name in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>',content))
 nav+='<h3>FORMULA / 13 处公式</h3>'+''.join(f'<a href="#formula-{s["id"]}">{s["title"]}</a>' for s in scenes)
 banner='<div class="reader-banner"><b>公式下方，直接动手。</b>点击公式项查看解释，拖动参数联动图形，播放或单步观察过程。动画为解析模型与机制示意；无自动播放。此页面由 index.md 生成。</div>'
 head='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 1 章 · 公式动画阅读版 | Hello MLSys</title><link rel="stylesheet" href="formula.css"><script src="formula-models.js" defer></script><script src="formula.js" defer></script></head><body id="top"><header><a href="reader.html">HELLO MLSYS / 公式动画阅读版</a><small>CHAPTER 01 · 13 处交互 · 离线可用</small><a href="index.html">三个综合实验 ↗</a></header><div class="layout"><aside>'
 foot='<footer>参考交互：<a href="https://datawhalechina.github.io/hello-gpu/atlas/amd/?architecture=cdna4#explore">GPU Atlas</a> · <a href="https://datawhalechina.github.io/dive-into-embodied-ai/cs123/pd-playground">PD Playground</a> · <a href="https://datawhalechina.github.io/hello-rocm/zh/">hello-rocm</a>。本页为独立实现，未复制参考站资源或代码。</footer></article></div><a class="backtop" href="#top">回到顶部 ↑</a></body></html>'
 page=head+nav+'</aside><article>'+banner+content+foot
 (HERE/'reader.html').write_text(page,encoding='utf-8')
 print('Built reader.html:',len(scenes),'inline formula placements.')
 return page
if __name__=='__main__':build()
