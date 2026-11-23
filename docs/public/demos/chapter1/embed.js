(()=>{
const scenes={"max": "max", "roof": "roof", "ridge": "ridge", "compose": "compose", "ai": "ai", "reuse": "reuse", "prefill": "prefill", "attention": "attention", "flash": "flash", "kv": "kv", "vector": "vector", "compose-step": "compose", "inequality": "compose"};
const key=new URLSearchParams(location.search).get('scene')||'max';
const host=document.getElementById('scene');
if(!Object.prototype.hasOwnProperty.call(scenes,key)){host.textContent='未知动画，请返回章节重新选择。';return;}
const section=document.createElement('section');
section.className='formula-lab';section.id='formula-'+key;section.dataset.formula=scenes[key];
host.append(section);
let pending=false;
function report(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;const height=Math.ceil(host.getBoundingClientRect().height)+2;parent.postMessage({type:'hello-mlsys:formula-height',scene:key,height},location.origin);});}
new ResizeObserver(report).observe(host);
window.addEventListener('load',report);
})();
