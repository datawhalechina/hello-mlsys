import json,os,statistics,subprocess,time,torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path='/mnt/public/models/Qwen3-1.7B'
out='/mnt/luyuzhou/hello-mlsys/hello-mlsys/runs/course-all-20261120/hw-ch9-1-qwen1.7b'
os.makedirs(out,exist_ok=True)
reqs=json.load(open('examples/agent/task-9.1.protocol.request.json'))['workload']['requests']
def vram():
 s=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True).stdout
 rows={}
 for line in s.splitlines():
  if 'GPU[' in line and 'VRAM Total' in line:
   idx=line.split('GPU[')[1].split(']')[0]; nums=[x for x in line.replace('(B)','').replace(':',' ').split() if x.isdigit()]
   if nums: rows[idx]=int(nums[-1])
 return rows
before=vram(); tok=AutoTokenizer.from_pretrained(model_path); t0=time.time()
model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True).eval()
load=time.time()-t0; after=vram()
rows=[]; torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
 _w=torch.arange(1000,1064,dtype=torch.long,device='cuda:0').remainder(model.config.vocab_size).unsqueeze(0)
 _o=model(input_ids=_w,use_cache=True); _p=_o.past_key_values; _n=_o.logits[:,-1,:].argmax(-1,keepdim=True)
 for _ in range(4): _o=model(input_ids=_n,use_cache=True,past_key_values=_p); _p=_o.past_key_values; _n=_o.logits[:,-1,:].argmax(-1,keepdim=True)
 torch.cuda.synchronize(); del _p,_o
with torch.inference_mode():
 for r in reqs:
  ids=torch.arange(1000,1000+r['input_tokens'],dtype=torch.long,device='cuda:0').remainder(model.config.vocab_size).unsqueeze(0)
  torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
  o=model(input_ids=ids,use_cache=True); past=o.past_key_values; nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
  e.record(); torch.cuda.synchronize(); ttft=s.elapsed_time(e)
  step_times=[]
  for _ in range(r['output_tokens_target']-1):
   s2=torch.cuda.Event(True); e2=torch.cuda.Event(True); s2.record()
   o=model(input_ids=nxt,use_cache=True,past_key_values=past); e2.record(); torch.cuda.synchronize()
   step_times.append(s2.elapsed_time(e2)); past=o.past_key_values; nxt=o.logits[:,-1,:].argmax(-1,keepdim=True)
  rows.append({'request_id':r['request_id'],'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens_target'],
               'ttft_ms':ttft,'tpot_ms':statistics.mean(step_times),'decode_tok_s':1000/statistics.mean(step_times),
               'e2e_ms':ttft+sum(step_times)})
  del past,o
res={'model':'Qwen3-1.7B BF16','device':torch.cuda.get_device_name(0),'load_s':load,'vram_before':before,'vram_after':after,
     'load_delta_bytes':{k:after.get(k,0)-before.get(k,0) for k in after},
     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
     'requests':rows,'hardware_validated':True}
json.dump(res,open(out+'/result.json','w'),indent=2)
print(json.dumps(res,indent=2))
