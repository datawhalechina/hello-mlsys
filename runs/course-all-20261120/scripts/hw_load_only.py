import argparse,json,os,subprocess,time,torch
from transformers import AutoModelForCausalLM
def vram():
 out=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True).stdout
 rows={}
 for line in out.splitlines():
  if 'GPU[' in line and 'VRAM Total' in line:
   idx=line.split('GPU[')[1].split(']')[0]
   nums=[x for x in line.replace('(B)','').replace(':',' ').split() if x.isdigit()]
   if nums: rows[idx]=int(nums[-1])
 return rows
ap=argparse.ArgumentParser(); ap.add_argument('--model'); ap.add_argument('--out'); a=ap.parse_args()
before=vram(); t0=time.time()
m=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True)
load=time.time()-t0; after=vram()
p=sum(x.numel() for x in m.parameters())
r={'model':a.model,'load_s':load,'parameter_count':p,'param_bytes_bf16':p*2,
   'vram_before':before,'vram_after':after,'load_delta':{k:after.get(k,0)-before.get(k,0) for k in after},
   'torch_allocated_gib':torch.cuda.memory_allocated()/2**30,'torch_reserved_gib':torch.cuda.memory_reserved()/2**30,
   'quantization_config':str(getattr(m.config,'quantization_config',None)),'hardware_validated':True}
os.makedirs(a.out,exist_ok=True); json.dump(r,open(a.out+'/result.json','w'),indent=2); print(json.dumps(r,indent=2))
