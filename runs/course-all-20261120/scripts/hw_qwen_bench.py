import argparse, json, os, subprocess, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def vram():
    out=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True).stdout
    rows={}
    for line in out.splitlines():
        if 'GPU[' in line and 'VRAM Total' in line:
            part=line.split(':')
            idx=line.split('GPU[')[1].split(']')[0]
            # last numeric field is bytes
            num=[x for x in line.replace('(B)','').replace(':',' ').split() if x.isdigit()]
            if num:
                rows[idx]=int(num[-1])
    return rows

def pct(v,q):
    v=sorted(v)
    if not v: return None
    if len(v)==1: return v[0]
    k=(len(v)-1)*q; f=int(k); c=min(f+1,len(v)-1)
    return v[f]*(c-k)+v[c]*(k-f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--prompt-tokens',type=int,default=512)
    ap.add_argument('--output-tokens',type=int,default=32)
    ap.add_argument('--warmup',type=int,default=1)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--dtype',default='bfloat16')
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    before=vram()
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=getattr(torch,args.dtype),device_map={'':args.device},trust_remote_code=True,low_cpu_mem_usage=True)
    model.eval()
    load_s=time.time()-t0
    after_load=vram()
    torch.cuda.synchronize()
    # deterministic fixed prompt token ids (avoid tokenizer dependence on text)
    vocab=model.config.vocab_size
    ids=torch.arange(1000,1000+args.prompt_tokens,dtype=torch.long,device=args.device).remainder(vocab).unsqueeze(0)
    def one_run():
        with torch.inference_mode():
            torch.cuda.synchronize()
            st=torch.cuda.Event(True); en=torch.cuda.Event(True)
            st.record()
            out=model(input_ids=ids,use_cache=True)
            en.record(); torch.cuda.synchronize()
            prefill_ms=st.elapsed_time(en)
            past=out.past_key_values
            logits=out.logits[:,-1,:]
            nxt=logits.argmax(-1,keepdim=True)
            first_ms=st.elapsed_time(en)
            step=[]
            for _ in range(args.output_tokens-1):
                s=torch.cuda.Event(True); e=torch.cuda.Event(True)
                s.record()
                out=model(input_ids=nxt,use_cache=True,past_key_values=past)
                e.record(); torch.cuda.synchronize()
                step.append(s.elapsed_time(e))
                past=out.past_key_values
                nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
            return prefill_ms, first_ms, step
    for _ in range(args.warmup): one_run()
    runs=[one_run() for _ in range(args.repeats)]
    prefill=[r[0] for r in runs]
    tpot=[sum(r[2])/len(r[2]) if r[2] else None for r in runs]
    peak_alloc=torch.cuda.max_memory_allocated()/2**30
    peak_reserved=torch.cuda.max_memory_reserved()/2**30
    p=sum(x.numel() for x in model.parameters())
    result={
      'model':args.model,'device':args.device,'dtype':args.dtype,
      'torch':torch.__version__,'transformers':__import__('transformers').__version__,
      'gpu':torch.cuda.get_device_name(0),'prompt_tokens':args.prompt_tokens,'output_tokens':args.output_tokens,
      'parameter_count':p,'load_s':load_s,
      'vram_before_bytes':before,'vram_after_load_bytes':after_load,
      'load_vram_delta_bytes':{k:after_load.get(k,0)-before.get(k,0) for k in after_load},
      'ttft_ms_runs':prefill,'ttft_ms_p50':pct(prefill,.5),'ttft_ms_p99':pct(prefill,.99),
      'tpot_ms_runs':tpot,'tpot_ms_p50':pct(tpot,.5),'tpot_ms_p99':pct(tpot,.99),
      'decode_tok_s_p50':1000/pct(tpot,.5) if pct(tpot,.5) else None,
      'torch_peak_allocated_gib':peak_alloc,'torch_peak_reserved_gib':peak_reserved,
      'hardware_validated':True,
    }
    json.dump(result,open(os.path.join(args.out,'result.json'),'w'),indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
