import json,os,statistics,torch
out='/mnt/luyuzhou/hello-mlsys/hello-mlsys/runs/course-all-20261120/hw-ch2-calibration'
os.makedirs(out,exist_ok=True)
dev='cuda:0'
def timeit(fn,warmup=5,iters=20):
 for _ in range(warmup): fn()
 torch.cuda.synchronize()
 ts=[]
 for _ in range(iters):
  s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
 return statistics.median(ts),ts
res={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'gemm':[],'bandwidth':[],'launch':None}
peak_tflops=195.0
for n in [2048,4096,5120,8192]:
 a=torch.randn(n,n,device=dev,dtype=torch.bfloat16); b=torch.randn(n,n,device=dev,dtype=torch.bfloat16)
 med,_=timeit(lambda: a@b,warmup=3,iters=10)
 flops=2*n**3
 tflops=flops/(med*1e-3)/1e12
 res['gemm'].append({'n':n,'median_ms':med,'tflops':tflops,'ratio_to_195':tflops/peak_tflops})
 del a,b; torch.cuda.empty_cache()
# streaming bandwidth: read one array, write one array (copy)
for mb in [64,256,1024,2048]:
 n=mb*1024*1024//2
 a=torch.empty(n,device=dev,dtype=torch.bfloat16); b=torch.empty_like(a)
 med,_=timeit(lambda: b.copy_(a),warmup=3,iters=10)
 gbps=(2*n*2)/(med*1e-3)/1e9
 res['bandwidth'].append({'size_mib':mb,'median_ms':med,'effective_gbps':gbps})
 del a,b; torch.cuda.empty_cache()
# launch overhead: tiny elementwise op
x=torch.empty(1,device=dev); y=torch.empty(1,device=dev)
med,_=timeit(lambda: y.copy_(x),warmup=20,iters=100)
res['launch']={'median_ms':med,'us':med*1000}
json.dump(res,open(out+'/result.json','w'),indent=2)
print(json.dumps(res,indent=2))
