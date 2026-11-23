/* Decimal units throughout: GFLOP, MB, TFLOP/s, GB/s, ms. */
(function(root){
  function roofline(gflops,mb,peak,bw){const ai=gflops*1000/mb,compute=gflops/peak,memory=mb/bw;return{ai,ridge:peak*1000/bw,compute,memory,time:Math.max(compute,memory),bound:compute>=memory?'compute':'memory',attainable:Math.min(peak,ai*bw/1000)};}
  function batching(batch,ctx){const gemm=roofline(2*596e6*batch/1e9,2*596e6/1e6,195,640);const kv=roofline(4*batch*ctx*2048*28/1e9,4*batch*ctx*1024*28/1e6,195,640);const time=gemm.time+kv.time;return{gemm,kv,time,aggregate:1000*batch/time,perRequest:1000/time};}
  function timeline(dispatch,kernel,count=40){let end=0;const items=[];for(let i=0;i<count;i++){const ready=(i+1)*dispatch,start=Math.max(ready,end);end=start+kernel;items.push({cpuStart:i*dispatch,ready,start,end});}return{items,wall:end,busy:count*kernel,fraction:count*kernel/end};}
  const api={roofline,batching,timeline};root.LabModels=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
