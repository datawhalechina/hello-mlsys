(function(root){
 const peak=195,bw=640;
 const cost=(f,bytes,p=peak,b=bw)=>({compute:f/(p*1e12)*1000,memory:bytes/(b*1e9)*1000,ai:f/bytes});
 function prefill(s){const f=2*440e6*s,bytes=880e6+.8e6*s,c=cost(f,bytes);return{...c,f,bytes,time:Math.max(c.compute,c.memory)};}
 function attention(s){const f=4*s*s*2048*28,linear=2*s*(2*2048+2*1024)*28,quadratic=4*16*s*s*28;return{f,linear,quadratic,faAI:f/linear,naiveAI:f/(linear+quadratic),fa:cost(f,linear),naive:cost(f,linear+quadratic),limit:128};}
 function kv(b,s,ratio=2){const dq=2048,dk=dq/ratio,f=4*b*s*dq*28,bytes=4*b*s*dk*28;return{...cost(f,bytes),f,bytes};}
 function composition(c1,m1,c2,m2,fixed=0){return{correct:Math.max(c1,m1)+Math.max(c2,m2)+fixed,wrong:Math.max(c1+c2,m1+m2)+fixed};}
 const api={cost,prefill,attention,kv,composition};root.FormulaModels=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
