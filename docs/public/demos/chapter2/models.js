(function(root){
function calibrated(f,bytes,eta,beta){const c=f/(195e12*eta)*1000,m=bytes/(640e9*beta)*1000;return{compute:c,memory:m,time:Math.max(c,m),ridge:195e3*eta/(640*beta),bound:c>=m?'compute':'memory'};}
function fit(points,ids){const chosen=points.filter(p=>ids.includes(p.id));if(!chosen.length)return null;return chosen.reduce((s,p)=>s+p.nominal,0)/chosen.reduce((s,p)=>s+p.measured,0);}
function cache(bytes,beta,hit){const traffic=bytes*(1-hit);return{traffic,time:traffic/(640e9*beta)*1000,apparent:beta/(1-hit)};}
function overhead(n,gpu,dispatch,tax,cold,warmup){const clean=dispatch+n*Math.max(gpu,dispatch)+(gpu>dispatch?0:gpu-dispatch);const profile=(dispatch+tax)+n*Math.max(gpu,dispatch+tax)+(gpu>dispatch+tax?0:gpu-dispatch-tax);return{clean,profile,first:profile+(warmup?0:cold),inflation:profile/clean};}
function error(predicted,measured){return(measured-predicted)/predicted*100;}
function timing(row,mode){const wall=mode==='clean'?row.clean:row.profile;return{wall,self:row.self,difference:wall-row.self,sameCondition:mode!=='clean',ratio:row.self/wall};}
function nearest(rows,time){return rows.reduce((a,b)=>Math.abs(a.t_offset_s-time)<=Math.abs(b.t_offset_s-time)?a:b);}
const api={calibrated,fit,cache,overhead,error,timing,nearest};root.Ch2Models=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
