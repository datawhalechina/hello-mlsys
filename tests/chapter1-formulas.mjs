import assert from 'node:assert/strict';
await import('../docs/public/demos/chapter1/formula-models.js');
const M = globalThis.FormulaModels;
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-8, `${a} != ${b}`);
near(M.attention(512).faAI, 512*2/3);
near(M.attention(512).naiveAI, 93.0909090909091);
near(M.attention(512).limit, 128);
assert.ok(M.attention(1e8).naiveAI<128 && M.attention(1e8).naiveAI>127.99);
near(M.kv(1,1024).ai,2); near(M.kv(32,8192).ai,2);
near(M.kv(4,1024,8).ai,8);
assert.ok(M.prefill(128).memory>M.prefill(128).compute);
assert.ok(M.prefill(512).compute>M.prefill(512).memory);
near(M.composition(8,2,2,8).correct,16);
near(M.composition(8,2,2,8).wrong,10);
near(M.composition(8,2,8,2).correct,M.composition(8,2,8,2).wrong);
for(let i=1;i<=12;i++) for(let j=1;j<=12;j++){
 const r=M.composition(i,j,13-i,13-j);assert.ok(r.correct>=r.wrong);
}
near(M.cost(2e6*8,2e6*4,48.7,640).ai,2);
console.log('Formula model checks passed: attention limits, GQA cancellation, prefill crossover, component inequality and BF16 byte accounting.');
