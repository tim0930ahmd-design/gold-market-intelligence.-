// Advanced AI layers for the Gold PAPER engine.
// All model artifacts are optional. Missing/unvalidated artifacts produce COLLECTING_DATA,
// never fabricated probabilities or automatic entries.
export type Bar={time:string;open:number;high:number;low:number;close:number;volume?:number};
const clamp=(x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const mean=(x:number[])=>x.length?x.reduce((a,b)=>a+b,0)/x.length:0;
const sigmoid=(x:number)=>1/(1+Math.exp(-clamp(x,-30,30)));
const atr=(b:Bar[],p=14)=>{if(b.length<p+1)return 0;const tr=b.slice(1).map((x,i)=>Math.max(x.high-x.low,Math.abs(x.high-b[i].close),Math.abs(x.low-b[i].close)));return mean(tr.slice(-p));};
export const DISCOVERY_FEATURES=[
 "ret1","ret3","ret6","ret12","rangeAtr","bodyAtr","upperWickAtr","lowerWickAtr",
 "efficiency12","efficiency24","atrRatio","closePosition20","slope6","slope18",
 "volatilityRatio","hourSin","hourCos","isBuyContext","isSellContext","regimeTrend",
 "regimeRange","regimeCompression","regimeExpansion","regimeImpulse"
];
export function marketVector(bars:Bar[],regime:string,atTime?:string):number[]|null {
 const b=bars.slice(-60);if(b.length<30)return null;
 const c=b.map(x=>x.close),last=b.at(-1)!,a=atr(b,14),a50=atr(b,50)||a;
 if(!a||!a50)return null;
 const ret=(k:number)=>c.at(-1)!/c.at(-1-k)!-1;
 const path=(k:number)=>b.slice(-k+1).reduce((s,x,i)=>s+Math.abs(x.close-b.slice(-k+1)[i-1]?.close||0),0);
 const eff=(k:number)=>{const p=b.slice(-k);const travel=p.slice(1).reduce((s,x,i)=>s+Math.abs(x.close-p[i].close),0);return travel?Math.abs(p.at(-1)!.close-p[0].close)/travel:0;};
 const hi=Math.max(...b.slice(-20).map(x=>x.high)),lo=Math.min(...b.slice(-20).map(x=>x.low));
 const body=Math.abs(last.close-last.open),upper=last.high-Math.max(last.open,last.close),lower=Math.min(last.open,last.close)-last.low;
 const d=new Date(atTime||last.time),hour=d.getUTCHours()+d.getUTCMinutes()/60,ang=2*Math.PI*hour/24;
 const vals=[ret(1),ret(3),ret(6),ret(12),(last.high-last.low)/a,body/a,upper/a,lower/a,eff(12),eff(24),a/a50,(last.close-lo)/Math.max(a,hi-lo),(c.at(-1)!-c.at(-7)!)/a,(c.at(-1)!-c.at(-19)!)/a,
  mean(b.slice(-6).map(x=>x.high-x.low))/Math.max(1e-9,mean(b.slice(-30).map(x=>x.high-x.low))),Math.sin(ang),Math.cos(ang),Math.max(0,ret(6)),Math.max(0,-ret(6)),regime.startsWith("TREND")?1:0,regime==="RANGE"?1:0,regime==="COMPRESSION"?1:0,regime==="VOLATILITY_EXPANSION"?1:0,regime==="IMPULSE"?1:0];
 return vals.map(x=>Number.isFinite(x)?x:0);
}
function linearScore(model:any,x:number[]) {
 if(!model||!Array.isArray(model.weights)||model.weights.length!==x.length)return null;
 return sigmoid(Number(model.bias||0)+model.weights.reduce((s:number,w:number,i:number)=>{const z=(x[i]-Number(model.mean?.[i]||0))/Math.max(1e-9,Number(model.scale?.[i]||1));return s+Number(w||0)*z;},0));
}
export function discoverOpportunities(model:any,x:number[],entry:number,atrValue:number,barTime:string,regime:string) {
 if(!model||model.kind!=="gold-opportunity-linear-v1"||model.validated!==true)return {status:"MODEL_NOT_VALIDATED",candidates:[] as any[]};
 const out:any[]=[];
 for(const side of ["BUY","SELL"]) {
  const m=model.sides?.[side],p=m?linearScore(m,x):null;
  if(p===null||p<Number(model.thresholds?.[side]??1.01))continue;
  const stopDistance=clamp(Number(model.stopAtr||0.75)*atrValue,0.25*atrValue,2.5*atrValue);
  const stop=side==="BUY"?entry-stopDistance:entry+stopDistance,target=side==="BUY"?entry+stopDistance*Number(model.rewardRisk||1.8):entry-stopDistance*Number(model.rewardRisk||1.8);
  out.push({id:[barTime,"AI_DISCOVERY",side].join("|"),expert:"AI_DISCOVERY",direction:side,entry,stopLoss:Number(stop.toFixed(2)),target:Number(target.toFixed(2)),risk:Number(stopDistance.toFixed(3)),rr:Number(model.rewardRisk||1.8),stopAtr:Number((stopDistance/atrValue).toFixed(2)),qualityScore:Number((p*100).toFixed(1)),barTime,regime,features:{discoveryProbability:p,modelVersion:model.modelVersion},discoveryProbability:p,discoveryModelVersion:model.modelVersion});
 }
 return {status:"ACTIVE_VALIDATED_MODEL",candidates:out};
}
export function predictScenarios(model:any,x:number[]) {
 if(!model||model.kind!=="gold-scenario-softmax-v1"||model.validated!==true)return {status:"MODEL_NOT_VALIDATED",horizons:{} as any};
 const horizons:any={};
 for(const [h,classes] of Object.entries(model.horizons||{})) {
  const logits:any={};for(const side of ["UP","DOWN","RANGE"]) {
   const m=(classes as any)[side];if(!m||!Array.isArray(m.weights)||m.weights.length!==x.length)continue;
   logits[side]=Number(m.bias||0)+m.weights.reduce((s:number,w:number,i:number)=>{const z=(x[i]-Number(m.mean?.[i]||0))/Math.max(1e-9,Number(m.scale?.[i]||1));return s+Number(w||0)*z;},0);
  }
  if(Object.keys(logits).length!==3)continue;
  const mx=Math.max(...Object.values(logits) as number[]),ex:any={};for(const k of Object.keys(logits))ex[k]=Math.exp(clamp(logits[k]-mx,-30,30));
  const z=Object.values(ex).reduce((a:any,b:any)=>a+b,0) as number;
  horizons[h]={pUp:ex.UP/z,pDown:ex.DOWN/z,pRange:ex.RANGE/z};
 }
 return {status:Object.keys(horizons).length?"ACTIVE_VALIDATED_MODEL":"INVALID_ARTIFACT",modelVersion:model.modelVersion||null,horizons};
}
export function learnExpertWeights(closed:any[],regime:string) {
 const out:any={};
 for(const expert of [...new Set(closed.map(x=>x.expert))]) {
  const rows=closed.filter(x=>x.expert===expert&&(!x.regime||x.regime===regime)).slice(-80);
  if(rows.length<12) {out[expert]={weight:1,samples:rows.length,status:"INSUFFICIENT_HISTORY"};continue;}
  const wins=rows.filter(x=>Number(x.netUsd)>0).length,alpha=1+wins,beta=1+rows.length-wins;
  const posterior=alpha/(alpha+beta),avgR=mean(rows.map(x=>Number(x.resultR)||0));
  out[expert]={weight:Number(clamp(0.55+posterior*0.9+avgR*0.12,0.5,1.5).toFixed(3)),samples:rows.length,posteriorWinRate:Number(posterior.toFixed(3)),avgR:Number(avgR.toFixed(3)),status:"LEARNED_FROM_PAPER_OUTCOMES"};
 }
 return out;
}
export function applyExpertWeights(candidates:any[],weights:any) {
 return candidates.map(c=>{const w=weights[c.expert]?.weight??1;return {...c,metaLearningWeight:w,metaLearningScore:Number((c.qualityScore*w).toFixed(2))};}).sort((a,b)=>b.metaLearningScore-a.metaLearningScore);
}
export function decisionIntelligence(c:any,prob:any,cost:number,lotOz:number,stressCost:number) {
 if(!prob)return {decision:"NO_TRADE",reason:"NO_VALIDATED_PROBABILITY_MODEL",evUsd:null,stressEvUsd:null};
 const pT=Number(prob.pTarget),pS=Number(prob.pStop),pX=Number(prob.pTimeout||0);
 if(![pT,pS,pX].every(Number.isFinite)||Math.min(pT,pS,pX)<0)return {decision:"NO_TRADE",reason:"INVALID_PROBABILITIES",evUsd:null,stressEvUsd:null};
 const reward=Math.abs(c.target-c.entry)*lotOz,risk=Math.abs(c.entry-c.stopLoss)*lotOz;
 const ev=pT*reward-pS*risk-cost,stress=pT*reward-pS*risk-stressCost;
 return {decision:ev>0&&stress>0?"PAPER_ELIGIBLE":"NO_TRADE",reason:ev>0&&stress>0?"POSITIVE_EV_BASE_AND_STRESS":"NON_POSITIVE_EXPECTED_VALUE",evUsd:Number(ev.toFixed(3)),stressEvUsd:Number(stress.toFixed(3)),pTarget:pT,pStop:pS,pTimeout:pX};
}

/** Resolve strictly forward-only labels and append the current completed-bar snapshot. */
export function collectForwardLabels(state:any,bars:Bar[],regime:string,barTime:string) {
 state.aiLearningPending=Array.isArray(state.aiLearningPending)?state.aiLearningPending:[];
 state.aiLearningSamples=Array.isArray(state.aiLearningSamples)?state.aiLearningSamples:[];
 const current=bars.findIndex(b=>b.time===barTime);
 if(current<0||current<30)return {added:0,resolved:0,pending:state.aiLearningPending.length,samples:state.aiLearningSamples.length};
 let resolved=0;const keep:any[]=[];
 for(const p of state.aiLearningPending) {
  const idx=bars.findIndex(b=>b.time===p.barTime);if(idx<0)continue;const age=current-idx;
  if(age<12){keep.push(p);continue;}
  const future=bars.slice(idx+1,idx+13),outcomes:any={};
  for(const h of [3,6,12]) {
   const end=bars[idx+h];if(!end)continue;
   const move=(end.close-p.entry)/p.atr;
   outcomes[String(h)]=move>0.35?"UP":move< -0.35?"DOWN":"RANGE";
  }
  const discovery:any={};
  for(const side of ["BUY","SELL"]) {
   const target=side==="BUY"?p.entry+p.atr: p.entry-p.atr;
   const stop=side==="BUY"?p.entry-p.atr*0.75:p.entry+p.atr*0.75;
   let label="TIMEOUT";
   for(const b of future) {
    const hitStop=side==="BUY"?b.low<=stop:b.high>=stop;
    const hitTarget=side==="BUY"?b.high>=target:b.low<=target;
    if(hitStop){label="STOP";break;} if(hitTarget){label="TARGET";break;}
   }
   discovery[side]=label;
  }
  state.aiLearningSamples.push({barTime:p.barTime,entry:p.entry,atr:p.atr,features:p.features,regime:p.regime,discovery,scenarios:outcomes});
  resolved++;
 }
 state.aiLearningPending=keep;
 let added=0;
 if(!state.aiLearningPending.some((p:any)=>p.barTime===barTime)) {
  const x=marketVector(bars,regime,barTime),a=atr(bars,14),entry=bars[current].close;
  if(x&&a>0){state.aiLearningPending.push({barTime,entry,atr:a,features:x,regime});added++;}
 }
 if(state.aiLearningPending.length>500)state.aiLearningPending=state.aiLearningPending.slice(-500);
 if(state.aiLearningSamples.length>12000)state.aiLearningSamples=state.aiLearningSamples.slice(-12000);
 return {added,resolved,pending:state.aiLearningPending.length,samples:state.aiLearningSamples.length};
}
