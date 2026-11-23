export type Candle = { time:string; open:number; high:number; low:number; close:number; volume?:number };
export type RegimeResult = {
  regime:"TREND_UP"|"TREND_DOWN"|"RANGE"|"COMPRESSION"|"VOLATILITY_EXPANSION"|"IMPULSE"|"TRANSITION";
  confidence:number; changed:boolean; features:Record<string,number>;
};
const mean=(a:number[])=>a.length?a.reduce((x,y)=>x+y,0)/a.length:0;
const ema=(a:number[],p:number)=>{if(a.length<p)return null;let v=mean(a.slice(0,p));const k=2/(p+1);for(const x of a.slice(p))v=x*k+v*(1-k);return v;};
const clamp=(x:number,a:number,b:number)=>Math.max(a,Math.min(b,x));
const atr=(b:Candle[],p:number)=>{if(b.length<p+1)return null;const tr=b.slice(1).map((x,i)=>Math.max(x.high-x.low,Math.abs(x.high-b[i].close),Math.abs(x.low-b[i].close)));return mean(tr.slice(-p));};
export function classifyMarketRegime(bars:Candle[],previousRegime?:string|null):RegimeResult {
  const b=bars.slice(-100),closes=b.map(x=>x.close),a14=atr(b,14),a50=atr(b,50);
  if(b.length<55||!a14||!a50)return {regime:"TRANSITION",confidence:0,changed:previousRegime!== "TRANSITION",features:{bars:b.length}};
  const fast=ema(closes,9)!,slow=ema(closes,21)!,prior=ema(closes.slice(0,-5),21)!;
  const path=b.slice(-20).slice(1).reduce((s,x,i)=>s+Math.abs(x.close-b.slice(-20)[i].close),0);
  const efficiency=path?Math.abs(closes.at(-1)!-closes.at(-21)!)/path:0;
  const atrRatio=a14/a50,body=Math.abs(b.at(-1)!.close-b.at(-1)!.open)/a14;
  const slope=(slow-prior)/a14,alignment=(fast-slow)/a14;
  const width=(Math.max(...b.slice(-20).map(x=>x.high))-Math.min(...b.slice(-20).map(x=>x.low)))/a14;
  let regime:RegimeResult["regime"]="TRANSITION",confidence=0;
  if(atrRatio<0.72){regime="COMPRESSION";confidence=clamp((0.9-atrRatio)*2.2,0.45,0.95);}
  else if(atrRatio>1.35){regime="VOLATILITY_EXPANSION";confidence=clamp((atrRatio-1.1)*1.4,0.45,0.95);}
  else if(body>1.25){regime="IMPULSE";confidence=clamp(0.55+(body-1.25)*0.22,0.55,0.95);}
  else if(efficiency>0.42&&alignment>0.08&&slope>0.025){regime="TREND_UP";confidence=clamp(0.48+efficiency*0.45+Math.abs(alignment)*0.18,0.5,0.96);}
  else if(efficiency>0.42&&alignment< -0.08&&slope< -0.025){regime="TREND_DOWN";confidence=clamp(0.48+efficiency*0.45+Math.abs(alignment)*0.18,0.5,0.96);}
  else if(efficiency<0.25&&width<8){regime="RANGE";confidence=clamp(0.58+(0.25-efficiency)*1.1,0.5,0.9);}
  else {regime="TRANSITION";confidence=clamp(0.55+Math.abs(efficiency-0.32)*0.25,0.45,0.72);}
  return {regime,confidence:Number(confidence.toFixed(3)),changed:!!previousRegime&&previousRegime!==regime,features:{atr14:a14,atr50:a50,atrRatio,efficiency,bodyAtr:body,emaAlignment:alignment,slope,width}};
}
const EXPERTS:Record<string,string[]> = {
  TREND_UP:["TREND_PULLBACK","BREAKOUT_CONTINUATION","MOMENTUM_IMPULSE","GEMINI_VISION"],
  TREND_DOWN:["TREND_PULLBACK","BREAKOUT_CONTINUATION","MOMENTUM_IMPULSE","GEMINI_VISION"],
  RANGE:["RANGE_REJECTION","LIQUIDITY_SWEEP_RECLAIM","GEMINI_VISION"],
  COMPRESSION:["BREAKOUT_CONTINUATION","LIQUIDITY_SWEEP_RECLAIM"],
  VOLATILITY_EXPANSION:["MOMENTUM_IMPULSE","BREAKOUT_CONTINUATION","LIQUIDITY_SWEEP_RECLAIM"],
  IMPULSE:["MOMENTUM_IMPULSE","BREAKOUT_CONTINUATION","TREND_PULLBACK"],
  TRANSITION:["LIQUIDITY_SWEEP_RECLAIM","RANGE_REJECTION","GEMINI_VISION"]
};
export function selectAdaptiveExperts(candidates:any[],regime:RegimeResult) {
  const eligible=EXPERTS[regime.regime]||[];
  const ranked=candidates.filter(c=>eligible.includes(c.expert)).map(c=>{
    const regimeFit=eligible.indexOf(c.expert)===0?12:eligible.indexOf(c.expert)===1?8:4;
    const confidenceAdjustment=(regime.confidence-0.5)*10;
    return {...c,selectorScore:Number((c.qualityScore+regimeFit+confidenceAdjustment).toFixed(2)),selector:{regime:regime.regime,regimeConfidence:regime.confidence,expertRank:eligible.indexOf(c.expert)+1}};
  }).sort((a,b)=>b.selectorScore-a.selectorScore);
  // Keep independent expert opportunities; no cross-expert consensus requirement.
  const seen=new Set<string>();
  return ranked.filter(c=>{const k=c.expert+":"+c.direction;if(seen.has(k))return false;seen.add(k);return true;});
}
export const META_FEATURES=["qualityScore","stopAtr","rr","regimeConfidence","atrRatio","efficiency","bodyAtr","emaAlignment","slope","spreadAtr","isBuy","expertSweep","expertBreakout","expertPullback","expertRange","expertMomentum","expertVision","regimeTrend","regimeRange","regimeCompression","regimeExpansion","regimeImpulse"];
export function buildMetaFeatures(c:any,regime:RegimeResult,spread:number|null,atrValue:number) {
  const f=c.features||{},e=c.expert||"",r=regime.regime;
  return [Number(c.qualityScore||0),Number(c.stopAtr||0),Number(c.rr||0),regime.confidence,
    Number(regime.features.atrRatio||1),Number(regime.features.efficiency||0),Number(f.bodyAtr||regime.features.bodyAtr||0),
    Number(regime.features.emaAlignment||0),Number(regime.features.slope||0),spread!==null&&atrValue>0?spread/atrValue:0,
    c.direction==="BUY"?1:0,e.includes("SWEEP")?1:0,e.includes("BREAKOUT")?1:0,e.includes("PULLBACK")?1:0,
    e.includes("RANGE")?1:0,e.includes("MOMENTUM")?1:0,e.includes("VISION")?1:0,
    r.startsWith("TREND")?1:0,r==="RANGE"?1:0,r==="COMPRESSION"?1:0,r==="VOLATILITY_EXPANSION"?1:0,r==="IMPULSE"?1:0];
}
function walkTree(node:any,x:number[]):number {
  if(node.leaf!==undefined)return Number(node.leaf);
  const i=Number(String(node.split||"f0").replace(/^f/,"")),v=x[i]??0;
  const child=v<Number(node.split_condition)?node.yes:node.no;
  const next=node.children?.find((z:any)=>z.nodeid===child);
  return next?walkTree(next,x):0;
}
export function predictMetaLabel(model:any,x:number[]) {
  if(!model||model.kind!=="xgboost-one-vs-rest-v1")return null;
  const out:any={};
  for(const cls of ["TARGET","STOP","TIMEOUT"]) {
    const m=model.models?.[cls];if(!m||!Array.isArray(m.trees))return null;
    let margin=Number(m.baseMargin||0);
    for(const tree of m.trees)margin+=walkTree(tree,x);
    out[cls]=1/(1+Math.exp(-clamp(margin,-30,30)));
  }
  const sum=out.TARGET+out.STOP+out.TIMEOUT;
  if(!Number.isFinite(sum)||sum<=0)return null;
  return {pTarget:out.TARGET/sum,pStop:out.STOP/sum,pTimeout:out.TIMEOUT/sum,modelVersion:model.modelVersion||"unknown",trainedAt:model.trainedAt||null};
}
