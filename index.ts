import { classifyMarketRegime, selectAdaptiveExperts, buildMetaFeatures, predictMetaLabel, META_FEATURES } from "./ai_layers.ts";
import { marketVector, discoverOpportunities, predictScenarios, learnExpertWeights, applyExpertWeights, decisionIntelligence, collectForwardLabels } from "./advanced_intelligence.ts";

type Candle = { time: string; open: number; high: number; low: number; close: number; volume?: number };
type Candidate = { id: string; expert: string; direction: "BUY"|"SELL"; entry: number; stopLoss: number; target: number; risk: number; rr: number; stopAtr: number; qualityScore: number; barTime: string; regime: string; features: any; aiComment?: string|null };

const VERSION = "gold-ai-intelligence-0.7.1";
const BUILD_MARKER = "gold-ai-intelligence-0.7.1-kronos-forecast-audit";
const PORT = Number(Bun.env.PORT || 8080);
const URLS = {
  bridge: String(Bun.env.BRIDGE_URL || "https://gold-ctrader-bridge-production.up.railway.app").replace(/\/+$/,""),
  intelligence: String(Bun.env.INTELLIGENCE_URL || "https://gold-market-intelligence-production.up.railway.app").replace(/\/+$/,""),
  chartLab: String(Bun.env.CHART_LAB_URL || "https://gold-chart-lab-production.up.railway.app").replace(/\/+$/,""),
  kronos: String(Bun.env.KRONOS_URL || "https://gold-market-api-production.up.railway.app").replace(/\/+$/,"")
};
const STATE_PATH = "/data/gold-ai-paper-state.json";
const API_TOKEN = String(Bun.env.PAPER_API_TOKEN || "");
const isAuthorized = (req:Request) => !!API_TOKEN && req.headers.get("authorization") === "Bearer " + API_TOKEN;
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data,null,2),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});
const n = (x:any) => Number.isFinite(Number(x)) ? Number(x) : null;
const sleep = (ms:number) => new Promise(r=>setTimeout(r,ms));
const now = () => new Date().toISOString();
const clamp = (x:number,a:number,b:number) => Math.max(a,Math.min(b,x));

function freshState() {
  return { version:VERSION, createdAt:now(), updatedAt:now(), lastCycleAt:null as string|null, lastProcessedBar:null as string|null,
    openPositions:[] as any[], closedTrades:[] as any[], shadowPositions:[] as any[], candidateHistory:[] as any[], events:[] as any[], kronosForecastHistory:[] as any[], geminiDecisionHistory:[] as any[],
    daily:{date:now().slice(0,10),trades:0,netUsd:0}, lastReport:null as any,
    metaLabeler:{status:"COLLECTING_LABELS",trained:false,model:null,note:"No trained XGBoost model is claimed. Candidate features and outcomes are being collected."} };
}
async function readState() {
  const file = Bun.file(STATE_PATH);
  if(!await file.exists()) { console.log("AI_ENGINE_STATE_FILE_MISSING",STATE_PATH); return freshState(); }
  const x = await file.json();
  return {...freshState(),...x,openPositions:Array.isArray(x.openPositions)?x.openPositions:[],closedTrades:Array.isArray(x.closedTrades)?x.closedTrades:[],shadowPositions:Array.isArray(x.shadowPositions)?x.shadowPositions:[],candidateHistory:Array.isArray(x.candidateHistory)?x.candidateHistory:[],events:Array.isArray(x.events)?x.events:[],kronosForecastHistory:Array.isArray(x.kronosForecastHistory)?x.kronosForecastHistory:[],geminiDecisionHistory:Array.isArray(x.geminiDecisionHistory)?x.geminiDecisionHistory:[]};
}
async function writeState(s:any) {
  s.updatedAt=now();
  await Bun.write(STATE_PATH,JSON.stringify(s));
}
function event(s:any,x:any) { s.events.push({at:now(),...x}); if(s.events.length>1000)s.events=s.events.slice(-1000); }
async function getJson(url:string, options:any={}, timeout=18000) {
  const r=await fetch(url,{...options,signal:AbortSignal.timeout(timeout)});
  const body=await r.json().catch(()=>null);
  if(!r.ok||!body) throw new Error("HTTP_"+r.status+" "+url+" "+JSON.stringify(body).slice(0,200));
  return body;
}
function normalizeBars(data:any) {
  const raw=data?.multiTimeframe||data?.candles||{},mins:any={M5:5,M15:15,M30:30,H1:60,H4:240},out:any={};
  for(const tf of Object.keys(mins)) {
    const src=Array.isArray(raw[tf]?.items)?raw[tf].items:Array.isArray(raw[tf])?raw[tf]:[];
    const byTime=new Map<number,Candle>();
    for(const x of src) {
      const t=Date.parse(x.time||x.timestamp||""),o=n(x.open),h=n(x.high),l=n(x.low),c=n(x.close);
      if(!Number.isFinite(t)||[o,h,l,c].some(v=>v===null)||h<Math.max(o,c,l)||l>Math.min(o,c,h)||l<=0||t+mins[tf]*60000>Date.now())continue;
      byTime.set(t,{time:new Date(t).toISOString(),open:o!,high:h!,low:l!,close:c!,volume:n(x.volume)||0});
    }
    out[tf]=[...byTime.values()].sort((a,b)=>Date.parse(a.time)-Date.parse(b.time)).slice(-240);
  }
  return out;
}
function calcAtr(bars:Candle[],p=14) {
  if(bars.length<p+1)return null;
  const tr=bars.slice(1).map((b,i)=>Math.max(b.high-b.low,Math.abs(b.high-bars[i].close),Math.abs(b.low-bars[i].close))).slice(-p);
  return tr.reduce((a,b)=>a+b,0)/tr.length;
}
function nearestZone(zones:any[],kind:string,price:number,max:number) {
  return (Array.isArray(zones)?zones:[]).filter(z=>z.kind===kind&&n(z.zoneLow)!==null&&n(z.zoneHigh)!==null)
    .map(z=>({...z,distance:price<z.zoneLow?z.zoneLow-price:price>z.zoneHigh?price-z.zoneHigh:0}))
    .filter(z=>z.distance<=max).sort((a,b)=>a.distance-b.distance)[0]||null;
}
function buildCandidate(expert:string,side:"BUY"|"SELL",mid:number,atr:number,rawStop:number,bar:Candle,regime:string,features:any,baseScore=50) {
  if(!Number.isFinite(atr)||atr<=0)return null;
  const rawRisk=side==="BUY"?mid-rawStop:rawStop-mid;
  if(!Number.isFinite(rawRisk)||rawRisk<=0)return null;
  const risk=clamp(rawRisk,0.8*atr,1.5*atr),stop=side==="BUY"?mid-risk:mid+risk,rr=1.8;
  const target=side==="BUY"?mid+risk*rr:mid-risk*rr;
  return {id:[bar.time,expert,side].join("|"),expert,direction:side,entry:Number(mid.toFixed(2)),stopLoss:Number(stop.toFixed(2)),target:Number(target.toFixed(2)),risk:Number(risk.toFixed(3)),rr,stopAtr:Number((risk/atr).toFixed(2)),qualityScore:baseScore,barTime:bar.time,regime,features};
}
function expertAdjustment(closed:any[],expert:string) {
  const rows=closed.filter(x=>x.expert===expert).slice(-30);
  if(rows.length<8)return expert==="RANGE_REJECTION"?-6:0;
  const wins=rows.filter(x=>x.netUsd>0).length/rows.length;
  const avgR=rows.reduce((a,x)=>a+Number(x.resultR||0),0)/rows.length;
  const learned=clamp((wins-0.5)*16+avgR*4,-8,8);
  return expert==="RANGE_REJECTION"?Math.min(-4,learned-4):learned;
}
function generateCandidates(bars:any,chart:any,mid:number,kronos:any,refEngine:any,closed:any[],regimeOverride:any=null) {
  const out:Candidate[]=[],m5:Candle[]=bars.M5||[],last=m5.at(-1),prev=m5.at(-2),frames=chart.frames||{};
  const f5=frames.M5||{},f15=frames.M15||{},h1=frames.H1||{},a=n(f5.atr14)||calcAtr(m5);
  if(!last||!prev||!a||!Number.isFinite(mid))return {candidates:out,bar:last||null,atr:a,regime:"UNKNOWN"};
  const pivots=(Array.isArray(f5.pivots)?f5.pivots:[]).filter((p:any)=>Date.parse(p.confirmedAt||p.time||"")<=Date.parse(last.time));
  const highs=pivots.filter((p:any)=>p.side==="HIGH"&&n(p.price)!==null),lows=pivots.filter((p:any)=>p.side==="LOW"&&n(p.price)!==null);
  const sh=highs.at(-1),sl=lows.at(-1);
  const zones=[...(f15.zones||[]),...(h1.zones||[]),...(frames.H4?.zones||[])];
  const demand=nearestZone(zones,"DEMAND",mid,a*0.45),supply=nearestZone(zones,"SUPPLY",mid,a*0.45);
  const h1b=String(h1.bias||"NEUTRAL"),m15b=String(f15.bias||"NEUTRAL"),regime=String(regimeOverride?.regime||f15.regime||"TRANSITION");
  const body=Math.abs(last.close-last.open),pos=(last.close-last.low)/Math.max(0.00001,last.high-last.low);
  const add=(expert:string,side:"BUY"|"SELL",stop:number,features:any,bonus=0)=>{
    const c=buildCandidate(expert,side,mid,a,stop,last,regime,features,50+bonus);
    if(c)out.push(c);
  };
  if(sl&&last.low<sl.price&&last.close>sl.price&&last.close>last.open)add("LIQUIDITY_SWEEP_RECLAIM","BUY",Math.min(last.low,sl.price)-a*0.18,{level:sl.price,h1b,m15b},18);
  if(sh&&last.high>sh.price&&last.close<sh.price&&last.close<last.open)add("LIQUIDITY_SWEEP_RECLAIM","SELL",Math.max(last.high,sh.price)+a*0.18,{level:sh.price,h1b,m15b},18);
  if(sh&&last.close>sh.price&&prev.close<=sh.price)add("BREAKOUT_CONTINUATION","BUY",Math.min(last.low,sh.price-a*0.25),{level:sh.price,h1b,m15b},14);
  if(sl&&last.close<sl.price&&prev.close>=sl.price)add("BREAKOUT_CONTINUATION","SELL",Math.max(last.high,sl.price+a*0.25),{level:sl.price,h1b,m15b},14);
  const recent=m5.slice(-7,-1);
  if(sh&&recent.some(b=>b.close>sh.price)&&last.low<=sh.price+a*0.12&&last.close>sh.price&&last.close>last.open)
    add("BREAKOUT_RETEST","BUY",Math.min(last.low,sh.price-a*0.16),{level:sh.price,retest:true,h1b,m15b},16);
  if(sl&&recent.some(b=>b.close<sl.price)&&last.high>=sl.price-a*0.12&&last.close<sl.price&&last.close<last.open)
    add("BREAKOUT_RETEST","SELL",Math.max(last.high,sl.price+a*0.16),{level:sl.price,retest:true,h1b,m15b},16);
  const priorHigh=Math.max(...m5.slice(-6,-1).map(b=>b.high)),priorLow=Math.min(...m5.slice(-6,-1).map(b=>b.low));
  if(regime==="TREND_UP"&&h1b==="BULLISH"&&m15b==="BULLISH"&&last.close>priorHigh&&body>=a*0.35)
    add("TREND_CONTINUATION","BUY",Math.min(last.low,priorHigh-a*0.18),{breakLevel:priorHigh,h1b,m15b},15);
  if(regime==="TREND_DOWN"&&h1b==="BEARISH"&&m15b==="BEARISH"&&last.close<priorLow&&body>=a*0.35)
    add("TREND_CONTINUATION","SELL",Math.max(last.high,priorLow+a*0.18),{breakLevel:priorLow,h1b,m15b},15);
  if(sl&&last.low<sl.price&&last.close>sl.price&&last.close>prev.high&&last.close>last.open)
    add("REVERSAL","BUY",Math.min(last.low,sl.price)-a*0.22,{sweptLevel:sl.price,structureShift:true},17);
  if(sh&&last.high>sh.price&&last.close<sh.price&&last.close<prev.low&&last.close<last.open)
    add("REVERSAL","SELL",Math.max(last.high,sh.price)+a*0.22,{sweptLevel:sh.price,structureShift:true},17);
  if(demand&&(h1b==="BULLISH"||m15b==="BULLISH")&&last.close>last.open&&last.close>=demand.zoneLow)add("TREND_PULLBACK","BUY",Math.min(last.low,demand.zoneLow)-a*0.18,{zone:demand,h1b,m15b},12);
  if(supply&&(h1b==="BEARISH"||m15b==="BEARISH")&&last.close<last.open&&last.close<=supply.zoneHigh)add("TREND_PULLBACK","SELL",Math.max(last.high,supply.zoneHigh)+a*0.18,{zone:supply,h1b,m15b},12);
  if((regime==="RANGE"||regime==="RANGING")&&demand&&last.low<=demand.zoneHigh&&last.close>last.open)add("RANGE_REJECTION","BUY",Math.min(last.low,demand.zoneLow)-a*0.15,{zone:demand},0);
  if((regime==="RANGE"||regime==="RANGING")&&supply&&last.high>=supply.zoneLow&&last.close<last.open)add("RANGE_REJECTION","SELL",Math.max(last.high,supply.zoneHigh)+a*0.15,{zone:supply},0);
  if(body>=a*0.8&&pos>=0.78&&(h1b==="BULLISH"||m15b==="BULLISH"))add("MOMENTUM_IMPULSE","BUY",last.low-a*0.25,{bodyAtr:Number((body/a).toFixed(2)),closePosition:pos},8);
  if(body>=a*0.8&&pos<=0.22&&(h1b==="BEARISH"||m15b==="BEARISH"))add("MOMENTUM_IMPULSE","SELL",last.high+a*0.25,{bodyAtr:Number((body/a).toFixed(2)),closePosition:pos},8);

  const refAge=Date.now()-Date.parse(refEngine?.lastEvaluatedBar||"1970-01-01T00:00:00Z");
  if(refEngine?.ok&&refAge<=20*60000)for(const sc of (refEngine.scenarios||[])) {
    if(sc.status!=="ACTIVE"||!["BUY","SELL"].includes(sc.direction))continue;
    const lo=n(sc.entry_zone?.low),hi=n(sc.entry_zone?.high),stop=n(sc.stop_loss),target=n(sc.target);
    if(lo===null||hi===null||stop===null||target===null)continue;
    const dist=mid<lo?lo-mid:mid>hi?mid-hi:0;
    if(dist>a*0.5)continue;
    const risk=sc.direction==="BUY"?mid-stop:stop-mid,rew=sc.direction==="BUY"?target-mid:mid-target;
    if(risk<=0||rew/risk<1.5)continue;
    const c=buildCandidate("GEMINI_VISION",sc.direction,mid,a,stop,last,regime,{referenceScenario:sc.id,setupClass:sc.setup_class,trigger:sc.trigger},62);
    if(c)out.push(c);
  }
  for(const c of out) {
    if(c.direction==="BUY"&&h1b==="BULLISH")c.qualityScore+=8;
    if(c.direction==="SELL"&&h1b==="BEARISH")c.qualityScore+=8;
    if(c.direction==="BUY"&&m15b==="BULLISH")c.qualityScore+=5;
    if(c.direction==="SELL"&&m15b==="BEARISH")c.qualityScore+=5;
    if(kronos?.forecastBias==="UP"&&c.direction==="BUY"||kronos?.forecastBias==="DOWN"&&c.direction==="SELL")c.qualityScore+=7;
    if(kronos?.forecastBias==="UP"&&c.direction==="SELL"||kronos?.forecastBias==="DOWN"&&c.direction==="BUY")c.qualityScore-=4;
    c.qualityScore=Number(clamp(c.qualityScore+expertAdjustment(closed,c.expert),0,100).toFixed(1));
  }
  return {candidates:out.sort((a,b)=>b.qualityScore-a.qualityScore),bar:last,atr:a,regime,h1Bias:h1b,m15Bias:m15b};
}
function recordModelDecisions(s:any,engine:any,report:any,kronos:any) {
  const at=now();
  s.geminiDecisionHistory.push({at,engineOk:!!engine?.ok,engineVersion:engine?.engineVersion||null,lastEvaluatedBar:engine?.lastEvaluatedBar||null,planDecision:engine?.plan?.decision||null,visionBlocked:engine?.visionHealth?.entryBlocked??null,geminiPlan:engine?.plan||null,geminiVisionHealth:engine?.visionHealth||null,scenarios:(engine?.scenarios||[]).map((x:any)=>({...x})),error:engine?.error?String(engine.error).slice(0,300):null,primaryDecision:report?.decision||null,primaryReason:report?.reason||null,bar:report?.bar||report?.lastM5||null});
  if(s.geminiDecisionHistory.length>5000)s.geminiDecisionHistory=s.geminiDecisionHistory.slice(-5000);
  s.kronosForecastHistory.push({at,barTime:report?.bar||report?.lastM5||null,...(kronos||{status:"MISSING"})});
  if(s.kronosForecastHistory.length>5000)s.kronosForecastHistory=s.kronosForecastHistory.slice(-5000);
}
async function fetchKronos(bars:any) {
  const m5=bars.M5||[];
  if(m5.length<100)return {status:"UNAVAILABLE",reason:"INSUFFICIENT_CANDLES"};
  try {
    const r=await getJson(URLS.chartLab+"/kronos-forecast",{},50000);
    // Preserve the exact model response for audit. Do not invent a direction or confidence
    // when the upstream service does not provide one.
    const rawForecast=r.forecast??r.predictions??r.forecast_values??r.forecastValues??r.output??r.result??null;
    return {
      status:r.ok===false?"ERROR":"AVAILABLE",
      model:r.model??r.model_name??null,
      forecastBias:r.forecastBias??r.forecast_bias??r.direction??null,
      forecastChange:r.forecastChange??r.forecast_change??null,
      forecastChangePct:r.forecastChangePct??r.forecast_change_pct??null,
      forecast:rawForecast,
      quantiles:r.quantiles??r.forecast_quantiles??null,
      timestamps:r.timestamps??r.forecast_timestamps??null,
      latencyMs:r.latencyMs??r.latency_ms??null,
      barsUsed:r.barsUsed??r.bars_used??null,
      volumeUsed:r.volumeUsed??r.volume_used??null,
      cached:!!r.cached,
      outputAvailable:rawForecast!==null||r.forecastBias!=null||r.forecast_bias!=null||r.direction!=null,
      rawResponse:r,
      error:r.error||null
    };
  } catch(e:any) { return {status:"ERROR",reason:String(e?.message||e).slice(0,240)}; }
}
async function loadArtifact(path:string) {
  try { const file=Bun.file(path); if(!await file.exists())return null; return await file.json(); }
  catch(e:any) { console.log("AI_ARTIFACT_LOAD_ERROR",JSON.stringify({path,error:String(e?.message||e)})); return null; }
}
async function loadMetaModel() {
  try {
    const file=Bun.file("/data/gold-ai-xgb-model.json");
    if(!await file.exists())return null;
    const model=await file.json();
    const sides=["BUY","SELL"],classes=["TARGET","STOP","TIMEOUT"];
    const shapeOk=sides.every(side=>classes.every(cls=>Array.isArray(model.modelsBySide?.[side]?.[cls]?.trees)));
    const featureOrderOk=Array.isArray(model.featureNames)&&JSON.stringify(model.featureNames)===JSON.stringify(META_FEATURES);
    return model?.kind==="xgboost-one-vs-rest-v1"&&model.featureCount===25&&featureOrderOk&&shapeOk&&model.validated===true&&model.validation?.passed===true?model:null;
  } catch(e:any) { console.log("META_MODEL_LOAD_ERROR",String(e?.message||e)); return null; }
}
function stats(closed:any[]) {
  const sum=(a:any[])=>{const wins=a.filter(x=>x.netUsd>0),loss=a.filter(x=>x.netUsd<0),gw=wins.reduce((v,x)=>v+x.netUsd,0),gl=Math.abs(loss.reduce((v,x)=>v+x.netUsd,0));return {trades:a.length,wins:wins.length,losses:loss.length,winRate:a.length?Number((wins.length/a.length*100).toFixed(2)):null,profitFactor:gl?Number((gw/gl).toFixed(3)):null,netUsd:Number(a.reduce((v,x)=>v+x.netUsd,0).toFixed(2))};};
  const equity=closed.reduce((a,x)=>{a.push((a.at(-1)||0)+Number(x.netUsd||0));return a;},[] as number[]);
  let peak=0,dd=0;for(const x of equity){peak=Math.max(peak,x);dd=Math.max(dd,peak-x);}
  return {...sum(closed),maxDrawdownUsd:Number(dd.toFixed(2)),BUY:sum(closed.filter(x=>x.direction==="BUY")),SELL:sum(closed.filter(x=>x.direction==="SELL")),byExpert:Object.fromEntries([...new Set(closed.map(x=>x.expert))].map(k=>[k,sum(closed.filter(x=>x.expert===k))]))};
}
function freshnessPlaceholder(chart:any){return chart?.freshness||null;}
function resetDaily(s:any) {const d=now().slice(0,10);if(s.daily?.date!==d)s.daily={date:d,trades:0,netUsd:0};}
function calculatePositionOz(stopDistance:number,baseExposureOz:number,maxRiskUsd:number) {
  if(!Number.isFinite(stopDistance)||stopDistance<=0||!Number.isFinite(baseExposureOz)||baseExposureOz<=0||!Number.isFinite(maxRiskUsd)||maxRiskUsd<=0)return 0;
  return Number(Math.min(baseExposureOz,maxRiskUsd/stopDistance).toFixed(4));
}
function riskReason(c:Candidate,quote:any,a:number,s:any,cost:number,maxRisk:number,dailyLimit:number,maxOpen:number) {
  const risk=Math.abs(c.entry-c.stopLoss),reward=Math.abs(c.target-c.entry),spread=n(quote.spread),spreadRatio=spread!==null?spread/a:null;
  const size=Number(c.positionOz||0),riskUsd=risk*size,netRR=(reward*size-cost)/Math.max(0.0001,risk*size+cost);
  if(risk<=0||c.rr<1.5)return "INVALID_RISK_OR_RR";
  if(c.stopAtr<0.8||c.stopAtr>1.5)return "STOP_ATR_OUT_OF_RANGE";
  if(spreadRatio===null||spreadRatio>0.15)return "SPREAD_UNAVAILABLE_OR_HIGH";
  if(size<=0||riskUsd>maxRisk+0.01)return "POSITION_SIZE_OR_RISK_LIMIT";
  if(netRR<=0)return "NEGATIVE_NET_REWARD_AFTER_COST";
  const openRisk=s.openPositions.reduce((sum:number,p:any)=>sum+Number(p.positionRiskUsd??(Math.abs(p.entry-p.stopLoss)*Number(p.lotOz||0))),0);
  const remainingDailyRisk=Math.max(0,dailyLimit+Math.min(0,Number(s.daily.netUsd||0))-openRisk);
  if(riskUsd>remainingDailyRisk+0.01)return "DAILY_LOSS_OR_OPEN_RISK_BUDGET";
  if(s.openPositions.length>=maxOpen)return "MAX_OPEN_POSITIONS";
  const cLow=Math.min(c.entry,c.stopLoss,c.target),cHigh=Math.max(c.entry,c.stopLoss,c.target);
  if(s.openPositions.some((p:any)=>Math.max(cLow,Math.min(Number(p.entry),Number(p.stopLoss),Number(p.target)))<=Math.min(cHigh,Math.max(Number(p.entry),Number(p.stopLoss),Number(p.target)))))return "DUPLICATE_OR_OVERLAP";
  return null;
}
async function monitor(s:any,bar:Candle,lotOz:number,cost:number) {
  const keep=[];
  for(const p of s.openPositions) {
    if(Date.parse(bar.time)<=Date.parse(p.openedBarTime)){keep.push(p);continue;}
    const stop=p.direction==="BUY"?bar.low<=p.stopLoss:bar.high>=p.stopLoss;
    const target=p.direction==="BUY"?bar.high>=p.target:bar.low<=p.target;
    p.mfe=Math.max(Number(p.mfe||0),p.direction==="BUY"?bar.high-p.entry:p.entry-bar.low);
    p.mae=Math.max(Number(p.mae||0),p.direction==="BUY"?p.entry-bar.low:bar.high-p.entry);
    const bothHit=stop&&target;
    let why=null,exit=null;
    if(stop){why=bothHit?"STOP_BOTH_HIT_STOP_FIRST":"STOP";exit=p.stopLoss;}else if(target){why="TARGET";exit=p.target;}else if(Date.parse(bar.time)-Date.parse(p.openedBarTime)>=21600000){why="TIMEOUT";exit=bar.close;}
    if(!why){keep.push(p);continue;}
    const move=p.direction==="BUY"?exit-p.entry:p.entry-exit,positionOz=Number(p.lotOz??lotOz),gross=move*positionOz,net=gross-cost;
    const row={...p,closedAt:now(),exit,exitReason:why,bothHitStopFirst:bothHit,grossUsd:Number(gross.toFixed(2)),costUsd:cost,netUsd:Number(net.toFixed(2)),stressNetUsd:Number((gross-0.74).toFixed(2)),resultR:Number((move/Math.max(0.0001,p.risk)).toFixed(3))};
    s.closedTrades.push(row);s.daily.trades++;s.daily.netUsd+=row.netUsd;
    s.candidateHistory.push({type:"SHADOW_OUTCOME",id:p.id,expert:p.expert,direction:p.direction,outcome:why,netUsd:row.netUsd,stressNetUsd:row.stressNetUsd,resultR:row.resultR});
    event(s,{type:"PAPER_CLOSE",id:p.id,expert:p.expert,direction:p.direction,exitReason:why,netUsd:row.netUsd,resultR:row.resultR});
    console.log("AI_PAPER_CLOSE",JSON.stringify({id:p.id,expert:p.expert,direction:p.direction,why,netUsd:row.netUsd,resultR:row.resultR}));
  }
  s.openPositions=keep;
  if(s.closedTrades.length>2000)s.closedTrades=s.closedTrades.slice(-2000);
  const shadows=[];
  for(const p of s.shadowPositions) {
    if(Date.parse(bar.time)<=Date.parse(p.barTime)){shadows.push(p);continue;}
    const stop=p.direction==="BUY"?bar.low<=p.stopLoss:bar.high>=p.stopLoss,target=p.direction==="BUY"?bar.high>=p.target:bar.low<=p.target;
    const bothHit=stop&&target;let why=null,exit=null;if(stop){why=bothHit?"STOP_BOTH_HIT_STOP_FIRST":"STOP";exit=p.stopLoss;}else if(target){why="TARGET";exit=p.target;}else if(Date.parse(bar.time)-Date.parse(p.barTime)>=21600000){why="TIMEOUT";exit=bar.close;}
    if(!why){shadows.push(p);continue;}
    const move=p.direction==="BUY"?exit-p.entry:p.entry-exit,gross=move*Number(p.lotOz??lotOz);
    s.candidateHistory.push({type:"SHADOW_OUTCOME",id:p.id,expert:p.expert,direction:p.direction,outcome:why,netUsd:Number((gross-cost).toFixed(2)),stressNetUsd:Number((gross-0.74).toFixed(2)),resultR:Number((move/Math.max(.0001,p.risk)).toFixed(3))});
    event(s,{type:"SHADOW_OUTCOME",id:p.id,expert:p.expert,outcome:why,netUsd:Number((gross-cost).toFixed(2))});
  }
  s.shadowPositions=shadows;
}
async function runCycle(source="scheduled") {
  console.log("AI_CYCLE_BEGIN",JSON.stringify({source,at:now()}));
  const s=await readState();resetDaily(s);
  console.log("AI_FETCH_BRIDGE_START",URLS.bridge);
  const market=await getJson(URLS.bridge+"/market-data",{},15000);
  console.log("AI_FETCH_BRIDGE_OK",JSON.stringify({ok:market.ok,keys:Object.keys(market||{}).slice(0,20)}));
  console.log("AI_FETCH_CHART_START",URLS.chartLab);
  const chart=await getJson(URLS.chartLab+"/market-read?refresh=1",{},18000);
  console.log("AI_FETCH_CHART_OK",JSON.stringify({ok:chart.ok,timeframes:Object.keys(chart.timeframes||{}),freshness:chart.freshness||null}));
  const levels=await getJson(URLS.chartLab+"/levels?refresh=1",{},18000);
  console.log("AI_FETCH_LEVELS_OK",JSON.stringify({ok:levels.ok,frames:Object.keys(levels.frames||{})}));
  const chartContext={...chart,frames:levels.frames||{}};
  const engine=await getJson(URLS.intelligence+"/playbook-status",{},10000).catch(e=>({ok:false,error:String(e)}));
  console.log("AI_FETCH_ENGINE",JSON.stringify({ok:engine.ok,scenarioCount:engine.scenarios?.length||0,version:engine.engineVersion||null}));
  const trades=await getJson(URLS.intelligence+"/playbook-trades",{},10000).catch(e=>({ok:false,error:String(e)}));
  console.log("AI_FETCH_TRADES",JSON.stringify({ok:trades.ok,overall:trades.overall||null}));
  if(!market.ok)throw new Error("BRIDGE_MARKET_DATA_NOT_OK");
  const bars=normalizeBars(market),quote={bid:n(market.bid),ask:n(market.ask),spread:n(market.spread)};
  const mid=quote.bid!==null&&quote.ask!==null?(quote.bid+quote.ask)/2:null,last=bars.M5.at(-1);
  const f=chart.freshness||{},fresh=chart.ok===true&&f.dataFresh===true&&Number(f.quoteAgeMinutes)<=10&&Number(f.m5BarAgeMinutes)<=20;
  const cost=n(Bun.env.PAPER_TRADE_COST_USD)??0.37,stress=n(Bun.env.PAPER_STRESS_COST_USD)??0.74;
  const lotSize=n(Bun.env.PAPER_LOT_SIZE)??0.01,contract=n(Bun.env.CONTRACT_OZ_PER_LOT)??100,lotOz=lotSize*contract;
  const maxRisk=n(Bun.env.MAX_TRADE_RISK_USD)??2,dailyLimit=n(Bun.env.DAILY_LOSS_LIMIT_USD)??4,maxOpenPositions=Math.max(1,Math.floor(n(Bun.env.MAX_OPEN_POSITIONS)??4));
  let kronos:any={status:"SKIPPED",reason:"STALE_OR_INSUFFICIENT_MARKET_DATA"};
  if(last&&bars.M5.length>=100&&fresh&&mid!==null){kronos=await fetchKronos(bars);console.log("AI_KRONOS_RESULT",JSON.stringify(kronos));}
  const ref=engine.ok?engine:null;
  if(!last||bars.M5.length<30||!fresh||mid===null) {
    console.log("AI_DATA_GATE",JSON.stringify({fresh,barCount:bars.M5.length,quote,chartFreshness:freshnessPlaceholder(chart)}));
    const report={at:now(),source,decision:"NO_TRADE",reason:!fresh?"STALE_OR_INVALID_MARKET_DATA":"INSUFFICIENT_M5_CANDLES",marketFresh:fresh,lastM5:last?.time||null,quote,kronos};
    s.lastCycleAt=now();s.lastReport=report;recordModelDecisions(s,engine,report,kronos);event(s,{type:"NO_TRADE",reason:report.reason});await writeState(s);return report;
  }
  await monitor(s,last,lotOz,cost);
  if(s.lastProcessedBar===last.time) {
    const report={at:now(),source,decision:"NO_NEW_BAR",reason:"BAR_ALREADY_PROCESSED",bar:last.time,openPositions:s.openPositions.length,kronos};
    s.lastCycleAt=now();s.lastReport=report;recordModelDecisions(s,engine,report,kronos);await writeState(s);return report;
  }
  const regimeResult=classifyMarketRegime(bars.M5,s.lastReport?.regime||null);
  const learningCollection=collectForwardLabels(s,bars.M5,regimeResult.regime,last.time);
  const generated=generateCandidates(bars,chartContext,mid,kronos,ref,s.closedTrades,regimeResult);
  const model=await loadMetaModel();
  const opportunityModel=await loadArtifact("/data/gold-ai-opportunity-model.json");
  const scenarioModel=await loadArtifact("/data/gold-ai-scenario-model.json");
  const discoveryVector=marketVector(bars.M5,regimeResult.regime,last.time);
  const scenarioForecast=discoveryVector?predictScenarios(scenarioModel,discoveryVector):{status:"INSUFFICIENT_FEATURE_HISTORY",horizons:{}};
  const discovery=discoveryVector?discoverOpportunities(opportunityModel,discoveryVector,mid,generated.atr,last.time,regimeResult.regime):{status:"INSUFFICIENT_FEATURE_HISTORY",candidates:[]};
  let adaptiveCandidates=selectAdaptiveExperts([...generated.candidates,...discovery.candidates],regimeResult);
  const shadowLearning=s.candidateHistory.filter((x:any)=>x.type==="SHADOW_OUTCOME").map((o:any)=>({...o,regime:s.candidateHistory.find((z:any)=>z.id===o.id&&!z.type)?.regime||null}));
  const expertLearning=learnExpertWeights([...s.closedTrades,...shadowLearning],regimeResult.regime);
  adaptiveCandidates=applyExpertWeights(adaptiveCandidates,expertLearning);
  for(const c of adaptiveCandidates) {
    c.metaFeatures=buildMetaFeatures(c,regimeResult,quote.spread,generated.atr);
    c.metaLabel=model?predictMetaLabel(model,c.metaFeatures):null;
    if(c.metaLabel)c.metaLabel.timeoutMeanR=Number(model?.timeoutMeanR?.[c.direction]??model?.timeoutMeanR??0);
    c.metaEVUsd=c.metaLabel?decisionIntelligence(c,c.metaLabel,cost,c.positionOz||lotOz,stress).evUsd:null;
    c.metaEVStressUsd=c.metaLabel?decisionIntelligence(c,c.metaLabel,cost,c.positionOz||lotOz,stress).stressEvUsd:null;
    c.decisionIntelligence=decisionIntelligence(c,c.metaLabel,cost,c.positionOz||lotOz,stress);
    c.scenarioForecast=scenarioForecast;
  }
  const accepted:any[]=[],rejected:any[]=[];
  const operatingMode=model?"MODEL":"EXPERT_RULES";
  for(const c of adaptiveCandidates) {
    const stopDistance=Math.abs(c.entry-c.stopLoss);
    c.positionOz=calculatePositionOz(stopDistance,lotOz,maxRisk);
    c.positionRiskUsd=Number((stopDistance*c.positionOz).toFixed(2));
    if(model&&c.metaLabel) {
      c.metaLabel.timeoutMeanR=Number(model.timeoutMeanR?.[c.direction]??0);
      c.decisionIntelligence=decisionIntelligence(c,c.metaLabel,cost,c.positionOz,stress);
      c.metaEVUsd=c.decisionIntelligence.evUsd;
      c.metaEVStressUsd=c.decisionIntelligence.stressEvUsd;
    }
    let why=riskReason(c,quote,generated.atr,s,cost,maxRisk,dailyLimit,maxOpenPositions);
    if(!why&&model&&(!c.metaLabel||c.metaEVUsd===null||c.metaEVUsd<=0||c.metaEVStressUsd===null||c.metaEVStressUsd<=0||c.decisionIntelligence?.decision!=="PAPER_ELIGIBLE"))why="META_LABELER_LOW_EXPECTED_VALUE";
    const candidateMode=model?"MODEL":"EXPERT_RULES";
    const candidateDecision={engineVersion:VERSION,buildMarker:BUILD_MARKER,mode:candidateMode,expert:c.expert,direction:c.direction,decision:why?"NO_TRADE":"PAPER_RECOMMENDATION",reason:why,positionOz:c.positionOz,positionRiskUsd:c.positionRiskUsd,maxRiskUsd:maxRisk,baseExposureOz:lotOz,entry:c.entry,stopLoss:c.stopLoss,target:c.target,atr:generated.atr,stopAtr:c.stopAtr};
    console.log("AI_CANDIDATE_DECISION",JSON.stringify(candidateDecision));
    if(why) {
      rejected.push({id:c.id,expert:c.expert,direction:c.direction,reason:why,qualityScore:c.qualityScore,metaLabel:c.metaLabel,metaEVUsd:c.metaEVUsd,metaEVStressUsd:c.metaEVStressUsd});
      s.shadowPositions.push({id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,risk:c.risk,barTime:last.time,createdAt:now(),rejectedReason:why,lotOz:c.positionOz});
      event(s,{type:"CANDIDATE_REJECTED",id:c.id,expert:c.expert,direction:c.direction,reason:why,qualityScore:c.qualityScore,metaLabel:c.metaLabel,metaEVUsd:c.metaEVUsd,metaEVStressUsd:c.metaEVStressUsd});
      continue;
    }
    c.signalStatus=model?"MODEL_VALIDATED_PAPER":"UNVALIDATED_EXPERT";
    const p={id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,risk:c.risk,rr:c.rr,qualityScore:c.qualityScore,openedAt:now(),openedBarTime:last.time,regime:c.regime,features:c.features,metaFeatures:c.metaFeatures,metaLabel:c.metaLabel,metaEVUsd:c.metaEVUsd,metaEVStressUsd:c.metaEVStressUsd,kronosBias:kronos.forecastBias||null,lotOz:c.positionOz,positionRiskUsd:c.positionRiskUsd,costUsd:cost,mfe:0,mae:0,mode:"PAPER",signalStatus:c.signalStatus};
    s.openPositions.push(p);accepted.push({id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,rr:c.rr,qualityScore:c.qualityScore,lotOz:c.positionOz,positionRiskUsd:c.positionRiskUsd,status:c.signalStatus,metaLabel:c.metaLabel,metaEVUsd:c.metaEVUsd,metaEVStressUsd:c.metaEVStressUsd});
    event(s,{type:"PAPER_OPEN",id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,qualityScore:c.qualityScore,metaLabel:c.metaLabel,metaEVUsd:c.metaEVUsd});
    console.log("AI_PAPER_OPEN",JSON.stringify({id:c.id,engineVersion:VERSION,mode:operatingMode,signalStatus:c.signalStatus,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,qualityScore:c.qualityScore,positionOz:c.positionOz,positionRiskUsd:c.positionRiskUsd}));
    if(s.openPositions.length>=maxOpenPositions)break;
  }
  for(const c of adaptiveCandidates)s.candidateHistory.push({at:now(),barTime:last.time,id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,rr:c.rr,stopAtr:c.stopAtr,qualityScore:c.qualityScore,positionOz:c.positionOz,positionRiskUsd:c.positionRiskUsd,features:c.features,metaFeatures:c.metaFeatures,metaLabel:c.metaLabel,metaEVUsd:c.metaEVUsd,metaEVStressUsd:c.metaEVStressUsd,regime:regimeResult.regime,regimeConfidence:regimeResult.confidence,entered:accepted.some(x=>x.id===c.id),signalStatus:c.signalStatus||(rejected.find(x=>x.id===c.id)?.reason||"REJECTED"),rejectionReason:rejected.find(x=>x.id===c.id)?.reason||null});
  if(s.candidateHistory.length>20000)s.candidateHistory=s.candidateHistory.slice(-20000);
  if(s.shadowPositions.length>500)s.shadowPositions=s.shadowPositions.slice(-500);
  const report={at:now(),source,mode:operatingMode,decision:accepted.length?"PAPER_CANDIDATE_OPENED":"NO_TRADE",reason:accepted.length?(model?"MODEL_EV_VALIDATED":"UNVALIDATED_EXPERT_RULES_PAPER"):generated.candidates.length?"CANDIDATES_REJECTED_BY_RISK":"NO_EXPERT_TRIGGER",
    bar:last.time,regime:regimeResult.regime,regimeConfidence:regimeResult.confidence,regimeChanged:regimeResult.changed,regimeFeatures:regimeResult.features,h1Bias:generated.h1Bias,m15Bias:generated.m15Bias,marketFresh:fresh,quote,atrM5:generated.atr,
    learningData:learningCollection,
    aiOpportunityDiscovery:{status:discovery.status,modelVersion:opportunityModel?.modelVersion||null,candidates:discovery.candidates.length},
    scenarioPrediction:scenarioForecast,
    metaLearning:{status:"PAPER_OUTCOME_ADAPTATION",expertWeights:expertLearning},
    decisionIntelligence:{status:model?"XGBOOST_EV_AND_STRESS_GATE":"EXPERT_RULES_NO_PROBABILITIES",policy:"BUY_SELL_NO_TRADE",timeoutMeanR:model?.timeoutMeanR??null},
    expertsEvaluated:["BREAKOUT_CONTINUATION","LIQUIDITY_SWEEP_RECLAIM","BREAKOUT_RETEST","TREND_PULLBACK","TREND_CONTINUATION","RANGE_REJECTION","REVERSAL","MOMENTUM_IMPULSE"],
    candidatesGenerated:adaptiveCandidates.length,expertsSelected:[...new Set(adaptiveCandidates.map(c=>c.expert))],accepted,rejected,openPositions:s.openPositions.length,shadowTracking:s.shadowPositions.length,
    kronos,referenceEngine:{status:engine.status,version:engine.engineVersion,plan:engine.plan?.decision||null,visionBlocked:engine.visionHealth?.entryBlocked??null},
    referenceTrades:trades.ok?trades.overall:null,costUsd:cost,stressCostUsd:stress,stats:stats(s.closedTrades),
    metaLabeler:{status:model?"TRAINED":"COLLECTING_LABELS",trained:!!model,modelVersion:model?.modelVersion||null,trainingSamples:model?.trainSamples||0,validationSamples:model?.validationSamples||0,metrics:model?.walkForwardMetrics||null},liveTrading:false,
     riskSettings:{maxTradeRiskUsd:maxRisk,dailyLossLimitUsd:dailyLimit,maxOpenPositions,baseExposureOz:lotOz},sameBarPolicy:"If STOP and TARGET are both touched in one candle, STOP is assumed first (worst case).",disclaimer:model?"PAPER only. XGBoost accepted only with validated=true and validation.passed=true.":"PAPER only. EXPERT_RULES recommendations are UNVALIDATED_EXPERT; no probabilities are fabricated."};
  s.lastProcessedBar=last.time;s.lastCycleAt=now();s.lastReport=report;recordModelDecisions(s,engine,report,kronos);
  await writeState(s);event(s,{type:"CYCLE",bar:last.time,decision:report.decision,candidates:generated.candidates.length,accepted:accepted.length,rejected:rejected.length,regime:generated.regime});
  console.log("AI_ENGINE_CYCLE",JSON.stringify(report));
  return report;
}
async function status() {
  const s=await readState(); const model=await loadMetaModel();
  return {ok:true,service:"Gold AI Intelligence Engine",version:VERSION,mode:"PAPER_ONLY",operatingMode:(await loadMetaModel())?"MODEL":"EXPERT_RULES",liveTrading:false,lastCycleAt:s.lastCycleAt,lastProcessedBar:s.lastProcessedBar,
    openPositions:s.openPositions,shadowTracking:s.shadowPositions.length,stats:stats(s.closedTrades),lastReport:s.lastReport,metaLabeler:{status:model?"TRAINED":"COLLECTING_LABELS",trained:!!model,modelVersion:model?.modelVersion||null,trainingSamples:model?.trainSamples||0,validationSamples:model?.validationSamples||0,metrics:model?.walkForwardMetrics||null},storage:"persistent-volume"};
}
let running=false;
const server=Bun.serve({port:PORT,hostname:"0.0.0.0",async fetch(req) {
  const u=new URL(req.url);
  if(req.method==="GET"&&(u.pathname==="/"||u.pathname==="/health"))return json({ok:true,service:"Gold AI Intelligence Engine",version:VERSION,mode:"PAPER_ONLY",liveTrading:false});
  if(req.method==="GET"&&u.pathname==="/status")try{return json(await status());}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  if(req.method==="GET"&&u.pathname==="/analysis")try{const s=await readState();return json(s.lastReport||{decision:"NO_TRADE",reason:"NO_CYCLE_YET"});}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  if(req.method==="GET"&&u.pathname==="/signals")try{
    const s=await readState(),closed=s.closedTrades||[],history=s.candidateHistory||[],byExpert:any={};
    for(const expert of [...new Set([...closed.map((x:any)=>x.expert),...history.map((x:any)=>x.expert)])]) {
      const trades=closed.filter((x:any)=>x.expert===expert),wins=trades.filter((x:any)=>Number(x.netUsd)>0),losses=trades.filter((x:any)=>Number(x.netUsd)<0);
      const gp=wins.reduce((a:number,x:any)=>a+Number(x.netUsd),0),gl=Math.abs(losses.reduce((a:number,x:any)=>a+Number(x.netUsd),0));
      byExpert[expert]={samples:history.filter((x:any)=>x.expert===expert).length,trades:trades.length,wins:wins.length,winRatePct:trades.length?Number((wins.length/trades.length*100).toFixed(2)):null,profitFactor:gl?Number((gp/gl).toFixed(3)):null,netUsd:Number(trades.reduce((a:number,x:any)=>a+Number(x.netUsd||0),0).toFixed(2))};
    }
    return json({ok:true,version:VERSION,mode:(await loadMetaModel())?"MODEL":"EXPERT_RULES",latestRecommendations:history.filter((x:any)=>x.entered).slice(-50).reverse(),byExpert,openPositions:s.openPositions,stats:stats(closed),liveTrading:false});
  }catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  if(req.method==="GET"&&u.pathname==="/model-history") {
    if(!isAuthorized(req))return json({ok:false,error:"UNAUTHORIZED"},401);
    try{const s=await readState();return json({ok:true,kronosForecastHistory:s.kronosForecastHistory.slice(-200),geminiDecisionHistory:s.geminiDecisionHistory.slice(-200),counts:{kronosForecasts:s.kronosForecastHistory.length,geminiDecisions:s.geminiDecisionHistory.length}});}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  }
  if(req.method==="GET"&&u.pathname==="/history") {
    if(!isAuthorized(req))return json({ok:false,error:"UNAUTHORIZED"},401);
    try{const s=await readState();return json({ok:true,events:s.events.slice(-120),candidates:s.candidateHistory.slice(-120),openPositions:s.openPositions,closedTrades:s.closedTrades.slice(-120),stats:stats(s.closedTrades)});}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  }
  if(req.method==="POST"&&u.pathname==="/sync") {
    if(!isAuthorized(req))return json({ok:false,error:"UNAUTHORIZED"},401);
    if(running)return json({ok:true,status:"ALREADY_RUNNING"},202);
    running=true;try{return json(await runCycle("manual"));}catch(e:any){return json({ok:false,error:String(e?.message||e)},502);}finally{running=false;}
  }
  return json({ok:false,error:"NOT_FOUND",available:["/health","/status","/analysis","/signals","/history","/model-history","/sync"]},404);
}});
console.log("AI_ENGINE_LISTENING",JSON.stringify({url:server.url,version:VERSION,engineVersion:VERSION,buildMarker:BUILD_MARKER,mode:"PAPER_ONLY",storage:"persistent-volume",urls:URLS}));
void (async()=>{await sleep(12000);while(true){if(!running){running=true;try{await runCycle("scheduled");}catch(e:any){console.log("AI_ENGINE_LOOP_ERROR",JSON.stringify({message:String(e?.message||e),name:e?.name||null,code:e?.code||null,stack:String(e?.stack||"").slice(0,900)}));}finally{running=false;}}await sleep(300000);}})();
