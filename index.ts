import { S3Client, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";

type Candle = { time: string; open: number; high: number; low: number; close: number; volume?: number };
type Candidate = { id: string; expert: string; direction: "BUY"|"SELL"; entry: number; stopLoss: number; target: number; risk: number; rr: number; stopAtr: number; qualityScore: number; barTime: string; regime: string; features: any; aiComment?: string|null };

const VERSION = "gold-ai-intelligence-0.3.0";
const PORT = Number(Bun.env.PORT || 8080);
const URLS = {
  bridge: String(Bun.env.BRIDGE_URL || "https://gold-ctrader-bridge-production.up.railway.app").replace(/\/+$/,""),
  intelligence: String(Bun.env.INTELLIGENCE_URL || "https://gold-market-intelligence-production.up.railway.app").replace(/\/+$/,""),
  chartLab: String(Bun.env.CHART_LAB_URL || "https://gold-chart-lab-production.up.railway.app").replace(/\/+$/,""),
  kronos: String(Bun.env.KRONOS_URL || "https://gold-market-api-production.up.railway.app").replace(/\/+$/,"")
};
const BUCKET = Bun.env.S3_BUCKET || "";
const STATE_KEY = "gold-ai-paper/state.json";
const S3 = new S3Client({
  endpoint: Bun.env.S3_ENDPOINT,
  region: Bun.env.S3_REGION || "auto",
  credentials: { accessKeyId: Bun.env.S3_ACCESS_KEY_ID || "", secretAccessKey: Bun.env.S3_SECRET_ACCESS_KEY || "" },
  forcePathStyle: Bun.env.S3_URL_STYLE === "path"
});
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data,null,2),{status,headers:{"content-type":"application/json; charset=utf-8","cache-control":"no-store"}});
const n = (x:any) => Number.isFinite(Number(x)) ? Number(x) : null;
const sleep = (ms:number) => new Promise(r=>setTimeout(r,ms));
const now = () => new Date().toISOString();
const clamp = (x:number,a:number,b:number) => Math.max(a,Math.min(b,x));

function freshState() {
  return { version:VERSION, createdAt:now(), updatedAt:now(), lastCycleAt:null as string|null, lastProcessedBar:null as string|null,
    openPositions:[] as any[], closedTrades:[] as any[], shadowPositions:[] as any[], candidateHistory:[] as any[], events:[] as any[],
    daily:{date:now().slice(0,10),trades:0,netUsd:0}, lastReport:null as any,
    metaLabeler:{status:"COLLECTING_LABELS",trained:false,model:null,note:"No trained XGBoost model is claimed. Candidate features and outcomes are being collected."} };
}
async function readState() {
  if(!BUCKET || !Bun.env.S3_ENDPOINT || !Bun.env.S3_ACCESS_KEY_ID || !Bun.env.S3_SECRET_ACCESS_KEY) throw new Error("PERSISTENT_STORAGE_NOT_CONFIGURED");
  try {
    const r:any=await S3.send(new GetObjectCommand({Bucket:BUCKET,Key:STATE_KEY}));
    const x=JSON.parse(await r.Body.transformToString());
    return {...freshState(),...x,openPositions:Array.isArray(x.openPositions)?x.openPositions:[],closedTrades:Array.isArray(x.closedTrades)?x.closedTrades:[],shadowPositions:Array.isArray(x.shadowPositions)?x.shadowPositions:[],candidateHistory:Array.isArray(x.candidateHistory)?x.candidateHistory:[],events:Array.isArray(x.events)?x.events:[]};
  } catch(e:any) {
    if(/NoSuchKey|NotFound|404/i.test(String(e?.name||e?.Code||e?.message||e))) return freshState();
    throw e;
  }
}
async function writeState(s:any) {
  s.updatedAt=now();
  await S3.send(new PutObjectCommand({Bucket:BUCKET,Key:STATE_KEY,Body:JSON.stringify(s),ContentType:"application/json"}));
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
function buildCandidate(expert:string,side:"BUY"|"SELL",mid:number,atr:number,stop:number,bar:Candle,regime:string,features:any,baseScore=50) {
  const risk=side==="BUY"?mid-stop:stop-mid;
  if(!Number.isFinite(risk)||risk<=0||risk/atr<0.25||risk/atr>2.5)return null;
  const target=side==="BUY"?mid+risk*1.8:mid-risk*1.8;
  return {id:[bar.time,expert,side].join("|"),expert,direction:side,entry:Number(mid.toFixed(2)),stopLoss:Number(stop.toFixed(2)),target:Number(target.toFixed(2)),risk:Number(risk.toFixed(3)),rr:1.8,stopAtr:Number((risk/atr).toFixed(2)),qualityScore:baseScore,barTime:bar.time,regime,features};
}
function expertAdjustment(closed:any[],expert:string) {
  const rows=closed.filter(x=>x.expert===expert).slice(-30);
  if(rows.length<8)return 0;
  const wins=rows.filter(x=>x.netUsd>0).length/rows.length;
  const avgR=rows.reduce((a,x)=>a+Number(x.resultR||0),0)/rows.length;
  return clamp((wins-0.5)*16+avgR*4,-8,8);
}
function generateCandidates(bars:any,chart:any,mid:number,kronos:any,refEngine:any,closed:any[]) {
  const out:Candidate[]=[],m5:Candle[]=bars.M5||[],last=m5.at(-1),prev=m5.at(-2),frames=chart.frames||{};
  const f5=frames.M5||{},f15=frames.M15||{},h1=frames.H1||{},a=n(f5.atr14)||calcAtr(m5);
  if(!last||!prev||!a||!Number.isFinite(mid))return {candidates:out,bar:last||null,atr:a,regime:"UNKNOWN"};
  const pivots=(Array.isArray(f5.pivots)?f5.pivots:[]).filter((p:any)=>Date.parse(p.confirmedAt||p.time||"")<=Date.parse(last.time));
  const highs=pivots.filter((p:any)=>p.side==="HIGH"&&n(p.price)!==null),lows=pivots.filter((p:any)=>p.side==="LOW"&&n(p.price)!==null);
  const sh=highs.at(-1),sl=lows.at(-1);
  const zones=[...(f15.zones||[]),...(h1.zones||[]),...(frames.H4?.zones||[])];
  const demand=nearestZone(zones,"DEMAND",mid,a*0.45),supply=nearestZone(zones,"SUPPLY",mid,a*0.45);
  const h1b=String(h1.bias||"NEUTRAL"),m15b=String(f15.bias||"NEUTRAL"),regime=String(f15.regime||"TRANSITION");
  const body=Math.abs(last.close-last.open),pos=(last.close-last.low)/Math.max(0.00001,last.high-last.low);
  const add=(expert:string,side:"BUY"|"SELL",stop:number,features:any,bonus=0)=>{
    const c=buildCandidate(expert,side,mid,a,stop,last,regime,features,50+bonus);
    if(c)out.push(c);
  };
  if(sl&&last.low<sl.price&&last.close>sl.price&&last.close>last.open)add("LIQUIDITY_SWEEP_RECLAIM","BUY",Math.min(last.low,sl.price)-a*0.18,{level:sl.price,h1b,m15b},18);
  if(sh&&last.high>sh.price&&last.close<sh.price&&last.close<last.open)add("LIQUIDITY_SWEEP_RECLAIM","SELL",Math.max(last.high,sh.price)+a*0.18,{level:sh.price,h1b,m15b},18);
  if(sh&&last.close>sh.price&&prev.close<=sh.price)add("BREAKOUT_CONTINUATION","BUY",Math.min(last.low,sh.price-a*0.25),{level:sh.price,h1b,m15b},14);
  if(sl&&last.close<sl.price&&prev.close>=sl.price)add("BREAKOUT_CONTINUATION","SELL",Math.max(last.high,sl.price+a*0.25),{level:sl.price,h1b,m15b},14);
  if(demand&&(h1b==="BULLISH"||m15b==="BULLISH")&&last.close>last.open&&last.close>=demand.zoneLow)add("TREND_PULLBACK","BUY",Math.min(last.low,demand.zoneLow)-a*0.18,{zone:demand,h1b,m15b},12);
  if(supply&&(h1b==="BEARISH"||m15b==="BEARISH")&&last.close<last.open&&last.close<=supply.zoneHigh)add("TREND_PULLBACK","SELL",Math.max(last.high,supply.zoneHigh)+a*0.18,{zone:supply,h1b,m15b},12);
  if((regime==="RANGE"||regime==="RANGING")&&demand&&last.low<=demand.zoneHigh&&last.close>last.open)add("RANGE_REJECTION","BUY",Math.min(last.low,demand.zoneLow)-a*0.15,{zone:demand},10);
  if((regime==="RANGE"||regime==="RANGING")&&supply&&last.high>=supply.zoneLow&&last.close<last.open)add("RANGE_REJECTION","SELL",Math.max(last.high,supply.zoneHigh)+a*0.15,{zone:supply},10);
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
    out.push({id:[last.time,"GEMINI_VISION",sc.id].join("|"),expert:"GEMINI_VISION",direction:sc.direction,entry:Number(mid.toFixed(2)),stopLoss:stop,target,risk:Number(risk.toFixed(3)),rr:Number((rew/risk).toFixed(2)),stopAtr:Number((risk/a).toFixed(2)),qualityScore:62,barTime:last.time,regime,features:{referenceScenario:sc.id,setupClass:sc.setup_class,trigger:sc.trigger}});
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
async function fetchKronos(bars:any) {
  const key=Bun.env.KRONOS_API_KEY,m5=bars.M5||[];
  if(!key||m5.length<100)return {status:"UNAVAILABLE",reason:!key?"NO_API_KEY":"INSUFFICIENT_CANDLES"};
  try {
    const url=URLS.kronos.endsWith("/forecast")?URLS.kronos:URLS.kronos+"/forecast";
    const r=await getJson(url,{method:"POST",headers:{"content-type":"application/json","authorization":"Bearer "+key},body:JSON.stringify({timeframe:"M5",candles:m5.slice(-200),pred_len:12,sample_count:1})},45000);
    return {status:"AVAILABLE",model:r.model,forecastBias:r.forecastBias,forecastChange:r.forecastChange,forecastChangePct:r.forecastChangePct,latencyMs:r.latencyMs,barsUsed:r.barsUsed,volumeUsed:r.volumeUsed};
  } catch(e:any) { return {status:"ERROR",reason:String(e?.message||e).slice(0,240)}; }
}
function stats(closed:any[]) {
  const sum=(a:any[])=>{const wins=a.filter(x=>x.netUsd>0),loss=a.filter(x=>x.netUsd<0),gw=wins.reduce((v,x)=>v+x.netUsd,0),gl=Math.abs(loss.reduce((v,x)=>v+x.netUsd,0));return {trades:a.length,wins:wins.length,losses:loss.length,winRate:a.length?Number((wins.length/a.length*100).toFixed(2)):null,profitFactor:gl?Number((gw/gl).toFixed(3)):null,netUsd:Number(a.reduce((v,x)=>v+x.netUsd,0).toFixed(2))};};
  const equity=closed.reduce((a,x)=>{a.push((a.at(-1)||0)+Number(x.netUsd||0));return a;},[] as number[]);
  let peak=0,dd=0;for(const x of equity){peak=Math.max(peak,x);dd=Math.max(dd,peak-x);}
  return {...sum(closed),maxDrawdownUsd:Number(dd.toFixed(2)),BUY:sum(closed.filter(x=>x.direction==="BUY")),SELL:sum(closed.filter(x=>x.direction==="SELL")),byExpert:Object.fromEntries([...new Set(closed.map(x=>x.expert))].map(k=>[k,sum(closed.filter(x=>x.expert===k))]))};
}
function resetDaily(s:any) {const d=now().slice(0,10);if(s.daily?.date!==d)s.daily={date:d,trades:0,netUsd:0};}
function riskReason(c:Candidate,quote:any,a:number,s:any,cost:number,lotOz:number,maxRisk:number,dailyLimit:number) {
  const risk=Math.abs(c.entry-c.stopLoss),reward=Math.abs(c.target-c.entry),spread=n(quote.spread),spreadRatio=spread!==null?spread/a:null;
  const netRR=(reward-cost)/Math.max(0.0001,risk+cost);
  if(risk<=0||c.rr<1.5)return "INVALID_RISK_OR_RR";
  if(c.stopAtr<0.25||c.stopAtr>2.5)return "STOP_ATR_OUT_OF_RANGE";
  if(spreadRatio===null||spreadRatio>0.15)return "SPREAD_UNAVAILABLE_OR_HIGH";
  if(risk*lotOz>maxRisk)return "MAX_RISK_USD";
  if(netRR<=0)return "NEGATIVE_NET_REWARD_AFTER_COST";
  if(Number(s.daily.netUsd||0)<=-dailyLimit)return "DAILY_LOSS_LIMIT";
  if(s.openPositions.length>=4)return "MAX_OPEN_POSITIONS";
  if(s.openPositions.some((p:any)=>p.direction===c.direction&&p.expert===c.expert&&Math.abs(p.entry-c.entry)<a*0.35))return "DUPLICATE_OR_OVERLAP";
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
    let why=null,exit=null;
    if(stop){why="STOP";exit=p.stopLoss;}else if(target){why="TARGET";exit=p.target;}else if(Date.now()-Date.parse(p.openedAt)>=21600000){why="TIMEOUT";exit=bar.close;}
    if(!why){keep.push(p);continue;}
    const move=p.direction==="BUY"?exit-p.entry:p.entry-exit,gross=move*lotOz,net=gross-cost;
    const row={...p,closedAt:now(),exit,exitReason:why,grossUsd:Number(gross.toFixed(2)),costUsd:cost,netUsd:Number(net.toFixed(2)),stressNetUsd:Number((gross-0.74).toFixed(2)),resultR:Number((move/Math.max(0.0001,p.risk)).toFixed(3))};
    s.closedTrades.push(row);s.daily.trades++;s.daily.netUsd+=row.netUsd;
    event(s,{type:"PAPER_CLOSE",id:p.id,expert:p.expert,direction:p.direction,exitReason:why,netUsd:row.netUsd,resultR:row.resultR});
    console.log("AI_PAPER_CLOSE",JSON.stringify({id:p.id,expert:p.expert,direction:p.direction,why,netUsd:row.netUsd,resultR:row.resultR}));
  }
  s.openPositions=keep;
  if(s.closedTrades.length>2000)s.closedTrades=s.closedTrades.slice(-2000);
  const shadows=[];
  for(const p of s.shadowPositions) {
    if(Date.parse(bar.time)<=Date.parse(p.barTime)){shadows.push(p);continue;}
    const stop=p.direction==="BUY"?bar.low<=p.stopLoss:bar.high>=p.stopLoss,target=p.direction==="BUY"?bar.high>=p.target:bar.low<=p.target;
    let why=null,exit=null;if(stop){why="STOP";exit=p.stopLoss;}else if(target){why="TARGET";exit=p.target;}else if(Date.now()-Date.parse(p.createdAt)>=21600000){why="TIMEOUT";exit=bar.close;}
    if(!why){shadows.push(p);continue;}
    const move=p.direction==="BUY"?exit-p.entry:p.entry-exit,gross=move*lotOz;
    s.candidateHistory.push({type:"SHADOW_OUTCOME",id:p.id,expert:p.expert,direction:p.direction,outcome:why,netUsd:Number((gross-cost).toFixed(2)),stressNetUsd:Number((gross-0.74).toFixed(2)),resultR:Number((move/Math.max(.0001,p.risk)).toFixed(3))});
    event(s,{type:"SHADOW_OUTCOME",id:p.id,expert:p.expert,outcome:why,netUsd:Number((gross-cost).toFixed(2))});
  }
  s.shadowPositions=shadows;
}
async function runCycle(source="scheduled") {
  const s=await readState();resetDaily(s);
  const [market,chart,engine,trades]=await Promise.all([
    getJson(URLS.bridge+"/market-data",{},22000),
    getJson(URLS.chartLab+"/market-read?refresh=1",{},25000),
    getJson(URLS.intelligence+"/playbook-status",{},15000).catch(e=>({ok:false,error:String(e)})),
    getJson(URLS.intelligence+"/playbook-trades",{},15000).catch(e=>({ok:false,error:String(e)}))
  ]);
  if(!market.ok)throw new Error("BRIDGE_MARKET_DATA_NOT_OK");
  const bars=normalizeBars(market),quote={bid:n(market.bid),ask:n(market.ask),spread:n(market.spread)};
  const mid=quote.bid!==null&&quote.ask!==null?(quote.bid+quote.ask)/2:null,last=bars.M5.at(-1);
  const f=chart.freshness||{},fresh=chart.ok===true&&f.dataFresh===true&&Number(f.quoteAgeMinutes)<=10&&Number(f.m5BarAgeMinutes)<=20;
  const cost=n(Bun.env.PAPER_TRADE_COST_USD)??0.37,stress=n(Bun.env.PAPER_STRESS_COST_USD)??0.74;
  const lotSize=n(Bun.env.PAPER_LOT_SIZE)??0.01,contract=n(Bun.env.CONTRACT_OZ_PER_LOT)??100,lotOz=lotSize*contract;
  const maxRisk=n(Bun.env.MAX_TRADE_RISK_USD)??2,dailyLimit=n(Bun.env.DAILY_LOSS_LIMIT_USD)??2;
  const kronos=await fetchKronos(bars);
  const ref=engine.ok?engine:null;
  if(!last||bars.M5.length<30||!fresh||mid===null) {
    const report={at:now(),source,decision:"NO_TRADE",reason:!fresh?"STALE_OR_INVALID_MARKET_DATA":"INSUFFICIENT_M5_CANDLES",marketFresh:fresh,lastM5:last?.time||null,quote,kronos};
    s.lastCycleAt=now();s.lastReport=report;event(s,{type:"NO_TRADE",reason:report.reason});await writeState(s);return report;
  }
  await monitor(s,last,lotOz,cost);
  if(s.lastProcessedBar===last.time) {
    const report={at:now(),source,decision:"NO_NEW_BAR",reason:"BAR_ALREADY_PROCESSED",bar:last.time,openPositions:s.openPositions.length,kronos};
    s.lastCycleAt=now();s.lastReport=report;await writeState(s);return report;
  }
  const generated=generateCandidates(bars,chart,mid,kronos,ref,s.closedTrades);
  const accepted:any[]=[],rejected:any[]=[];
  for(const c of generated.candidates) {
    const why=riskReason(c,quote,generated.atr,s,cost,lotOz,maxRisk,dailyLimit);
    if(why) {
      rejected.push({id:c.id,expert:c.expert,direction:c.direction,reason:why,qualityScore:c.qualityScore});
      s.shadowPositions.push({id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,risk:c.risk,barTime:last.time,createdAt:now(),rejectedReason:why});
      event(s,{type:"CANDIDATE_REJECTED",id:c.id,expert:c.expert,direction:c.direction,reason:why,qualityScore:c.qualityScore});
      continue;
    }
    const p={id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,risk:c.risk,rr:c.rr,qualityScore:c.qualityScore,openedAt:now(),openedBarTime:last.time,regime:c.regime,features:c.features,kronosBias:kronos.forecastBias||null,lotOz,costUsd:cost,mfe:0,mae:0,mode:"PAPER"};
    s.openPositions.push(p);accepted.push({id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,rr:c.rr,qualityScore:c.qualityScore});
    event(s,{type:"PAPER_OPEN",id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,qualityScore:c.qualityScore});
    console.log("AI_PAPER_OPEN",JSON.stringify({id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,qualityScore:c.qualityScore}));
    if(s.openPositions.length>=4)break;
  }
  for(const c of generated.candidates)s.candidateHistory.push({at:now(),barTime:last.time,id:c.id,expert:c.expert,direction:c.direction,entry:c.entry,stopLoss:c.stopLoss,target:c.target,rr:c.rr,qualityScore:c.qualityScore,features:c.features,entered:accepted.some(x=>x.id===c.id),rejectionReason:rejected.find(x=>x.id===c.id)?.reason||null});
  if(s.candidateHistory.length>2500)s.candidateHistory=s.candidateHistory.slice(-2500);
  if(s.shadowPositions.length>500)s.shadowPositions=s.shadowPositions.slice(-500);
  const report={at:now(),source,decision:accepted.length?"PAPER_CANDIDATE_OPENED":"NO_TRADE",reason:accepted.length?"INDEPENDENT_EXPERTS_RISK_VALIDATED":generated.candidates.length?"CANDIDATES_REJECTED_BY_RISK":"NO_EXPERT_TRIGGER",
    bar:last.time,regime:generated.regime,h1Bias:generated.h1Bias,m15Bias:generated.m15Bias,marketFresh:fresh,quote,atrM5:generated.atr,
    expertsEvaluated:["LIQUIDITY_SWEEP_RECLAIM","BREAKOUT_CONTINUATION","TREND_PULLBACK","RANGE_REJECTION","MOMENTUM_IMPULSE","GEMINI_VISION"],
    candidatesGenerated:generated.candidates.length,accepted,rejected,openPositions:s.openPositions.length,shadowTracking:s.shadowPositions.length,
    kronos,referenceEngine:{status:engine.status,version:engine.engineVersion,plan:engine.plan?.decision||null,visionBlocked:engine.visionHealth?.entryBlocked??null},
    referenceTrades:trades.ok?trades.overall:null,costUsd:cost,stressCostUsd:stress,stats:stats(s.closedTrades),
    metaLabeler:s.metaLabeler,liveTrading:false,
    disclaimer:"PAPER only. QualityScore is a heuristic ranking, not a calibrated probability. Kronos is a forecast expert, not a probability model. XGBoost is not trained yet."};
  s.lastProcessedBar=last.time;s.lastCycleAt=now();s.lastReport=report;
  await writeState(s);event(s,{type:"CYCLE",bar:last.time,decision:report.decision,candidates:generated.candidates.length,accepted:accepted.length,rejected:rejected.length,regime:generated.regime});
  console.log("AI_ENGINE_CYCLE",JSON.stringify(report));
  return report;
}
async function status() {
  const s=await readState();
  return {ok:true,service:"Gold AI Intelligence Engine",version:VERSION,mode:"PAPER_ONLY",liveTrading:false,lastCycleAt:s.lastCycleAt,lastProcessedBar:s.lastProcessedBar,
    openPositions:s.openPositions,shadowTracking:s.shadowPositions.length,stats:stats(s.closedTrades),lastReport:s.lastReport,metaLabeler:s.metaLabeler,storage:"persistent-object-storage"};
}
let running=false;
const server=Bun.serve({port:PORT,hostname:"0.0.0.0",async fetch(req) {
  const u=new URL(req.url);
  if(req.method==="GET"&&(u.pathname==="/"||u.pathname==="/health"))return json({ok:true,service:"Gold AI Intelligence Engine",version:VERSION,mode:"PAPER_ONLY",liveTrading:false});
  if(req.method==="GET"&&u.pathname==="/status")try{return json(await status());}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  if(req.method==="GET"&&u.pathname==="/analysis")try{const s=await readState();return json(s.lastReport||{decision:"NO_TRADE",reason:"NO_CYCLE_YET"});}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  if(req.method==="GET"&&u.pathname==="/history")try{const s=await readState();return json({ok:true,events:s.events.slice(-120),candidates:s.candidateHistory.slice(-120),openPositions:s.openPositions,closedTrades:s.closedTrades.slice(-120),stats:stats(s.closedTrades)});}catch(e:any){return json({ok:false,error:String(e?.message||e)},503);}
  if((req.method==="GET"||req.method==="POST")&&u.pathname==="/sync") {
    if(running)return json({ok:true,status:"ALREADY_RUNNING"},202);
    running=true;try{return json(await runCycle("manual"));}catch(e:any){return json({ok:false,error:String(e?.message||e)},502);}finally{running=false;}
  }
  return json({ok:false,error:"NOT_FOUND",available:["/health","/status","/analysis","/history","/sync"]},404);
}});
console.log("AI_ENGINE_LISTENING",JSON.stringify({url:server.url,version:VERSION,mode:"PAPER_ONLY",storage:!!BUCKET,urls:URLS}));
void (async()=>{await sleep(12000);while(true){if(!running){running=true;try{await runCycle("scheduled");}catch(e:any){console.log("AI_ENGINE_LOOP_ERROR",String(e?.message||e));}finally{running=false;}}await sleep(300000);}})();
