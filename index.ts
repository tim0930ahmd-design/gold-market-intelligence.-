type Candle = { time: string; open: number; high: number; low: number; close: number; volume?: number };
type Frame = { timeframe: string; count: number; items: Candle[] };
const VERSION = "paper-integration-0.2.0";
const URLS = {
  bridge: "https://gold-ctrader-bridge-production.up.railway.app",
  intelligence: "https://gold-market-intelligence-production.up.railway.app",
  chartLab: "https://gold-chart-lab-production.up.railway.app",
  kronos: "https://gold-market-api-production.up.railway.app"
};
const TIMEFRAMES: Record<string, number> = { M5: 5, M15: 15, M30: 30, H1: 60, H4: 240 };
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
let snapshot: any = { syncedAt: null, sources: {}, market: null, referenceEngine: null, referenceTrades: null, analysis: null, errors: [] };
const n = (x: any) => Number.isFinite(Number(x)) ? Number(x) : null;
function cleanFrame(raw: any, tf: string): Candle[] {
  const duration = (TIMEFRAMES[tf] || 5) * 60000;
  const now = Date.now();
  return (Array.isArray(raw?.items) ? raw.items : [])
    .map((x: any) => ({ time: String(x.time || x.timestamp || ""), open: n(x.open), high: n(x.high), low: n(x.low), close: n(x.close), volume: n(x.volume) ?? undefined }))
    .filter((x: any) => Number.isFinite(Date.parse(x.time)) && [x.open,x.high,x.low,x.close].every(Number.isFinite) && x.high >= Math.max(x.open,x.close,x.low) && x.low <= Math.min(x.open,x.close,x.high) && Date.parse(x.time) + duration <= now)
    .sort((a: Candle,b: Candle) => Date.parse(a.time)-Date.parse(b.time)
    );
}
function atr(bars: Candle[], period = 14) {
  if (bars.length < period + 1) return null;
  const ranges = bars.slice(1).map((b,i) => Math.max(b.high-b.low,Math.abs(b.high-bars[i].close),Math.abs(b.low-bars[i].close))).slice(-period);
  return ranges.reduce((a,b)=>a+b,0)/ranges.length;
}
function swingLevels(bars: Candle[]) {
  const highs: any[] = [], lows: any[] = [];
  for (let i=2;i<bars.length-2;i++) {
    const b=bars[i], left=bars.slice(i-2,i), right=bars.slice(i+1,i+3);
    if(left.every(x=>b.high>x.high) && right.every(x=>b.high>=x.high)) highs.push({price:b.high,time:bars[i].time});
    if(left.every(x=>b.low<x.low) && right.every(x=>b.low<=x.low)) lows.push({price:b.low,time:bars[i].time});
  }
  return { highs:highs.slice(-5), lows:lows.slice(-5), lastHigh:highs.at(-1)||null, previousHigh:highs.at(-2)||null, lastLow:lows.at(-1)||null, previousLow:lows.at(-2)||null };
}
function frameRead(bars: Candle[]) {
  if(bars.length<30) return {ready:false,count:bars.length,structure:"INSUFFICIENT_DATA",regime:"UNKNOWN",atr14:atr(bars)};
  const s=swingLevels(bars), last=bars.at(-1)!;
  let structure="MIXED";
  if(s.lastHigh&&s.previousHigh&&s.lastLow&&s.previousLow) {
    if(s.lastHigh.price>s.previousHigh.price && s.lastLow.price>s.previousLow.price) structure="HH_HL_BULLISH";
    else if(s.lastHigh.price<s.previousHigh.price && s.lastLow.price<s.previousLow.price) structure="LH_LL_BEARISH";
  }
  const a=atr(bars), recent=bars.slice(-20), range=Math.max(...recent.map(x=>x.high))-Math.min(...recent.map(x=>x.low));
  const avgRange=recent.reduce((v,x)=>v+(x.high-x.low),0)/recent.length;
  const regime=a && avgRange/a<0.65 ? "COMPRESSION" : a && range/a>5 ? "EXPANSION" : structure.includes("BULLISH")||structure.includes("BEARISH") ? "TREND" : "RANGE_OR_TRANSITION";
  const prior=bars.slice(-21,-1), priorHigh=prior.length?Math.max(...prior.map(x=>x.high)):null, priorLow=prior.length?Math.min(...prior.map(x=>x.low)):null;
  const breakout=priorHigh!==null&&last.close>priorHigh?"BULLISH":priorLow!==null&&last.close<priorLow?"BEARISH":null;
  const liquiditySweep=s.lastHigh&&last.high>s.lastHigh.price&&last.close<s.lastHigh.price?"BUY_SIDE_SWEEP_REJECTED":s.lastLow&&last.low<s.lastLow.price&&last.close>s.lastLow.price?"SELL_SIDE_SWEEP_RECLAIMED":null;
  return {ready:true,count:bars.length,lastBar:last.time,lastClose:last.close,atr14:a,structure,regime,swings:s,breakout,liquiditySweep,range20:{high:priorHigh,low:priorLow}};
}
function buildAnalysis(data: any) {
  const raw=data?.multiTimeframe||data?.candles||{}, frames: any={};
  for(const tf of Object.keys(TIMEFRAMES)) frames[tf]=frameRead(cleanFrame(raw[tf],tf));
  const candidates:any[]=[];
  const h1=frames.H1,m15=frames.M15,m5=frames.M5;
  if(h1.ready&&m15.ready&&m15.structure==="HH_HL_BULLISH") candidates.push({expert:"TREND_CONTINUATION",side:"BUY",timeframe:"M15",evidence:"M15 confirmed HH/HL; H1 context available"});
  if(h1.ready&&m15.ready&&m15.structure==="LH_LL_BEARISH") candidates.push({expert:"TREND_CONTINUATION",side:"SELL",timeframe:"M15",evidence:"M15 confirmed LH/LL; H1 context available"});
  if(m15.ready&&m15.liquiditySweep==="SELL_SIDE_SWEEP_RECLAIMED") candidates.push({expert:"LIQUIDITY_SWEEP",side:"BUY",timeframe:"M15",evidence:"Price swept a confirmed swing low and reclaimed it"});
  if(m15.ready&&m15.liquiditySweep==="BUY_SIDE_SWEEP_REJECTED") candidates.push({expert:"LIQUIDITY_SWEEP",side:"SELL",timeframe:"M15",evidence:"Price swept a confirmed swing high and rejected it"});
  if(m15.ready&&m15.breakout) candidates.push({expert:"BREAKOUT",side:m15.breakout==="BULLISH"?"BUY":"SELL",timeframe:"M15",evidence:"Close beyond the prior 20-bar range"});
  return {generatedAt:new Date().toISOString(),market:"XAUUSD",quote:{bid:n(data?.bid),ask:n(data?.ask),spread:n(data?.spread),spotTime:data?.spotTime||null},frames,independentExpertCandidates:candidates,decision:"NO_TRADE",decisionReason:"Shadow integration only: candidates are recorded, but no calibrated probability model or validated after-cost edge is connected to this service.",liveTrading:false};
}
async function getJson(url:string) {
  const r=await fetch(url,{headers:{accept:"application/json"},signal:AbortSignal.timeout(18000)});
  let body:any; try{body=await r.json();}catch{body={error:"INVALID_JSON"};}
  return {ok:r.ok,status:r.status,body};
}
async function syncAll() {
  const errors:string[]=[];
  const [market,engine,trades,chart,kronos]=await Promise.all([
    getJson(URLS.bridge+"/market-data").catch(e=>({ok:false,status:0,body:{error:String(e)}})),
    getJson(URLS.intelligence+"/playbook-status").catch(e=>({ok:false,status:0,body:{error:String(e)}})),
    getJson(URLS.intelligence+"/playbook-trades").catch(e=>({ok:false,status:0,body:{error:String(e)}})),
    fetch(URLS.chartLab+"/",{signal:AbortSignal.timeout(12000)}).then(r=>({ok:r.ok,status:r.status})).catch(e=>({ok:false,status:0,error:String(e)})),
    getJson(URLS.kronos+"/health").catch(e=>({ok:false,status:0,body:{error:String(e)}}))
  ]);
  if(!market.ok||!market.body?.ok) errors.push("CTRADER_BRIDGE_UNAVAILABLE");
  if(!engine.ok||!engine.body?.ok) errors.push("PLAYBOOK_STATUS_UNAVAILABLE");
  if(!trades.ok||!trades.body?.ok) errors.push("PLAYBOOK_TRADES_UNAVAILABLE");
  if(!chart.ok) errors.push("CHART_LAB_UNAVAILABLE");
  if(!kronos.ok) errors.push("KRONOS_MODEL_NOT_READY_OR_UNAVAILABLE");
  snapshot={syncedAt:new Date().toISOString(),sources:{
    bridge:{ok:market.ok&&!!market.body?.ok,status:market.status,source:market.body?.source||null,symbol:market.body?.symbol||null,accountEnvironment:market.body?.accountEnvironment||null,quoteFresh:market.body?.quoteFresh??null,quoteValid:market.body?.quoteValid??null},
    intelligence:{ok:engine.ok&&!!engine.body?.ok,status:engine.status,engine:engine.body?.engine||null,version:engine.body?.engineVersion||null,planStatus:engine.body?.status||null},
    trades:{ok:trades.ok&&!!trades.body?.ok,status:trades.status,closedTrades:trades.body?.closedTrades??null,overall:trades.body?.overall||null},
    chartLab:{ok:chart.ok,status:chart.status},
    kronos:{ok:kronos.ok,status:kronos.status,model:kronos.body?.model||null,detail:kronos.body?.detail||kronos.body?.error||null}
  },market:market.body?.ok?market.body:null,referenceEngine:engine.body?.ok?engine.body:null,referenceTrades:trades.body?.ok?trades.body:null,analysis:market.body?.ok?buildAnalysis(market.body):null,errors};
  return snapshot;
}
const server=Bun.serve({port:Number(Bun.env.PORT||3000),hostname:"0.0.0.0",async fetch(req){
  const url=new URL(req.url);
  if(req.method==="GET"&&(url.pathname==="/"||url.pathname==="/health")) return json({ok:true,service:"Gold AI PAPER Integration",mode:"PAPER_SHADOW",engine_version:VERSION,connectedSources:Object.keys(URLS),liveTrading:false});
  if(req.method==="GET"&&url.pathname==="/status") return json({ok:true,engine_version:VERSION,mode:"PAPER_SHADOW",lastSync:snapshot.syncedAt,sources:snapshot.sources,errors:snapshot.errors,decision:snapshot.analysis?.decision||"WAITING_FOR_SYNC",persistence:"memory_only",liveTrading:false});
  if(req.method==="GET"&&url.pathname==="/sync") return json(await syncAll());
  if(req.method==="GET"&&url.pathname==="/analysis") return snapshot.analysis?json(snapshot.analysis):json({ok:false,error:"NOT_SYNCED",next:"GET /sync"},503);
  if(req.method==="GET"&&url.pathname==="/reference") return json({engine:snapshot.referenceEngine,trades:snapshot.referenceTrades});
  if(req.method==="GET"&&url.pathname==="/candles") return json({syncedAt:snapshot.syncedAt,multiTimeframe:snapshot.market?.multiTimeframe||null});
  if(req.method==="GET"&&url.pathname==="/analyze") return json(snapshot.analysis||{decision:"NO_TRADE",reason:"Run GET /sync first."});
  return json({error:"not_found",available:["/health","/status","/sync","/analysis","/reference","/candles","/analyze"]},404);
}});
console.log("Gold AI PAPER integration listening on "+server.url);\nvoid syncAll().then(()=>console.log("INTEGRATION_SYNC_OK",snapshot.syncedAt,JSON.stringify(snapshot.errors))).catch(e=>console.log("INTEGRATION_SYNC_ERROR",String(e)));\nsetInterval(()=>{void syncAll().then(()=>console.log("INTEGRATION_SYNC_OK",snapshot.syncedAt,JSON.stringify(snapshot.errors))).catch(e=>console.log("INTEGRATION_SYNC_ERROR",String(e)));},300000);
