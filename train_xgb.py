#!/usr/bin/env python3
"""XAUUSD M1/M5 -> M5 first-touch labels -> purged walk-forward XGBoost.
Features use bars through t only; outcomes start at t+1. Same-candle TP/SL = STOP first.
"""
import argparse, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

META_FEATURES=["qualityScore","stopAtr","rr","regimeConfidence","atrRatio","efficiency","bodyAtr","emaAlignment","slope","spreadAtr","isBuy","expertSweep","expertBreakout","expertRetest","expertContinuation","expertReversal","expertPullback","expertRange","expertMomentum","expertVision","regimeTrend","regimeRange","regimeCompression","regimeExpansion","regimeImpulse"]
CLASSES=("TARGET","STOP","TIMEOUT"); HORIZON=12; STOP_ATR=1.0; RR=1.8

def ema(x,p):
    x=np.asarray(x,float); out=np.full(len(x),np.nan)
    if len(x)<p:return out
    v=float(np.mean(x[:p])); out[p-1]=v; k=2/(p+1)
    for i in range(p,len(x)): v=x[i]*k+v*(1-k); out[i]=v
    return out

def load_bars(path,naive_offset_hours=0):
    d=pd.read_csv(path); names={c.lower().strip():c for c in d.columns}
    def find(*xs):
        return next((names[x] for x in xs if x in names),None)
    tc=find("time","timestamp","datetime","date")
    if not tc: raise ValueError("Missing time/timestamp column")
    cols={k:find(k) for k in ("open","high","low","close")}
    if any(v is None for v in cols.values()): raise ValueError("CSV needs OHLC columns")
    parsed=pd.to_datetime(d[tc],errors="coerce")
    is_naive=parsed.dt.tz is None
    if is_naive:
        parsed=parsed.dt.tz_localize("UTC")
        if naive_offset_hours: parsed=parsed-pd.Timedelta(hours=naive_offset_hours)
    x=pd.DataFrame({"time":parsed,**{k:pd.to_numeric(d[v],errors="coerce") for k,v in cols.items()}})
    sc=find("spread","spread_usd")
    if sc:x["spread"]=pd.to_numeric(d[sc],errors="coerce")
    x=x.dropna(subset=["time","open","high","low","close"]).sort_values("time").drop_duplicates("time").set_index("time")
    valid=(x.low>0)&(x.high>=x[["open","close","low"]].max(axis=1))&(x.low<=x[["open","close","high"]].min(axis=1))
    x=x[valid]
    if len(x)<300: raise ValueError(f"Only {len(x)} valid rows; need >=300")
    if x.index.to_series().diff().median()<=pd.Timedelta(minutes=1.5):
        agg={"open":"first","high":"max","low":"min","close":"last"}
        if "spread" in x:agg["spread"]="mean"
        x=x.resample("5min",label="left",closed="left").agg(agg).dropna(subset=["open","high","low","close"])
    return x.reset_index().sort_values("time").reset_index(drop=True)

def indicators(d):
    prev=d.close.shift(1)
    tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    a14=tr.rolling(14,min_periods=14).mean().to_numpy()
    a50=tr.rolling(50,min_periods=50).mean().to_numpy()
    c=d.close.to_numpy(float); e9=ema(c,9); e21=ema(c,21)
    return a14,a50,e9,e21

def regime_at(d,i,a14,a50,e9,e21):
    if i<54 or not np.isfinite(a14[i]) or not np.isfinite(a50[i]) or a14[i]<=0 or a50[i]<=0:
        return "TRANSITION",0.,{"atrRatio":1.,"efficiency":0.,"bodyAtr":0.,"emaAlignment":0.,"slope":0.}
    w=d.iloc[max(0,i-99):i+1]; c=w.close.to_numpy(float); a=a14[i]
    path=np.abs(np.diff(c[-20:])).sum(); eff=abs(c[-1]-c[-21])/path if path>0 else 0.
    ratio=a14[i]/a50[i]; body=abs(float(w.close.iloc[-1]-w.open.iloc[-1]))/a
    slope=(e21[i]-e21[i-5])/a; align=(e9[i]-e21[i])/a
    width=(w.high.iloc[-20:].max()-w.low.iloc[-20:].min())/a
    if ratio<.72:r,cf="COMPRESSION",max(.45,min(.95,(.9-ratio)*2.2))
    elif ratio>1.35:r,cf="VOLATILITY_EXPANSION",max(.45,min(.95,(ratio-1.1)*1.4))
    elif body>1.25:r,cf="IMPULSE",max(.55,min(.95,.55+(body-1.25)*.22))
    elif eff>.42 and align>.08 and slope>.025:r,cf="TREND_UP",max(.5,min(.96,.48+eff*.45+abs(align)*.18))
    elif eff>.42 and align<-.08 and slope<-.025:r,cf="TREND_DOWN",max(.5,min(.96,.48+eff*.45+abs(align)*.18))
    elif eff<.25 and width<8:r,cf="RANGE",max(.5,min(.9,.58+(.25-eff)*1.1))
    else:r,cf="TRANSITION",max(.45,min(.72,.55+abs(eff-.32)*.25))
    return r,cf,{"atrRatio":ratio,"efficiency":eff,"bodyAtr":body,"emaAlignment":align,"slope":slope}

def candidate_rows(d,i,regime,conf,f,a,ema21):
    row=d.iloc[i]; prev=d.iloc[i-1]; hist=d.iloc[i-12:i]
    if len(hist)<12:return []
    ph=float(hist.high.max()); pl=float(hist.low.min()); body=abs(row.close-row.open)
    pos=(row.close-row.low)/max(1e-9,row.high-row.low); up=row.close>row.open; down=row.close<row.open
    spread_raw=row.get("spread",0); spread=float(spread_raw) if pd.notna(spread_raw) else 0.; spr=spread/a if spread>0 else 0.
    out=[]
    def add(expert,side,bonus):
        flags=[int("SWEEP" in expert),int("BREAKOUT" in expert),int("RETEST" in expert),int("CONTINUATION" in expert),int("REVERSAL" in expert),int("PULLBACK" in expert),int("RANGE" in expert),int("MOMENTUM" in expert),0]
        rf=[int(regime.startswith("TREND")),int(regime=="RANGE"),int(regime=="COMPRESSION"),int(regime=="VOLATILITY_EXPANSION"),int(regime=="IMPULSE")]
        score=50+bonus
        if side=="BUY" and regime=="TREND_UP":score+=8
        if side=="SELL" and regime=="TREND_DOWN":score+=8
        x=[score,STOP_ATR,RR,conf,f["atrRatio"],f["efficiency"],f["bodyAtr"],f["emaAlignment"],f["slope"],spr,int(side=="BUY"),*flags,*rf]
        if len(x)!=25:raise AssertionError("Meta feature order/count mismatch")
        out.append((expert,side,x))
    if row.low<pl and row.close>pl and up:add("LIQUIDITY_SWEEP_RECLAIM","BUY",18)
    if row.high>ph and row.close<ph and down:add("LIQUIDITY_SWEEP_RECLAIM","SELL",18)
    if row.close>ph and prev.close<=ph:add("BREAKOUT_CONTINUATION","BUY",14)
    if row.close<pl and prev.close>=pl:add("BREAKOUT_CONTINUATION","SELL",14)
    if regime=="TREND_UP" and row.close>ph and body>=a*.35:add("TREND_CONTINUATION","BUY",15)
    if regime=="TREND_DOWN" and row.close<pl and body>=a*.35:add("TREND_CONTINUATION","SELL",15)
    if regime=="RANGE" and pos>.78 and down:add("RANGE_REJECTION","SELL",0)
    if regime=="RANGE" and pos<.22 and up:add("RANGE_REJECTION","BUY",0)
    if body>=a*.8 and pos>=.78:add("MOMENTUM_IMPULSE","BUY",8)
    if body>=a*.8 and pos<=.22:add("MOMENTUM_IMPULSE","SELL",8)
    if abs(row.close-ema21)<=a*.35 and regime=="TREND_UP" and up:add("TREND_PULLBACK","BUY",12)
    if abs(row.close-ema21)<=a*.35 and regime=="TREND_DOWN" and down:add("TREND_PULLBACK","SELL",12)
    if row.low<pl and row.close>pl and row.close>prev.high and up:add("REVERSAL","BUY",17)
    if row.high>ph and row.close<ph and row.close<prev.low and down:add("REVERSAL","SELL",17)
    return out

def build_samples(d):
    a14,a50,e9,e21=indicators(d); samples=[]
    for i in range(60,len(d)-HORIZON-1):
        regime,conf,f=regime_at(d,i,a14,a50,e9,e21); a=a14[i]
        if not np.isfinite(a) or a<=0:continue
        for expert,side,x in candidate_rows(d,i,regime,conf,f,a,e21[i]):
            entry=float(d.open.iloc[i+1]); dist=STOP_ATR*a
            stop=entry-dist if side=="BUY" else entry+dist
            target=entry+dist*RR if side=="BUY" else entry-dist*RR
            label="TIMEOUT"; exitp=float(d.close.iloc[i+HORIZON]); exit_i=i+HORIZON
            for j in range(i+1,i+HORIZON+1):
                b=d.iloc[j]
                hs=(b.low<=stop) if side=="BUY" else (b.high>=stop)
                ht=(b.high>=target) if side=="BUY" else (b.low<=target)
                if hs:label="STOP";exitp=stop;exit_i=j;break
                if ht:label="TARGET";exitp=target;exit_i=j;break
            signed=(exitp-entry) if side=="BUY" else (entry-exitp)
            tr=float(signed/dist) if label=="TIMEOUT" else (RR if label=="TARGET" else -1.)
            samples.append({"time":d.time.iloc[i],"exitTime":d.time.iloc[exit_i],"entry":entry,"stop":stop,"target":target,"risk":dist,"atr":a,"zoneLow":min(entry,stop,target),"zoneHigh":max(entry,stop,target),"expert":expert,"direction":side,"label":label,"timeoutR":tr,"features":x,"exitIndex":exit_i})
    return samples

def tree_json(n):
    if "leaf" in n:return {"nodeid":int(n["nodeid"]),"leaf":float(n["leaf"])}
    return {"nodeid":int(n["nodeid"]),"split":str(n["split"]),"split_condition":float(n["split_condition"]),"yes":int(n["yes"]),"no":int(n["no"]),"children":[tree_json(c) for c in n["children"]]}

def fit_models(rows,XGBClassifier,seed):
    out={}
    for side in ("BUY","SELL"):
        side_rows=[r for r in rows if r["direction"]==side]
        if len(side_rows)<100:raise ValueError(f"Not enough {side} samples for separate model: {len(side_rows)}")
        out[side]={}
        for cls in CLASSES:
            y=np.array([r["label"]==cls for r in side_rows],dtype=int)
            m=XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.9,reg_lambda=2,objective="binary:logistic",eval_metric="logloss",n_jobs=4,random_state=seed,base_score=.5)
            m.fit(np.asarray([r["features"] for r in side_rows]),y);out[side][cls]=m
    return out

def probs(models,x,side):
    side_models=models[side]
    p={k:float(m.predict_proba(np.asarray(x).reshape(1,-1))[0,1]) for k,m in side_models.items()}; z=sum(p.values())
    return {k:v/z for k,v in p.items()} if z>0 else {k:1/3 for k in CLASSES}

def evaluate(models,rows,cost,timeout_mean,max_risk=2.,max_open=4):
    pnl=[]; active=[]
    for r in sorted(rows,key=lambda z:(pd.Timestamp(z["time"]),-z["risk"])):
        t=pd.Timestamp(r["time"]); active=[p for p in active if pd.Timestamp(p["exitTime"])>t]
        if len(active)>=max_open or any(max(p["zoneLow"],r["zoneLow"])<=min(p["zoneHigh"],r["zoneHigh"]) for p in active):continue
        p=probs(models,r["features"],r["direction"]); risk_usd=min(float(r["risk"]),max_risk)
        tm=float(timeout_mean.get(r["direction"],0.)) if isinstance(timeout_mean,dict) else float(timeout_mean)
        gross_ev=(p["TARGET"]*RR-p["STOP"]+p["TIMEOUT"]*tm)*risk_usd
        if gross_ev-cost>0 and gross_ev-.74>0:
            actual=(RR if r["label"]=="TARGET" else -1. if r["label"]=="STOP" else r["timeoutR"])*risk_usd-cost
            pnl.append(actual); active.append(r)
    if not pnl:return {"trades":0,"wins":0,"netUsd":0.,"profitFactor":0.,"maxDrawdownUsd":0.}
    a=np.asarray(pnl); gp=float(a[a>0].sum()); gl=float(abs(a[a<0].sum())); eq=np.cumsum(a); peak=np.maximum.accumulate(np.r_[0.,eq])
    return {"trades":len(a),"wins":int((a>0).sum()),"netUsd":round(float(a.sum()),4),"profitFactor":round(gp/gl,4) if gl else None,"maxDrawdownUsd":round(float(np.max(peak-np.r_[0.,eq])),4)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="M1/M5 OHLC CSV file or directory")
    ap.add_argument("--naive-offset-hours",type=float,default=0.,help="For naive fixed-UTC-offset source timestamps: subtract this local offset to convert to UTC; HistData fixed UTC-5 requires -5.")
    ap.add_argument("--output",default="gold-ai-xgb-model.json")
    ap.add_argument("--min-oos-trades",type=int,default=100)
    ap.add_argument("--max-dd-usd",type=float,default=10.)
    ap.add_argument("--max-risk-usd",type=float,default=2.)
    ap.add_argument("--max-open-positions",type=int,default=4)
    ap.add_argument("--confirm-runtime-context-parity",action="store_true",help="Only set after confirming the OHLC candidate proxy reproduces runtime candidate context.")
    args=ap.parse_args()
    from xgboost import XGBClassifier
    source=Path(args.input); files=[source] if source.is_file() else sorted(source.glob("*.csv"))
    if not files:raise SystemExit("No CSV files found")
    d=pd.concat([load_bars(p,args.naive_offset_hours) for p in files],ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    rows=build_samples(d); rows.sort(key=lambda r:r["time"]); n=len(rows)
    if n<600:raise SystemExit(f"Only {n} expert event samples; need at least 600.")
    if any(sum(r["direction"]==side for r in rows)<200 for side in ("BUY","SELL")):raise SystemExit("Need at least 200 candidate samples for each of BUY and SELL.")
    unique_times=sorted(set(pd.Timestamp(r["time"]) for r in rows))
    if len(unique_times)<200:raise SystemExit("Insufficient distinct timestamps for purged walk-forward.")
    purge_delta=pd.Timedelta(minutes=5*HORIZON)
    oos_time=unique_times[int(len(unique_times)*.8)]
    train=[r for r in rows if pd.Timestamp(r["time"])<oos_time-purge_delta]
    oos=[r for r in rows if pd.Timestamp(r["time"])>=oos_time]
    if len(oos)<args.min_oos_trades:raise SystemExit(f"Only {len(oos)} OOS samples; minimum is {args.min_oos_trades}.")
    wf=[]
    for frac in (.50,.60,.70):
        cut_time=unique_times[int(len(unique_times)*frac)]
        end_time=unique_times[min(int(len(unique_times)*(frac+.10)),int(len(unique_times)*.8)-1)]
        tr=[r for r in rows if pd.Timestamp(r["time"])<cut_time-purge_delta]
        va=[r for r in rows if cut_time<=pd.Timestamp(r["time"])<end_time]
        if len(tr)<200 or len(va)<40 or any(sum(r["direction"]==side for r in tr)<100 for side in ("BUY","SELL")):continue
        models=fit_models(tr,XGBClassifier,41+len(wf))
        timeout_mean={side:(float(np.mean([r["timeoutR"] for r in tr if r["direction"]==side and r["label"]=="TIMEOUT"])) if any(r["direction"]==side and r["label"]=="TIMEOUT" for r in tr) else 0.) for side in ("BUY","SELL")}
        b=evaluate(models,va,.37,timeout_mean,args.max_risk_usd,args.max_open_positions)
        s=evaluate(models,va,.74,timeout_mean,args.max_risk_usd,args.max_open_positions)
        wf.append({"trainSamples":len(tr),"validationSamples":len(va),"cost037":b,"cost074":s,"purgeMinutes":60})
    final=fit_models(train,XGBClassifier,73)
    timeout_by_side={side:(float(np.mean([r["timeoutR"] for r in train if r["direction"]==side and r["label"]=="TIMEOUT"])) if any(r["direction"]==side and r["label"]=="TIMEOUT" for r in train) else 0.) for side in ("BUY","SELL")}
    # Evaluate separately by direction so timeout EV uses that side's historical timeout payoff.
    def eval_side(rows,cost,side):
        subset=[r for r in rows if r["direction"]==side]; pnl=[]; active=[]
        for r in sorted(subset,key=lambda z:pd.Timestamp(z["time"])):
            t=pd.Timestamp(r["time"]); active=[p for p in active if pd.Timestamp(p["exitTime"])>t]
            if len(active)>=args.max_open_positions or any(max(p["zoneLow"],r["zoneLow"])<=min(p["zoneHigh"],r["zoneHigh"]) for p in active):continue
            p=probs(final,r["features"],side); risk_usd=min(float(r["risk"]),args.max_risk_usd)
            gross=(p["TARGET"]*RR-p["STOP"]+p["TIMEOUT"]*timeout_by_side[side])*risk_usd
            if gross-cost>0 and gross-.74>0:
                realized=(RR if r["label"]=="TARGET" else -1. if r["label"]=="STOP" else r["timeoutR"])*risk_usd-cost
                pnl.append(realized);active.append(r)
        if not pnl:return {"trades":0,"wins":0,"netUsd":0.,"profitFactor":0.,"maxDrawdownUsd":0.}
        a=np.asarray(pnl);gp=float(a[a>0].sum());gl=float(abs(a[a<0].sum()));eq=np.cumsum(a);peak=np.maximum.accumulate(np.r_[0.,eq])
        return {"trades":len(a),"wins":int((a>0).sum()),"netUsd":round(float(a.sum()),4),"profitFactor":round(gp/gl,4) if gl else None,"maxDrawdownUsd":round(float(np.max(peak-np.r_[0.,eq])),4)}
    oos_metrics={side:{"cost037":eval_side(oos,.37,side),"cost074":eval_side(oos,.74,side)} for side in ("BUY","SELL")}
    oos_combined_037=evaluate(final,oos,.37,timeout_by_side,args.max_risk_usd,args.max_open_positions)
    oos_combined_074=evaluate(final,oos,.74,timeout_by_side,args.max_risk_usd,args.max_open_positions)
    wf_ok=len(wf)==3 and all((f["cost037"]["profitFactor"] or 0)>=1.0 and (f["cost074"]["profitFactor"] or 0)>=1.0 and f["cost037"]["netUsd"]>0 and f["cost074"]["netUsd"]>0 for f in wf)
    oos_ok=all(m["cost037"]["trades"]>=args.min_oos_trades//2 and m["cost074"]["trades"]>=args.min_oos_trades//2 and (m["cost037"]["profitFactor"] or 0)>=1.25 and (m["cost074"]["profitFactor"] or 0)>=1.25 and m["cost037"]["netUsd"]>0 and m["cost074"]["netUsd"]>0 and m["cost037"]["maxDrawdownUsd"]<=args.max_dd_usd and m["cost074"]["maxDrawdownUsd"]<=args.max_dd_usd for m in oos_metrics.values())
    models_json={}
    for side,side_models in final.items():
        models_json[side]={}
        for cls,m in side_models.items():
            models_json[side][cls]={"baseMargin":0.,"plattA":1.,"plattB":0.,"trees":[tree_json(json.loads(t)) for t in m.get_booster().get_dump(dump_format="json")]}
    runtime_context_parity=bool(args.confirm_runtime_context_parity)
    combined_ok=(oos_combined_037["trades"]>=args.min_oos_trades and oos_combined_074["trades"]>=args.min_oos_trades and (oos_combined_037["profitFactor"] or 0)>=1.25 and (oos_combined_074["profitFactor"] or 0)>=1.25 and oos_combined_037["netUsd"]>0 and oos_combined_074["netUsd"]>0 and oos_combined_037["maxDrawdownUsd"]<=args.max_dd_usd and oos_combined_074["maxDrawdownUsd"]<=args.max_dd_usd)
    passed=bool(wf_ok and oos_ok and combined_ok and runtime_context_parity)
    artifact={"kind":"xgboost-one-vs-rest-v1","modelVersion":"gold-xgb-0.7.0","featureCount":25,"featureNames":META_FEATURES,"modelsBySide":models_json,"timeoutMeanR":timeout_by_side,"trainSamples":len(train),"validationSamples":len(oos),"trainedAt":datetime.now(timezone.utc).isoformat(),"validated":passed,"validation":{"passed":passed,"walkForwardFolds":wf,"oosByDirection":oos_metrics,"oosCombinedCost037":oos_combined_037,"oosCombinedCost074":oos_combined_074,"purgeBars":purge,"sameBarPolicy":"STOP_FIRST","featuresUseOnlyCurrentAndPastBars":True,"labelHorizonBars":HORIZON,"featureNamesExact":META_FEATURES==["qualityScore","stopAtr","rr","regimeConfidence","atrRatio","efficiency","bodyAtr","emaAlignment","slope","spreadAtr","isBuy","expertSweep","expertBreakout","expertRetest","expertContinuation","expertReversal","expertPullback","expertRange","expertMomentum","expertVision","regimeTrend","regimeRange","regimeCompression","regimeExpansion","regimeImpulse"],"runtimeContextParityConfirmed":runtime_context_parity,"candidateGenerator":"OHLC_ONLY_PROXY","maxTradeRiskUsd":args.max_risk_usd,"maxOpenPositions":args.max_open_positions}}
    Path(args.output).write_text(json.dumps(artifact,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"samples":n,"train":len(train),"oos":len(oos),"walkForwardFolds":len(wf),"validated":passed,"oosByDirection":oos_metrics,"oosCombinedCost037":oos_combined_037,"oosCombinedCost074":oos_combined_074,"output":args.output},indent=2))
    if not passed:print("NOT VALIDATED: saved artifact has validated=false and runtime will ignore it.")

if __name__=="__main__":main()
