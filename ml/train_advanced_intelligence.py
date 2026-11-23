#!/usr/bin/env python3
"""Train validated AI discovery + scenario models from forward-only PAPER learning samples.

Input: persistent PAPER state JSON containing aiLearningSamples.
Artifacts are written only when chronological holdout gates pass.
"""
import argparse, json, os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

CLASSES=("UP","DOWN","RANGE")

def chronological(rows):
    rows=sorted(rows,key=lambda r:r["barTime"])
    n=len(rows); a=int(n*.70); b=int(n*.85)
    return rows[:a],rows[a:b],rows[b:]

def fit_binary(rows,side,cost,stress_cost,lot_oz):
    rows=[r for r in rows if r.get("discovery",{}).get(side) in ("TARGET","STOP","TIMEOUT")]
    if len(rows)<500:return {"status":"INSUFFICIENT_SAMPLES","samples":len(rows)}
    tr,cal,te=chronological(rows)
    if min(len(tr),len(cal),len(te))<50:return {"status":"INSUFFICIENT_TIME_SPLIT","samples":len(rows)}
    X=np.asarray([r["features"] for r in tr],float); y=np.asarray([r["discovery"][side]=="TARGET" for r in tr],int)
    Xc=np.asarray([r["features"] for r in cal],float); yc=np.asarray([r["discovery"][side]=="TARGET" for r in cal],int)
    Xt=np.asarray([r["features"] for r in te],float); yt=np.asarray([r["discovery"][side]=="TARGET" for r in te],int)
    if len(np.unique(y))<2 or len(np.unique(yc))<2 or len(np.unique(yt))<2:return {"status":"MISSING_CLASS_IN_SPLIT","samples":len(rows)}
    scaler=StandardScaler().fit(X); model=LogisticRegression(C=.25,max_iter=1500,class_weight="balanced",random_state=17)
    model.fit(scaler.transform(X),y)
    cal_margins=model.decision_function(scaler.transform(Xc))
    calibrator=LogisticRegression(C=1e6,solver="lbfgs",max_iter=1000)
    calibrator.fit(cal_margins.reshape(-1,1),yc)
    test_margins=model.decision_function(scaler.transform(Xt))
    pc=calibrator.predict_proba(cal_margins.reshape(-1,1))[:,1]
    pt=calibrator.predict_proba(test_margins.reshape(-1,1))[:,1]
    brier=float(brier_score_loss(yt,pt)); base=float(brier_score_loss(yt,np.full(len(yt),yt.mean())))
    # Select a probability threshold on the middle chronological slice only.
    thresholds=np.arange(.50,.901,.025); choices=[]
    for threshold in thresholds:
        ix=np.where(pc>=threshold)[0]
        if len(ix)<30:continue
        # Barrier payoffs: +1 ATR for TARGET, -0.75 ATR for STOP, zero gross for TIMEOUT.
        atrs=np.asarray([float(cal[i].get("atr",0)) for i in ix])
        gross=np.asarray([atrs[j]*lot_oz if cal[i]["discovery"][side]=="TARGET" else (-.75*atrs[j]*lot_oz if cal[i]["discovery"][side]=="STOP" else 0.0) for j,i in enumerate(ix)])
        net=gross-cost
        loss=np.abs(net[net<0]).sum(); gain=net[net>0].sum()
        pf=gain/loss if loss>0 else 0
        stress_net=gross-stress_cost
        if net.mean()>0 and stress_net.mean()>0 and pf>=1.25:
            choices.append((float(net.mean()),float(threshold),len(ix),float(pf),float(stress_net.mean())))
    if not choices:return {"status":"NO_VALIDATED_POSITIVE_EV_THRESHOLD","samples":len(rows),"brier":brier,"baseBrier":base}
    best=max(choices,key=lambda z:z[0]); threshold=best[1]
    ix=np.where(pt>=threshold)[0]
    atrs=np.asarray([float(te[i].get("atr",0)) for i in ix])
    gross=np.asarray([atrs[j]*lot_oz if te[i]["discovery"][side]=="TARGET" else (-.75*atrs[j]*lot_oz if te[i]["discovery"][side]=="STOP" else 0.0) for j,i in enumerate(ix)])
    net=gross-cost; stress_net=gross-stress_cost
    gain=net[net>0].sum(); loss=np.abs(net[net<0]).sum()
    pf=float(gain/loss) if loss>0 else 0
    stress_gain=stress_net[stress_net>0].sum(); stress_loss=np.abs(stress_net[stress_net<0]).sum()
    stress_pf=float(stress_gain/stress_loss) if stress_loss>0 else 0
    passed=(brier<=base*.98 and len(ix)>=30 and net.sum()>0 and stress_net.sum()>0 and pf>=1.25 and stress_pf>1.0)
    artifact={"weights":model.coef_[0].tolist(),"bias":float(model.intercept_[0]),"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"plattA":float(calibrator.coef_[0][0]),"plattB":float(calibrator.intercept_[0])}
    return {"status":"VALIDATED" if passed else "HOLDOUT_GATE_FAILED","validated":passed,"samples":len(rows),"train":len(tr),"calibration":len(cal),"test":len(te),"brier":brier,"baseBrier":base,"threshold":threshold,"testTrades":len(ix),"testNetUsd":float(net.sum()),"testPF":pf,"stressNetUsd":float(stress_net.sum()),"stressPF":stress_pf,"artifact":artifact if passed else None}

def fit_scenarios(rows,horizon):
    rows=[r for r in rows if r.get("scenarios",{}).get(str(horizon)) in CLASSES]
    if len(rows)<900:return {"status":"INSUFFICIENT_SAMPLES","samples":len(rows)}
    tr,cal,te=chronological(rows)
    if min(len(tr),len(cal),len(te))<100:return {"status":"INSUFFICIENT_TIME_SPLIT","samples":len(rows)}
    X=np.asarray([r["features"] for r in tr],float); y=np.asarray([CLASSES.index(r["scenarios"][str(horizon)]) for r in tr])
    Xt=np.asarray([r["features"] for r in te],float); yt=np.asarray([CLASSES.index(r["scenarios"][str(horizon)]) for r in te])
    if set(y)!=set(range(3)) or set(yt)!=set(range(3)):return {"status":"MISSING_CLASS_IN_SPLIT","samples":len(rows)}
    scaler=StandardScaler().fit(X); model=LogisticRegression(C=.2,max_iter=1500,multi_class="multinomial",random_state=17)
    model.fit(scaler.transform(X),y); p=model.predict_proba(scaler.transform(Xt))
    one=np.eye(3)[yt]; brier=float(np.mean(np.sum((p-one)**2,axis=1)))
    basep=np.bincount(yt,minlength=3)/len(yt); base=float(np.mean(np.sum((basep-one)**2,axis=1)))
    passed=brier<=base*.98
    models={}
    if passed:
        for i,k in enumerate(model.classes_):
            models[CLASSES[int(k)]]={"weights":model.coef_[i].tolist(),"bias":float(model.intercept_[i]),"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist()}
    return {"status":"VALIDATED" if passed else "HOLDOUT_GATE_FAILED","validated":passed,"samples":len(rows),"train":len(tr),"test":len(te),"brier":brier,"baseBrier":base,"models":models if passed else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",default="/data/gold-ai-paper-state.json")
    ap.add_argument("--out-dir",default="/data")
    ap.add_argument("--cost",type=float,default=.37)
    ap.add_argument("--stress-cost",type=float,default=.74)
    ap.add_argument("--lot-oz",type=float,default=1.0)
    args=ap.parse_args()
    with open(args.state,encoding="utf-8") as f: state=json.load(f)
    rows=state.get("aiLearningSamples",[])
    result={"samples":len(rows),"discovery":{},"scenarios":{}}
    discovery_models={}
    for side in ("BUY","SELL"):
        r=fit_binary(rows,side,args.cost,args.stress_cost,args.lot_oz);result["discovery"][side]={k:v for k,v in r.items() if k!="artifact"}
        if r.get("validated"):discovery_models[side]=r["artifact"]
    if len(discovery_models)==2:
        artifact={"kind":"gold-opportunity-linear-v1","modelVersion":"gold-opportunity-1.0","validated":True,"sides":discovery_models,"thresholds":{s:fit_binary(rows,s,args.cost,args.stress_cost,args.lot_oz)["threshold"] for s in ("BUY","SELL")},"stopAtr":0.75,"rewardRisk":1.3333333333,"label":"first-touch target +1 ATR before stop -0.75 ATR within 12 M5 bars"}
        path=os.path.join(args.out_dir,"gold-ai-opportunity-model.json");tmp=path+".tmp"
        with open(tmp,"w",encoding="utf-8") as f:json.dump(artifact,f,separators=(",",":"))
        os.replace(tmp,path);result["opportunityArtifact"]=path
    else:result["opportunityArtifact"]="NOT_WRITTEN_VALIDATION_FAILED"
    horizons={}
    for h in (3,6,12):
        r=fit_scenarios(rows,h);result["scenarios"][str(h)]={k:v for k,v in r.items() if k!="models"}
        if r.get("validated"):horizons[str(h)]=r["models"]
    if len(horizons)==3:
        artifact={"kind":"gold-scenario-softmax-v1","modelVersion":"gold-scenario-1.0","validated":True,"horizons":horizons,"label":"future close return in ATR units: UP > +0.35, DOWN < -0.35, otherwise RANGE"}
        path=os.path.join(args.out_dir,"gold-ai-scenario-model.json");tmp=path+".tmp"
        with open(tmp,"w",encoding="utf-8") as f:json.dump(artifact,f,separators=(",",":"))
        os.replace(tmp,path);result["scenarioArtifact"]=path
    else:result["scenarioArtifact"]="NOT_WRITTEN_ALL_HORIZONS_NOT_VALIDATED"
    print(json.dumps(result,indent=2))
    return 0 if result["opportunityArtifact"]!="NOT_WRITTEN_VALIDATION_FAILED" or result["scenarioArtifact"]!="NOT_WRITTEN_ALL_HORIZONS_NOT_VALIDATED" else 3
if __name__=="__main__":sys.exit(main())
