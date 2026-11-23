#!/usr/bin/env python3
"""Train a walk-forward XGBoost meta-labeler from PAPER candidate outcomes.
Input: /data/gold-ai-paper-state.json (or --state).
Only completed TARGET/STOP/TIMEOUT shadow outcomes are used.
"""
import argparse, json, math, os, sys
from datetime import datetime, timezone
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

CLASSES = ("TARGET", "STOP", "TIMEOUT")
FEATURES = ["qualityScore","stopAtr","rr","regimeConfidence","atrRatio","efficiency","bodyAtr","emaAlignment","slope","spreadAtr","isBuy","expertSweep","expertBreakout","expertRetest","expertContinuation","expertReversal","expertPullback","expertRange","expertMomentum","expertVision","regimeTrend","regimeRange","regimeCompression","regimeExpansion","regimeImpulse"]

def load_rows(path):
    with open(path, "r", encoding="utf-8") as f: state=json.load(f)
    candidates={}
    for row in state.get("candidateHistory", []):
        if row.get("type") is None and row.get("id") and isinstance(row.get("metaFeatures"), list):
            candidates[row["id"]]=row
    outcomes={}
    for row in state.get("candidateHistory", []):
        if row.get("type")=="SHADOW_OUTCOME" and row.get("id") and row.get("outcome") in CLASSES:
            outcomes[row["id"]]=row["outcome"]
    rows=[]
    for cid,label in outcomes.items():
        c=candidates.get(cid)
        if c:
            rows.append((c["at"],np.asarray(c["metaFeatures"],dtype=float),label,cid))
    rows.sort(key=lambda x:x[0])
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",default="/data/gold-ai-paper-state.json")
    ap.add_argument("--out",default="/data/gold-ai-xgb-model.json")
    args=ap.parse_args()
    rows=load_rows(args.state)
    counts={k:sum(r[2]==k for r in rows) for k in CLASSES}
    report={"status":"INSUFFICIENT_LABELS","samples":len(rows),"classCounts":counts,"requiredTotal":240,"requiredPerClass":30}
    if len(rows)<240 or min(counts.values())<30:
        print(json.dumps(report)); return 2
    X=np.vstack([r[1] for r in rows]); y=np.asarray([r[2] for r in rows])
    train_end=int(len(rows)*0.70); calibration_end=int(len(rows)*0.85)
    Xtr,Xcal,Xv=X[:train_end],X[train_end:calibration_end],X[calibration_end:]
    ytr,ycal,yv=y[:train_end],y[train_end:calibration_end],y[calibration_end:]
    if (min(sum(ytr==k) for k in CLASSES)<20 or
        min(sum(ycal==k) for k in CLASSES)<8 or
        min(sum(yv==k) for k in CLASSES)<8):
        report.update(status="INSUFFICIENT_WALK_FORWARD_CLASS_COVERAGE",train=len(ytr),calibration=len(ycal),validation=len(yv))
        print(json.dumps(report)); return 2
    models={}; metrics={}
    for cls in CLASSES:
        yt=(ytr==cls).astype(int); ycal=(ycal==cls).astype(int); yval=(yv==cls).astype(int)
        model=XGBClassifier(n_estimators=120,max_depth=3,learning_rate=0.04,
            subsample=0.8,colsample_bytree=0.9,min_child_weight=8,reg_lambda=4,
            objective="binary:logistic",eval_metric="logloss",base_score=0.5,
            n_jobs=1,random_state=17)
        model.fit(Xtr,yt,verbose=False)
        cal_margins=model.get_booster().predict(__import__("xgboost").DMatrix(Xcal),output_margin=True)
        # Fit calibration on the middle chronological slice; evaluate on a later untouched slice.
        calibrator=LogisticRegression(C=1e6,solver="lbfgs")
        calibrator.fit(cal_margins.reshape(-1,1),ycal)
        margins=model.get_booster().predict(__import__("xgboost").DMatrix(Xv),output_margin=True)
        calibrated=calibrator.predict_proba(margins.reshape(-1,1))[:,1]
        brier=float(brier_score_loss(yval,calibrated))
        baseline=float(brier_score_loss(yval,np.full(len(yval),yval.mean())))
        metrics[cls]={"brier":round(brier,6),"baseRateBrier":round(baseline,6),
            "logLoss":round(float(log_loss(yval,np.clip(calibrated,1e-6,1-1e-6))),6),
            "accuracyAt50":round(float(accuracy_score(yval,calibrated>=0.5)),4),
            "validationCount":len(yval),"validationPositives":int(yval.sum())}
        if brier>baseline*0.98:
            report.update(status="VALIDATION_NOT_BEATING_BASELINE",metrics=metrics)
            print(json.dumps(report)); return 3
        dump=[json.loads(t) for t in model.get_booster().get_dump(dump_format="json")]
        models[cls]={"baseMargin":0.0,"plattA":float(calibrator.coef_[0][0]),
            "plattB":float(calibrator.intercept_[0]),"trees":dump}
    artifact={"kind":"xgboost-one-vs-rest-v1","modelVersion":"gold-meta-xgb-1.0",
        "trainedAt":datetime.now(timezone.utc).isoformat(),"featureNames":FEATURES,
        "featureCount":int(X.shape[1]),"trainSamples":len(ytr),"calibrationSamples":len(ycal),"validationSamples":len(yv),
        "classCounts":counts,"walkForwardMetrics":metrics,"models":models,
        "labelDefinition":"first barrier reached on M5 bars: TARGET, STOP, or TIMEOUT; labels use shadow outcomes after base cost"}
    tmp=args.out+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(artifact,f,separators=(",",":"))
    os.replace(tmp,args.out)
    report.update(status="TRAINED_AND_VALIDATED",modelVersion=artifact["modelVersion"],metrics=metrics,output=args.out)
    print(json.dumps(report)); return 0

if __name__=="__main__": sys.exit(main())
