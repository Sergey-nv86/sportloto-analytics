#!/usr/bin/env python3
import csv,json,random,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_analysis import hit
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/results.csv"; REPORT=ROOT/"reports/latest.md"; SIGNALS=ROOT/"reports/signals.md"
ALL=["RANDOM","HOT","COLD","RECENT30","MOMENTUM","GAP","SELFLAG","CROSSLAG","PAIRS","LEARNED","LEARNED_EWMA"]
def rows():
    with DATA.open(encoding="utf-8",newline="") as f:
        return sorted([(int(r["draw"]),r["datetime"],tuple(sorted(int(r[f"n{i}"]) for i in range(1,7)))) for r in csv.DictReader(f)],key=lambda x:(x[1],x[0]))
def mc(preds,actual,sims=1200):
    rng=random.Random(20261001)
    obs={s:sum(hit(preds[s][i],actual[i]) for i in range(len(actual))) for s in preds}; best=max(obs.values()); ge=0
    for _ in range(sims):
        t={s:0 for s in preds}
        for i in range(len(actual)):
            d=set(rng.sample(range(1,46),6))
            for s in preds:t[s]+=len(d&set(preds[s][i]))
        ge+=max(t.values())>=best
    return obs,(ge+1)/(sims+1)
def main():
    rs=rows(); files=sorted(ROOT.glob("reports/batch_*.json")); merged={}; dev={}; actual=None
    for f in files:
        x=json.loads(f.read_text()); merged.update(x["hold_preds"]); dev.update(x["dev"]); actual=x["hold_actual"]
    missing=[s for s in ALL if s not in merged]
    if missing: raise SystemExit("Missing batches: "+",".join(missing))
    ens=[]
    for i in range(len(actual)):
        rank={n:0 for n in range(1,46)}
        for s in ALL:
            for j,n in enumerate(merged[s][i]): rank[n]+=6-j
        ens.append(sorted(rank,key=lambda n:(-rank[n],n))[:6])
    merged["ENSEMBLE"]=ens
    dev["ENSEMBLE"]=statistics.mean(hit(ens[i],actual[i]) for i in range(len(ens)))
    H=len(actual); hold={s:statistics.mean(hit(merged[s][i],actual[i]) for i in range(H)) for s in merged}
    _,p=mc(merged,actual)
    lines=["# Sportloto 6/45 — full-history analysis","",f"Draws: **{len(rs)}** | range: **{rs[0][0]} → {rs[-1][0]}**",f"Holdout: **{H} draws** | independent model batches: **4**","","## Walk-forward / holdout","","| Strategy | Development | Holdout | Δ vs 0.8 |","|---|---:|---:|---:|"]
    for s in sorted(merged,key=lambda x:-hold[x]): lines.append(f"| {s} | {dev[s]:.3f} | {hold[s]:.3f} | {hold[s]-.8:+.3f} |")
    lines += ["","## Rolling holdout blocks","", "| Block | "+ " | ".join(ALL) + " |", "|---|" + "|".join(["---"]*len(ALL)) + "|"]
    for a in range(0,H,500):
        b=min(a+500,H)
        lines.append("| "+f"{a+1}-{b}"+" | "+" | ".join(f"{statistics.mean(hit(merged[s][i],actual[i]) for i in range(a,b)):.3f}" for s in ALL)+" |")
    lines += ["","## Multiple-model test","",f"- Random expectation: **0.800 hits** per ticket.",f"- Monte Carlo: **1,200 simulations** against the maximum across {len(merged)} strategies.",f"- Max-over-strategies p-value: **{p:.4f}**.","","## Interpretation","","The four calculation batches are independent at the workflow level and are combined only after chronological out-of-sample predictions are produced. Ensemble is constructed from the batch predictions rather than re-running all models in one long process.","","This is statistical research, not a guarantee of future lottery outcomes."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    SIGNALS.write_text("# Candidate signals\n\n"+"\n".join(f"- {s}: development {dev[s]:.3f}; holdout {hold[s]:.3f}; Δ vs random {hold[s]-.8:+.3f}" for s in sorted(merged,key=lambda x:-(hold[x]-.8)))+"\n",encoding="utf-8")
    print(f"AGGREGATED draws={len(rs)} holdout={H} p={p:.4f}")
if __name__=="__main__": main()
