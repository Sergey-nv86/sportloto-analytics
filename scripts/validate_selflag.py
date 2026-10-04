# Canonical validation: reuse run_analysis strategy implementations.
#!/usr/bin/env python3
import random, statistics
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_analysis import init_state, advance_state, predict_state, hit

DATA=ROOT/"data/results.csv"
OUT=ROOT/"reports/selflag_validation.md"

def load_csv():
    import csv
    with DATA.open(encoding="utf-8",newline="") as f:
        rows=[(int(r["draw"]),r["datetime"],tuple(sorted(int(r[f"n{i}"]) for i in range(1,7)))) for r in csv.DictReader(f)]
    return sorted(rows,key=lambda x:(x[1],x[0]))

def prepare(rows):
    st=init_state(rows)
    for w in (7,15,30,60,100):
        q=[set(r[2]) for r in rows[max(0,len(rows)-w):]]
        st[f"window{w}_queue"]=q
        st[f"window{w}"]=q[-1] if q else set()
    return st

def validate(rows,block=500,windows=6):
    N=len(rows); results=[]
    starts=[N-block*(windows-i) for i in range(windows)]
    names=["SELFLAG","RANDOM","HOT","PAIRS","CROSSLAG"]
    for w,start in enumerate(starts,1):
        st=prepare(rows[:start])
        rng=random.Random(20261004+w)
        scores={s:[] for s in names}
        for row in rows[start:start+block]:
            preds=predict_state(st,wanted=names)
            preds["RANDOM"]=sorted(rng.sample(range(1,46),6))
            for s in names:
                scores[s].append(hit(preds[s],row[2]))
            advance_state(st,row)
        results.append((w,start+1,start+block,scores))
    return results

def permutation(results,sims=5000):
    names=["RANDOM","HOT","PAIRS","CROSSLAG"]
    diffs={s:[] for s in names}
    for _,_,_,sc in results:
        for s in names:
            diffs[s].extend(a-b for a,b in zip(sc["SELFLAG"],sc[s]))
    obs={s:statistics.mean(v) for s,v in diffs.items()}
    observed=max(obs.values())
    rng=random.Random(20261004); ge=0
    for _ in range(sims):
        mx=max(statistics.mean((x if rng.getrandbits(1) else -x) for x in v) for v in diffs.values())
        ge += mx >= observed
    return obs,(ge+1)/(sims+1)

def main():
    rows=load_csv()
    if len(rows)<3500: raise SystemExit(f"Not enough data: {len(rows)}")
    res=validate(rows); obs,p=permutation(res)
    lines=["# SELFLAG — independent rolling validation","",f"Dataset: **{len(rows)} draws**",
           "Protocol: six chronological non-overlapping 500-draw test windows; each window uses only history strictly before that window.",
           "Canonical project strategies are used directly from run_analysis.py; no parameter tuning on test windows.","",
           "## Results","","| Window | RANDOM | HOT | PAIRS | CROSSLAG | SELFLAG |","|---|---:|---:|---:|---:|---:|"]
    for w,a,b,sc in res:
        m={s:statistics.mean(sc[s]) for s in sc}
        lines.append(f"| {w} ({a}-{b}) | {m['RANDOM']:.3f} | {m['HOT']:.3f} | {m['PAIRS']:.3f} | {m['CROSSLAG']:.3f} | {m['SELFLAG']:.3f} |")
    lines += ["","## Stability",""]
    for s,v in obs.items():
        wins=sum(statistics.mean(sc["SELFLAG"])>statistics.mean(sc[s]) for _,_,_,sc in res)
        lines.append(f"- SELFLAG vs {s}: mean paired advantage **{v:+.4f}** hits/draw; wins **{wins}/6** windows.")
    lines += ["","## Multiple comparisons","",
              "- Paired sign-flip max-statistic permutation: **5,000 simulations**.",
              f"- Max-statistic p-value across four comparisons: **{p:.4f}**.","",
              "This is validation, not model tuning. It is statistical research, not a guarantee of future lottery outcomes."]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"VALIDATION windows=6 p_max={p:.4f}")

if __name__=="__main__": main()
