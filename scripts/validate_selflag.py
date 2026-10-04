#!/usr/bin/env python3
import csv, random, statistics
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/results.csv"; OUT=ROOT/"reports/selflag_validation.md"

def load():
    with DATA.open(encoding="utf-8",newline="") as f:
        r=[(int(x["draw"]),x["datetime"],tuple(sorted(int(x[f"n{i}"]) for i in range(1,7)))) for x in csv.DictReader(f)]
    return sorted(r,key=lambda x:(x[1],x[0]))
def hit(a,b): return len(set(a)&set(b))
def init(rows):
    cnt={n:0 for n in range(1,46)}; pos={n:-1 for n in range(1,46)}
    tr={n:[0,0] for n in range(1,46)}; pair=defaultdict(int)
    for i,(_,_,ns) in enumerate(rows):
        B=set(ns)
        for n in B: cnt[n]+=1; pos[n]=i
        if i:
            A=set(rows[i-1][2])
            for n in A: tr[n][0]+=1; tr[n][1]+=n in B
            for x in A:
                for y in B:
                    if x!=y: pair[x,y]+=1
    return cnt,pos,tr,pair,set(rows[-1][2])
def predict(st,rng):
    cnt,pos,tr,pair,last=st
    return {
      "SELFLAG":sorted(range(1,46),key=lambda n:(-(tr[n][1]/tr[n][0] if tr[n][0] else 6/45),n))[:6],
      "HOT":sorted(range(1,46),key=lambda n:(-cnt[n],n))[:6],
      "PAIRS":sorted(range(1,46),key=lambda n:(-sum(pair[x,n] for x in last),n))[:6],
      "CROSSLAG":sorted(range(1,46),key=lambda n:(-(statistics.mean([pair[x,n]/max(1,tr[n][0]) for x in last]) if last else 6/45),n))[:6],
      "RANDOM":sorted(rng.sample(range(1,46),6))
    }
def advance(st,row):
    cnt,pos,tr,pair,last=st; B=set(row[2])
    for n in B: cnt[n]+=1; pos[n]+=1
    if last:
        for n in last: tr[n][0]+=1; tr[n][1]+=n in B
        for x in last:
            for y in B:
                if x!=y: pair[x,y]+=1
    last.clear(); last.update(B)
def validate(rows,block=500,windows=6):
    N=len(rows); out=[]
    starts=[N-block*(windows-i) for i in range(windows)]
    for w,start in enumerate(starts,1):
        st=init(rows[:start]); rng=random.Random(20261004+w)
        sc={s:[] for s in ["SELFLAG","RANDOM","HOT","PAIRS","CROSSLAG"]}
        for row in rows[start:start+block]:
            p=predict(st,rng)
            for s in sc: sc[s].append(hit(p[s],row[2]))
            advance(st,row)
        out.append((w,start+1,start+block,sc))
    return out
def permutation(results,sims=5000):
    names=["RANDOM","HOT","PAIRS","CROSSLAG"]; diffs={s:[] for s in names}
    for _,_,_,sc in results:
        for s in names: diffs[s].extend(a-b for a,b in zip(sc["SELFLAG"],sc[s]))
    obs={s:statistics.mean(v) for s,v in diffs.items()}; rng=random.Random(20261004); ge=0
    for _ in range(sims):
        mx=max(statistics.mean((x if rng.getrandbits(1) else -x) for x in v) for v in diffs.values())
        ge += mx >= max(obs.values())
    return obs,(ge+1)/(sims+1)
def main():
    rows=load()
    if len(rows)<3500: raise SystemExit(f"Not enough data: {len(rows)}")
    res=validate(rows); obs,p=permutation(res)
    lines=["# SELFLAG — independent rolling validation","",f"Dataset: **{len(rows)} draws**",
           "Protocol: six chronological 500-draw test windows; each window uses only history strictly before that window.",
           "No parameter tuning on test windows.","","## Results","","| Window | RANDOM | HOT | PAIRS | CROSSLAG | SELFLAG |","|---|---:|---:|---:|---:|---:|"]
    for w,a,b,sc in res:
        m={s:statistics.mean(sc[s]) for s in sc}
        lines.append(f"| {w} ({a}-{b}) | {m['RANDOM']:.3f} | {m['HOT']:.3f} | {m['PAIRS']:.3f} | {m['CROSSLAG']:.3f} | {m['SELFLAG']:.3f} |")
    lines += ["","## Stability",""]
    for s,v in obs.items():
        wins=sum(statistics.mean(sc["SELFLAG"])>statistics.mean(sc[s]) for _,_,_,sc in res)
        lines.append(f"- SELFLAG vs {s}: mean paired advantage **{v:+.4f}** hits/draw; wins **{wins}/6** windows.")
    lines += ["","## Multiple comparisons","", "- Paired sign-flip max-statistic permutation: **5,000 simulations**.",f"- Max-statistic p-value across four comparisons: **{p:.4f}**.","",
              "This is validation, not model tuning. It is statistical research, not a guarantee of future lottery outcomes."]
    OUT.write_text("\n".join(lines)+"\n",encoding="utf-8"); print(f"VALIDATION windows=6 p_max={p:.4f}")
if __name__=="__main__": main()
