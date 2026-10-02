#!/usr/bin/env python3
import csv, math, random, re, ssl, statistics, urllib.request, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/results.csv"; REPORT=ROOT/"reports/latest.md"; SIGNALS=ROOT/"reports/signals.md"
DATA.parent.mkdir(parents=True,exist_ok=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
URL="http://russkoe-loto.com/sportloto6x45/arhiv-rezultatov/{:04d}/{:02d}"; UA="Mozilla/5.0 SportlotoResearch/2.0"
STRATEGIES=["RANDOM","HOT","COLD","RECENT30","MOMENTUM","GAP","SELFLAG","CROSSLAG","PAIRS","ENSEMBLE"]

def months(a,b):
    y,m=a
    while (y,m)<=b:
        yield y,m
        m+=1
        if m==13:y,m=y+1,1

def fetch(url):
    err=None
    for k in range(3):
        try:
            q=urllib.request.Request(url,headers={"User-Agent":UA})
            with urllib.request.urlopen(q,timeout=35,context=ssl._create_unverified_context()) as r:return r.read().decode("utf-8","ignore")
        except Exception as e:
            err=e; time.sleep(1+k)
    raise err

def parse(html):
    out=[]
    for c in html.split('<div class="row">')[1:]:
        dm=re.search(r'href="/sportloto6x45/rezultaty/(\d+)"[^>]*>\s*[\d\s]+\s*</a>',c)
        dt=re.search(r'<time[^>]+datetime="([^"]+)"',c)
        b=re.findall(r'<li[^>]*class="ball"[^>]*>\s*(\d+)\s*</li>',c)
        if dm and len(b)>=6:
            ns=tuple(sorted(map(int,b[:6])))
            if len(set(ns))==6 and all(1<=n<=45 for n in ns):out.append((int(dm.group(1)),dt.group(1) if dt else "",ns))
    return out

def load():
    now=datetime.utcnow(); y,m=now.year,now.month-1
    if m==0:y,m=y-1,12
    d={}; errors=[]; scanned=0
    for yy,mm in months((1990,1),(y,m)):
        scanned+=1
        try:
            rs=parse(fetch(URL.format(yy,mm)))
            for r in rs:d[r[0]]=r
            print(f"{yy}-{mm:02d}: {len(rs)}")
        except Exception as e:errors.append(f"{yy}-{mm:02d}: {e}")
    rows=sorted(d.values(),key=lambda x:(x[1],x[0]))
    with DATA.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["draw","datetime","n1","n2","n3","n4","n5","n6"])
        for x,dt,ns in rows:w.writerow([x,dt,*ns])
    return rows,errors,scanned

def hit(a,b):return len(set(a)&set(b))

def predict_all(h):
    cnt=Counter(); r30=Counter(); r100=Counter(); pos={n:-1 for n in range(1,46)}
    tr={n:[0,0] for n in range(1,46)}; pair=Counter()
    for i,(*_,ns) in enumerate(h):
        for n in ns:
            cnt[n]+=1; pos[n]=i
            if i>=len(h)-30:r30[n]+=1
            if i>=len(h)-100:r100[n]+=1
        if i:
            A=set(h[i-1][2]); B=set(ns)
            for n in A:
                tr[n][0]+=1; tr[n][1]+=n in B
            for x in A:
                for y in B:
                    if x!=y:pair[x,y]+=1
    last=set(h[-1][2]); base={n:0.0 for n in range(1,46)}
    scores={}
    scores["HOT"]={n:cnt[n] for n in base}
    scores["COLD"]={n:-cnt[n] for n in base}
    scores["RECENT30"]={n:r30[n] for n in base}
    scores["MOMENTUM"]={n:r30[n]/30-r100[n]/100 for n in base}
    scores["GAP"]={n:len(h)-1-pos[n] for n in base}
    scores["SELFLAG"]={n:(tr[n][1]/tr[n][0] if tr[n][0] else 6/45) for n in base}
    scores["CROSSLAG"]={n:statistics.mean([pair[x,n]/max(1,tr[n][0]) for x in last]) if last else 6/45 for n in base}
    scores["PAIRS"]={n:sum(pair[x,n] for x in last) for n in base}
    preds={}
    for s,sc in scores.items():
        preds[s]=sorted(sc,key=lambda n:(-sc[n],n))[:6]
    ens={n:sum((46-p.index(n)) if n in p else 0 for p in preds.values()) for n in base}
    preds["ENSEMBLE"]=sorted(ens,key=lambda n:(-ens[n],n))[:6]
    return preds

def pair_counts(rows,triples=False):
    c=Counter()
    for *_,ns in rows:
        for i in range(6):
            for j in range(i+1,6):
                if triples:
                    for k in range(j+1,6):c[ns[i],ns[j],ns[k]]+=1
                else:c[ns[i],ns[j]]+=1
    return c

def lag(rows,L=20):
    return {k:statistics.mean(hit(rows[i-k][2],rows[i][2]) for i in range(k,len(rows))) for k in range(1,L+1)}

def self_lag(rows):
    den=Counter();num=Counter()
    for i in range(1,len(rows)):
        A=set(rows[i-1][2]);B=set(rows[i][2])
        for n in A:den[n]+=1;num[n]+=n in B
    return sorted(((abs(num[n]/den[n]-6/45),n,num[n]/den[n],den[n]) for n in den),reverse=True)

def mc(preds,actuals,sims=5000):
    rng=random.Random(20261001); names=list(preds); obs={m:sum(hit(preds[m][i],actuals[i]) for i in range(len(actuals))) for m in names}; best=max(obs.values());ge=0
    for _ in range(sims):
        t={m:0 for m in names}
        for i in range(len(actuals)):
            d=set(rng.sample(range(1,46),6))
            for m in names:t[m]+=len(d&set(preds[m][i]))
        ge+=max(t.values())>=best
    return obs,(ge+1)/(sims+1)

def main():
    rows,errors,scanned=load()
    if len(rows)<1000:raise SystemExit(f"Not enough data: {len(rows)}")
    rows=sorted({r[0]:r for r in rows}.values(),key=lambda x:(x[1],x[0]));N=len(rows);H=min(500,N//5);M=min(250,N-H-1);split=N-H-M
    preds={s:[] for s in STRATEGIES};actual=[]
    rng=random.Random(20261001)
    for i in range(M,N):
        h=rows[:i];actual.append(rows[i][2])
        all_preds=predict_all(h)
        for s in STRATEGIES:preds[s].append(sorted(rng.sample(range(1,46),6)) if s=="RANDOM" else all_preds[s])
    dev_actual=actual[:split];hold_actual=actual[split:]
    dev={s:statistics.mean(hit(preds[s][i],dev_actual[i]) for i in range(split)) for s in STRATEGIES}
    hold={s:statistics.mean(hit(preds[s][split+i],hold_actual[i]) for i in range(H)) for s in STRATEGIES}
    hp={s:preds[s][split:] for s in STRATEGIES};obs,p=mc(hp,hold_actual)
    freq=Counter(n for *_,ns in rows for n in ns); sums=[sum(ns) for *_,ns in rows]; overlaps=Counter(hit(rows[i-1][2],rows[i][2]) for i in range(1,N))
    pairs=pair_counts(rows); triples=pair_counts(rows,True); lags=lag(rows); sl=self_lag(rows)
    lines=[f"# Sportloto 6/45 — full-history analysis","",f"Generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC",f"Draws: **{N}** | range: **{rows[0][0]} → {rows[-1][0]}** | archive months scanned: **{scanned}**","", "## Executive result","",f"Random expectation: **0.800 hits** per 6-number ticket.",f"Final holdout: **{H} draws**. Monte Carlo max-over-{len(STRATEGIES)}-strategies p-value: **{p:.4f}** (5,000 simulations).","", "## Walk-forward / holdout","", "| Strategy | Development | Holdout | Δ vs 0.8 |","|---|---:|---:|---:|"]
    for s in sorted(STRATEGIES,key=lambda x:-hold[x]):lines.append(f"| {s} | {dev[s]:.3f} | {hold[s]:.3f} | {hold[s]-.8:+.3f} |")
    lines += ["","## Holdout blocks (100 draws)","", "| Block | "+" | ".join(STRATEGIES)+" |","|---|"+"|".join(["---"]*len(STRATEGIES))+"|"]
    for a in range(0,H,100):
        b=min(a+100,H);lines.append("| "+f"{a+1}-{b}"+" | "+" | ".join(f"{statistics.mean(hit(hp[s][i],hold_actual[i]) for i in range(a,b)):.3f}" for s in STRATEGIES)+" |")
    lines += ["","## Frequency windows","", "| Window | Top 10 |","|---|---|"]
    for name,w in [("All",N),("5y",min(1825,N)),("3y",min(1095,N)),("1y",min(365,N)),("6m",min(183,N)),("3m",min(92,N))]:
        f=Counter(n for *_,ns in rows[-w:] for n in ns);lines.append("| "+name+" | "+", ".join(f"{n}:{f[n]}" for n in f.most_common(10))+" |")
    lines += ["","## Pairs / triples","",f"Expected count for one specific pair: **{N*15/990:.2f}**",f"Expected count for one specific triple: **{N*20/14190:.2f}**","", "**Top pairs:** "+", ".join(f"{k}:{v}" for k,v in pairs.most_common(20)),"","**Top triples:** "+", ".join(f"{k}:{v}" for k,v in triples.most_common(20)),"","## Sequential dependence","", "| Lag | Mean overlap | Δ vs 0.8 |","|---:|---:|---:|"]
    for k,v in lags.items():lines.append(f"| {k} | {v:.4f} | {v-.8:+.4f} |")
    lines += ["","### Strongest self-lag deviations (unadjusted)","", "| Number | P(repeat next draw) | N |","|---:|---:|---:|"]
    for _,n,v,d in sl[:15]:lines.append(f"| {n} | {v:.4f} | {d} |")
    lines += ["","## Distribution","",f"- Sum mean: **{statistics.mean(sums):.3f}**; median: **{statistics.median(sums):.1f}**; SD: **{statistics.pstdev(sums):.3f}**; range: **{min(sums)}–{max(sums)}**.",f"- Previous-draw overlap: {', '.join(f'{k}:{v} ({v/(N-1):.2%})' for k,v in sorted(overlaps.items()))}","", "## Data integrity", "",f"- Unique draw IDs: **{len({r[0] for r in rows})}**",f"- Fetch warnings: **{len(errors)}**","", "## Interpretation","", "Historical frequencies, pairs, triples and lag extremes are descriptive and vulnerable to multiple testing. The decisive evidence is chronological out-of-sample performance. The holdout was not used to select models, and the Monte Carlo test accounts for searching across multiple strategies.","","This is statistical research, not a guarantee of future lottery outcomes."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    SIGNALS.write_text("# Candidate signals\n\n"+"\n".join(f"- {s}: development {dev[s]:.3f}; holdout {hold[s]:.3f}; Δ vs random {hold[s]-.8:+.3f}" for s in sorted(STRATEGIES,key=lambda x:-(hold[x]-dev[x])))+"\n\nRaw pair/triple leaders are included in latest.md; they are not predictive claims.\n",encoding="utf-8")
    print(f"REPORT={REPORT}\nSIGNALS={SIGNALS}\nDRAWS={N} WARNINGS={len(errors)}")

if __name__=="__main__":main()
