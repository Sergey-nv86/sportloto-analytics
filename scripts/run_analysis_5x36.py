#!/usr/bin/env python3
import csv, json, math, os, random, re, ssl, statistics, urllib.request, time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/results_5x36.csv"; REPORT=ROOT/"reports/latest_5x36.md"; SIGNALS=ROOT/"reports/signals_5x36.md"
DATA.parent.mkdir(parents=True,exist_ok=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
URL="https://russkoe-loto.com/sportloto5x36/arhiv-rezultatov/{:04d}/{:02d}"; UA="Mozilla/5.0 SportlotoResearch/2.1"
DEFAULT_STRATEGIES=["RANDOM","HOT","COLD","RECENT30","MOMENTUM","GAP","SELFLAG","CROSSLAG","PAIRS","LEARNED","LEARNED_EWMA","ENSEMBLE"]
STRATEGIES=[x for x in os.environ.get("STRATEGIES","RANDOM,HOT,COLD,RECENT30,MOMENTUM,GAP,SELFLAG,CROSSLAG,PAIRS,LEARNED,ENSEMBLE").split(",") if x]

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
    links=list(re.finditer(r'href="/sportloto5x36/rezultaty/(\d+)"[^>]*>.*?</a>',html,re.S|re.I))
    for i,m in enumerate(links):
        chunk=html[m.start():links[i+1].start() if i+1<len(links) else len(html)]
        draw=int(m.group(1))
        dt=re.search(r'<time[^>]+datetime="([^"]+)"',chunk,re.I)
        b=re.findall(r"<li[^>]*class=[\"']ball[\"'][^>]*>\s*(\d+)\s*</li>",chunk,re.I)
        if len(b)<6:
            b=re.findall(r"class=[\"'][^\"']*ball[^\"']*[\"'][^>]*>\\s*(\\d+)\\s*<",chunk,re.I)
        if len(b)>=6:
            main=tuple(sorted(map(int,b[:5])))
            bonus=int(b[5]); ns=main+(bonus,)
            if len(set(main))==5 and all(1<=n<=36 for n in main) and 1<=bonus<=4:
                out.append((draw,dt.group(1) if dt else "",ns))
    return out

def load():
    now=datetime.utcnow(); y,m=now.year,now.month-1
    if m==0:y,m=y-1,12
    d={}; errors=[]; scanned=0
    for yy,mm in months((2017,1),(y,m)):
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

def hit(a,b):return len(set(a[:5])&set(b[:5]))

def init_state(rows):
    cnt=Counter(); r7=Counter(); r15=Counter(); r30=Counter(); r60=Counter(); r100=Counter(); pos={n:-1 for n in range(1,37)}
    tr={n:[0,0] for n in range(1,37)}; pair=Counter()
    for i,(*_,ns) in enumerate(rows):
        for n in ns[:5]:
            cnt[n]+=1; pos[n]=i
            if i>=len(rows)-30:r30[n]+=1
            if i>=len(rows)-100:r100[n]+=1
        if i:
            A=set(rows[i-1][2]); B=set(ns[:5])
            for n in A:
                tr[n][0]+=1; tr[n][1]+=n in B
            for x in A:
                for y in B:
                    if x!=y:pair[x,y]+=1
    return {
        "cnt":cnt,"r7":r7,"r15":r15,"r30":r30,"r60":r60,"r100":r100,"pos":pos,"tr":tr,"pair":pair,
        "history_len":len(rows),"last":set(rows[-1][2][:5]) if rows else set(),
    }

def advance_state(st, row):
    i=st["history_len"]; ns=row[2]
    for w in (7,15,30,60,100):
        if i>=w:
            old=st[f"window{w}"]
            for n in old:
                st[f"r{w}"][n]-=1
    A=st["last"]; B=set(ns[:5])
    if A:
        for n in A:
            st["tr"][n][0]+=1; st["tr"][n][1]+=n in B
        for x in A:
            for y in B:
                if x!=y:st["pair"][x,y]+=1
    for n in ns[:5]:
        st["cnt"][n]+=1; st["pos"][n]=i; st["r30"][n]+=1; st["r100"][n]+=1
    for w in (7,15,30,60,100):
        st[f"window{w}"]=B if i < w else st[f"window{w}_queue"].pop(0)
        st[f"window{w}_queue"].append(B)
    st["last"]=B; st["history_len"]=i+1

def sigmoid(x):
    if x >= 35:return 1.0
    if x <= -35:return 0.0
    return 1.0/(1.0+math.exp(-x))

def feature_vector(st,n):
    h=max(1,st["history_len"])
    cnt=st["cnt"]; r30=st["r30"]; r100=st["r100"]; pos=st["pos"]; tr=st["tr"]
    gap=min(h-1,h-max(0,pos[n]))/h if pos[n]>=0 else 1.0
    repeat=1.0 if n in st["last"] else 0.0
    selfrate=tr[n][1]/tr[n][0] if tr[n][0] else 5/36
    return [1.0,cnt[n]/h,r30[n]/30.0,r100[n]/100.0,gap,repeat,selfrate,(r30[n]/30.0-r100[n]/100.0)]

def feature_vector_ewma(st,n):
    h=max(1,st["history_len"]); pos=st["pos"]
    gap=min(h-1,h-max(0,pos[n]))/h if pos[n]>=0 else 1.0
    return [1.0,st["cnt"][n]/h,st["r7"][n]/7.0,st["r15"][n]/15.0,st["r30"][n]/30.0,
            st["r60"][n]/60.0,st["r100"][n]/100.0,gap,1.0 if n in st["last"] else 0.0]

def learn_update(st,row,lr=0.08,weights_key="learned_w",feature_fn=feature_vector):
    w=st[weights_key]; ns=set(row[2][:5])
    rate=lr/(1.0+0.00002*st["learned_steps"])
    for n in range(1,37):
        x=feature_fn(st,n); y=1.0 if n in ns else 0.0
        p=sigmoid(sum(a*b for a,b in zip(w,x)))
        err=p-y
        for j,v in enumerate(x): w[j]-=rate*err*v
    st["learned_steps"]+=1

def train_initial_model(rows,weights_key="learned_w",feature_fn=feature_vector):
    if len(rows)<2:return [0.0]*len(feature_fn(init_state(rows[:1]),1))
    st=init_state(rows[:1])
    for w in (7,15,30,60,100):
        st[f"window{w}_queue"]=[set(rows[0][2][:5])]
        st[f"window{w}"]=set(rows[0][2][:5])
    st[weights_key]=[0.0]*len(feature_fn(st,1)); st["learned_steps"]=0
    for row in rows[1:]:
        learn_update(st,row,weights_key=weights_key,feature_fn=feature_fn)
        advance_state(st,row)
    return st[weights_key]

def predict_state(st,wanted=None):
    wanted=set(wanted or STRATEGIES)
    cnt=st["cnt"]; r30=st["r30"]; r100=st["r100"]; pos=st["pos"]; tr=st["tr"]; pair=st["pair"]; last=st["last"]
    base=range(1,37)
    scores={}
    if "HOT" in wanted:scores["HOT"]={n:cnt[n] for n in base}
    if "COLD" in wanted:scores["COLD"]={n:-cnt[n] for n in base}
    if "RECENT30" in wanted:scores["RECENT30"]={n:r30[n] for n in base}
    if "MOMENTUM" in wanted:scores["MOMENTUM"]={n:r30[n]/30-r100[n]/100 for n in base}
    if "GAP" in wanted:scores["GAP"]={n:st["history_len"]-1-pos[n] for n in base}
    if "SELFLAG" in wanted:scores["SELFLAG"]={n:(tr[n][1]/tr[n][0] if tr[n][0] else 5/36) for n in base}
    if "CROSSLAG" in wanted:scores["CROSSLAG"]={n:statistics.mean([pair[x,n]/max(1,tr[n][0]) for x in last]) if last else 5/36 for n in base}
    if "PAIRS" in wanted:scores["PAIRS"]={n:sum(pair[x,n] for x in last) for n in base}
    if "LEARNED" in wanted:scores["LEARNED"]={n:sum(a*b for a,b in zip(st["learned_w"],feature_vector(st,n))) for n in base}
    if "LEARNED_EWMA" in wanted:scores["LEARNED_EWMA"]={n:sum(a*b for a,b in zip(st["learned_ewma_w"],feature_vector_ewma(st,n))) for n in base}
    return {s:sorted(sc,key=lambda n:(-sc[n],n))[:5] for s,sc in scores.items()}

def pair_counts(rows,triples=False):
    c=Counter()
    for *_,ns in rows:
        for i in range(5):
            for j in range(i+1,5):
                if triples:
                    for k in range(j+1,5):c[ns[i],ns[j],ns[k]]+=1
                else:c[ns[i],ns[j]]+=1
    return c

def lag(rows,L=20):
    return {k:statistics.mean(hit(rows[i-k][2],rows[i][2]) for i in range(k,len(rows))) for k in range(1,L+1)}

def self_lag(rows):
    den=Counter();num=Counter()
    for i in range(1,len(rows)):
        A=set(rows[i-1][2][:5]);B=set(rows[i][2][:5])
        for n in A:den[n]+=1;num[n]+=n in B
    return sorted(((abs(num[n]/den[n]-5/36),n,num[n]/den[n],den[n]) for n in den),reverse=True)

def mc(preds,actuals,sims=1200):
    rng=random.Random(20261001); names=list(preds)
    masks={n:1<<(n-1) for n in range(1,37)}
    pm={m:[sum(masks[n] for n in preds[m][i]) for i in range(len(actuals))] for m in names}
    am=[sum(masks[n] for n in actuals[i]) for i in range(len(actuals))]
    obs={m:sum((pm[m][i]&am[i]).bit_count() for i in range(len(actuals))) for m in names}
    best=max(obs.values());ge=0
    nums=range(1,37)
    for _ in range(sims):
        t={m:0 for m in names}
        for i in range(len(actuals)):
            d=0
            # Fast uniform 5/36 ticket generation: one sample per draw,
            # while keeping the same exact null model as the original test.
            for n in rng.sample(nums,5): d |= masks[n]
            for m in names: t[m] += (pm[m][i] & d).bit_count()
        ge += max(t.values()) >= best
    return obs,(ge+1)/(sims+1)

def main():
    if os.environ.get("SKIP_LOAD"):
        rows=[]
        with DATA.open(encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                rows.append((int(r["draw"]),r["datetime"],tuple(sorted(int(r[f"n{i}"]) for i in range(1,6))) + (int(r["n6"]),)))
        rows=sorted(rows,key=lambda x:(x[1],x[0])); errors=[]; scanned=0
    else:
        rows,errors,scanned=load()
    if os.environ.get("FETCH_ONLY"):
        if len(rows) < 1000:
            raise SystemExit(f"Not enough data after archive fetch: {len(rows)}")
        print(f"FETCH_ONLY_OK draws={len(rows)}")
        return
    if len(rows)<1000:raise SystemExit(f"Not enough data: {len(rows)}")
    rows=sorted({r[0]:r for r in rows}.values(),key=lambda x:(x[1],x[0]));N=len(rows);H=min(2500,N//5);M=min(250,N-H-1);split=N-H-M
    preds={s:[] for s in STRATEGIES};actual=[]
    rng=random.Random(20261001)
    st=init_state(rows[:M])
    if "LEARNED" in STRATEGIES: st["learned_w"]=train_initial_model(rows[:M],"learned_w",feature_vector)
    if "LEARNED_EWMA" in STRATEGIES: st["learned_ewma_w"]=train_initial_model(rows[:M],"learned_ewma_w",feature_vector_ewma)
    st["learned_steps"]=max(0,M-1)
    for w in (7,15,30,60,100):
        st[f"window{w}_queue"]=[set(r[2][:5]) for r in rows[max(0,M-w):M]]
        st[f"window{w}"]=st[f"window{w}_queue"][-1] if st[f"window{w}_queue"] else set()
    for i in range(M,N):
        actual.append(rows[i][2])
        all_preds=predict_state(st)
        for s in STRATEGIES:preds[s].append(sorted(rng.sample(range(1,37),5)) if s=="RANDOM" else all_preds[s])
        if "LEARNED" in STRATEGIES: learn_update(st, rows[i],weights_key="learned_w",feature_fn=feature_vector)
        if "LEARNED_EWMA" in STRATEGIES: learn_update(st, rows[i],weights_key="learned_ewma_w",feature_fn=feature_vector_ewma)
        advance_state(st, rows[i])
    if os.environ.get("BATCH_OUT"):
        out=Path(os.environ["BATCH_OUT"]); out.parent.mkdir(parents=True,exist_ok=True)
        payload={"batch":os.environ.get("BATCH_NAME","batch"),"strategies":STRATEGIES,"draws":N,"holdout":H,"split":split,"dev":{s:statistics.mean(hit(preds[s][i],actual[i]) for i in range(split)) for s in STRATEGIES},"hold":{s:statistics.mean(hit(preds[s][split+i],actual[split+i]) for i in range(H)) for s in STRATEGIES},"hold_actual":actual[split:],"hold_preds":{s:preds[s][split:] for s in STRATEGIES}}
        out.write_text(json.dumps(payload),encoding="utf-8"); print(f"BATCH={payload['batch']} DRAWS={N} HOLDOUT={H}"); return
    dev_actual=actual[:split];hold_actual=actual[split:]
    dev={s:statistics.mean(hit(preds[s][i],dev_actual[i]) for i in range(split)) for s in STRATEGIES}
    hold={s:statistics.mean(hit(preds[s][split+i],hold_actual[i]) for i in range(H)) for s in STRATEGIES}
    hp={s:preds[s][split:] for s in STRATEGIES};obs,p=mc(hp,hold_actual)
    freq=Counter(n for *_,ns in rows for n in ns[:5]); sums=[sum(ns[:5]) for *_,ns in rows]; overlaps=Counter(hit(rows[i-1][2],rows[i][2]) for i in range(1,N))
    pairs=pair_counts(rows); triples=pair_counts(rows,True); lags=lag(rows); sl=self_lag(rows)
    lines=[f"# Sportloto 5/36 — full-history analysis","",f"Generated: {datetime.utcnow().isoformat(timespec='seconds')} UTC",f"Draws: **{N}** | range: **{rows[0][0]} → {rows[-1][0]}** | archive months scanned: **{scanned}**","", "## Executive result","",f"Random expectation: **0.694 hits** per 5-number ticket.",f"Final holdout: **{H} draws**. Monte Carlo max-over-{len(STRATEGIES)}-strategies p-value: **{p:.4f}** (1,200 simulations).","", "## Walk-forward / holdout","", "| Strategy | Development | Holdout | Δ vs 0.694 |","|---|---:|---:|---:|"]
    for s in sorted(STRATEGIES,key=lambda x:-hold[x]):lines.append(f"| {s} | {dev[s]:.3f} | {hold[s]:.3f} | {hold[s]-(5/36):+.3f} |")
    lines += ["","## Holdout blocks (100 draws)","", "| Block | "+" | ".join(STRATEGIES)+" |","|---|"+"|".join(["---"]*len(STRATEGIES))+"|"]
    for a in range(0,H,100):
        b=min(a+100,H);lines.append("| "+f"{a+1}-{b}"+" | "+" | ".join(f"{statistics.mean(hit(hp[s][i],hold_actual[i]) for i in range(a,b)):.3f}" for s in STRATEGIES)+" |")
    lines += ["","## Frequency windows","", "| Window | Top 10 |","|---|---|"]
    for name,w in [("All",N),("5y",min(1825,N)),("3y",min(1095,N)),("1y",min(365,N)),("6m",min(183,N)),("3m",min(92,N))]:
        f=Counter(n for *_,ns in rows[-w:] for n in ns);lines.append("| "+name+" | "+", ".join(f"{n}:{f[n]}" for n in f.most_common(10))+" |")
    lines += ["","## Pairs / triples","",f"Expected count for one specific pair: **{N*10/630:.2f}**",f"Expected count for one specific triple: **{N*10/7140:.2f}**","", "**Top pairs:** "+", ".join(f"{k}:{v}" for k,v in pairs.most_common(20)),"","**Top triples:** "+", ".join(f"{k}:{v}" for k,v in triples.most_common(20)),"","## Sequential dependence","", "| Lag | Mean overlap | Δ vs 0.6944444444444444 |","|---:|---:|---:|"]
    for k,v in lags.items():lines.append(f"| {k} | {v:.4f} | {v-(5/36):+.4f} |")
    lines += ["","### Strongest self-lag deviations (unadjusted)","", "| Number | P(repeat next draw) | N |","|---:|---:|---:|"]
    for _,n,v,d in sl[:15]:lines.append(f"| {n} | {v:.4f} | {d} |")
    lines += ["","## Distribution","",f"- Sum mean: **{statistics.mean(sums):.3f}**; median: **{statistics.median(sums):.1f}**; SD: **{statistics.pstdev(sums):.3f}**; range: **{min(sums)}–{max(sums)}**.",f"- Previous-draw overlap: {', '.join(f'{k}:{v} ({v/(N-1):.2%})' for k,v in sorted(overlaps.items()))}","", "## Data integrity", "",f"- Unique draw IDs: **{len({r[0] for r in rows})}**",f"- Fetch warnings: **{len(errors)}**","", "## Interpretation","", "The LEARNED strategy is an online logistic model trained only on information available before each draw; its weights are updated after the observed draw. It is included as an experimental model, not as evidence that lottery outcomes are predictable. Historical frequencies, pairs, triples and lag extremes are descriptive and vulnerable to multiple testing. The decisive evidence is chronological out-of-sample performance. The holdout was not used to select models, and the Monte Carlo test accounts for searching across multiple strategies.","","This is statistical research, not a guarantee of future lottery outcomes."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    SIGNALS.write_text("# Candidate signals\n\n"+"\n".join(f"- {s}: development {dev[s]:.3f}; holdout {hold[s]:.3f}; Δ vs random {hold[s]-(5/36):+.3f}" for s in sorted(STRATEGIES,key=lambda x:-(hold[x]-dev[x])))+"\n\nRaw pair/triple leaders are included in latest.md; they are not predictive claims.\n",encoding="utf-8")
    print(f"REPORT={REPORT}\nSIGNALS={SIGNALS}\nDRAWS={N} WARNINGS={len(errors)}")

if __name__=="__main__":main()
