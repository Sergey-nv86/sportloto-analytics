#!/usr/bin/env python3
import csv,json,random,statistics
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/results_5x36.csv"
REPORT=ROOT/"reports/latest_5x36.md"
SIGNALS=ROOT/"reports/signals_5x36.md"
NEXT=ROOT/"reports/next_bet_5x36.md"
PENDING=ROOT/"reports/pending_prediction_5x36.json"
HISTORY=ROOT/"reports/prediction_history_5x36.jsonl"

ALL=["RANDOM","HOT","COLD","RECENT30","MOMENTUM","GAP","SELFLAG","CROSSLAG","PAIRS","LEARNED","LEARNED_EWMA"]

def hit(a,b):
    return len(set(a[:5])&set(b[:5]))

def rows():
    with DATA.open(encoding="utf-8",newline="") as f:
        return sorted(
            [(int(r["draw"]),r["datetime"],tuple(sorted(int(r[f"n{i}"]) for i in range(1,6))) + (int(r["n6"]),))
             for r in csv.DictReader(f)],
            key=lambda x:(x[1],x[0])
        )

def mc(preds,actual,sims=1200):
    rng=random.Random(20261001)
    obs={s:sum(hit(preds[s][i],actual[i]) for i in range(len(actual))) for s in preds}
    best=max(obs.values()); ge=0
    for _ in range(sims):
        t={s:0 for s in preds}
        for i in range(len(actual)):
            d=set(rng.sample(range(1,37),5))
            for s in preds:t[s]+=len(d&set(preds[s][i][:5]))
        ge+=max(t.values())>=best
    return obs,(ge+1)/(sims+1)

def load_state(rows):
    # Reuse the project's battle-tested state/predictor implementation.
    import sys
    sys.path.insert(0,str(ROOT/"scripts"))
    import run_analysis_5x36 as ra
    M=len(rows)
    st=ra.init_state(rows[:1])
    for w in (7,15,30,60,100):
        q=[set(r[2][:5]) for r in rows[max(0,M-w):M]]
        st[f"window{w}_queue"]=q
        st[f"window{w}"]=q[-1] if q else set()
    if M>=2:
        st=ra.init_state(rows[:1])
        for w in (7,15,30,60,100):
            st[f"window{w}_queue"]=[set(rows[0][2][:5])]
            st[f"window{w}"]=set(rows[0][2][:5])
        st["learned_w"]=ra.train_initial_model(rows[:1],"learned_w",ra.feature_vector)
        st["learned_ewma_w"]=ra.train_initial_model(rows[:1],"learned_ewma_w",ra.feature_vector_ewma)
        st["learned_steps"]=0
        for row in rows[1:]:
            ra.learn_update(st,row,weights_key="learned_w",feature_fn=ra.feature_vector)
            ra.learn_update(st,row,weights_key="learned_ewma_w",feature_fn=ra.feature_vector_ewma)
            ra.advance_state(st,row)
    return ra,st

def evaluate_pending(rs):
    if not PENDING.exists():
        return []
    try:
        p=json.loads(PENDING.read_text(encoding="utf-8"))
    except Exception:
        return []
    source_draw=int(p["source_draw"])
    candidates=[r for r in rs if r[0] > source_draw]
    if not candidates:
        return []
    # Evaluate the first unseen draw after the prediction was issued.
    actual=candidates[0]
    rec=hit(p["ticket"],actual[2])
    result={
        "source_draw":source_draw,
        "target_draw":actual[0],
        "ticket":p["ticket"],
        "hits":rec,
        "datetime":actual[1],
        "strategy":p.get("strategy","dynamic_ensemble"),
    }
    with HISTORY.open("a",encoding="utf-8") as f:
        f.write(json.dumps(result,ensure_ascii=False)+"\n")
    PENDING.unlink()
    return [result]

def dynamic_next_ticket(rs):
    ra,st=load_state(rs)
    preds=ra.predict_state(st,set(ALL))
    # Champion/challenger weighting: only out-of-sample holdout performance
    # influences the next ticket; cap each model to avoid single-model lock-in.
    files=sorted(ROOT.glob("reports/batch_5x36_*.json"))
    scores={}
    for f in files:
        x=json.loads(f.read_text())
        for s,v in x.get("hold",{}).items():
            scores.setdefault(s,[]).append(float(v))
    hold={s:statistics.mean(v) for s,v in scores.items() if v}
    ranked=sorted((s for s in hold if s in preds),key=lambda s:hold[s],reverse=True)
    selected=[s for s in ranked if hold[s] >= 0.6944444444444444][:4]
    if not selected:
        selected=ranked[:3]
    if not selected:
        selected=["SELFLAG","HOT","PAIRS"]
    weights={s:max(0.05,hold.get(s,5/36)-0.67) for s in selected}
    # Recency stability bonus from the last 500 evaluated draws.
    for s in selected:
        bonus=0.0
        for f in files:
            x=json.loads(f.read_text())
            hp=x.get("hold_preds",{}).get(s,[])
            actual=x.get("hold_actual",[])
            if hp and actual:
                n=min(500,len(hp))
                bonus += statistics.mean(hit(hp[-n+i],actual[-n+i]) for i in range(n))
        weights[s] *= max(0.25, bonus/max(5/36,hold.get(s,5/36)))
    rank=Counter()
    for s in selected:
        ordered=preds[s]
        for pos,n in enumerate(ordered):
            rank[n] += weights[s]*(5-pos)
    ticket=sorted(rank,key=lambda n:(-rank[n],n))[:5]
    bonus_counts=Counter()
    for r in rs[-100:]:
        bonus_counts[int(r[2][5])] += 1
    bonus=max(range(1,5), key=lambda b:(bonus_counts[b],-b))
    return ticket,bonus,selected,{s:hold.get(s,5/36) for s in selected}

def main():
    rs=rows()
    files=sorted(ROOT.glob("reports/batch_*.json"))
    merged={}; dev={}; actual=None
    for f in files:
        x=json.loads(f.read_text())
        merged.update(x["hold_preds"]); dev.update(x["dev"]); actual=x["hold_actual"]
    missing=[s for s in ALL if s not in merged]
    if missing: raise SystemExit("Missing batches: "+",".join(missing))
    ens=[]
    for i in range(len(actual)):
        rank={n:0 for n in range(1,37)}
        for s in ALL:
            for j,n in enumerate(merged[s][i]): rank[n]+=5-j
        ens.append(sorted(rank,key=lambda n:(-rank[n],n))[:5])
    merged["ENSEMBLE"]=ens
    dev["ENSEMBLE"]=statistics.mean(hit(ens[i],actual[i]) for i in range(len(ens)))
    H=len(actual)
    hold={s:statistics.mean(hit(merged[s][i],actual[i]) for i in range(H)) for s in merged}
    _,p=mc(merged,actual)

    evaluated=evaluate_pending(rs)
    ticket,bonus,selected,selected_hold=dynamic_next_ticket(rs)
    source=rs[-1]
    PENDING.write_text(json.dumps({
        "source_draw":source[0],
        "source_datetime":source[1],
        "ticket":ticket,
        "bonus":bonus,
        "strategy":"dynamic_champion_ensemble",
        "models":selected,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    NEXT.write_text(
        "# Следующая ставка — Спортлото 5 из 36\n\n"
        f"Основание: после тиража **{source[0]}** ({source[1]}).\n\n"
        f"## Комбинация\n\n**{' · '.join(map(str,ticket))} + бонус **{bonus}****\n\n"
        "Это одна рекомендованная комбинация на следующий тираж. Она не является гарантией выигрыша.\n\n"
        "## Какие модели сформировали ставку\n\n"
        + "\n".join(f"- **{s}** — holdout {selected_hold[s]:.3f}" for s in selected)
        + "\n\n## Правило улучшения\n\n"
          "Вес моделей пересчитывается только по хронологическому out-of-sample holdout; "
          "после следующего тиража комбинация автоматически проверяется по фактическим 6 числам. "
          "Результат добавляется в историю, после чего веса и состав моделей могут измениться.\n",
        encoding="utf-8"
    )

    lines=["# Sportloto 5/36 — full-history analysis","",
           f"Draws: **{len(rs)}** | range: **{rs[0][0]} → {rs[-1][0]}**",
           f"Holdout: **{H} draws** | independent model batches: **4**","","## Walk-forward / holdout",
           "","| Strategy | Development | Holdout | Δ vs 0.694 |","|---|---:|---:|---:|"]
    for s in sorted(merged,key=lambda x:-hold[x]):
        lines.append(f"| {s} | {dev[s]:.3f} | {hold[s]:.3f} | {hold[s]-(5/36):+.3f} |")
    lines += ["","## Multiple-model test","",
              "- Random expectation: **0.694 hits** per 5-number ticket.",
              "- Monte Carlo: **1,200 simulations** against the maximum across 12 strategies.",
              f"- Max-over-strategies p-value: **{p:.4f}**."]
    if evaluated:
        lines += ["","## Previous recommendation check","",
                  f"- Ticket {evaluated[0]['ticket']} → draw **{evaluated[0]['target_draw']}**: **{evaluated[0]['hits']} / 5 hits**."]
    lines += ["","## Current next recommendation","",
              f"**{' · '.join(map(str,ticket))}** + бонус **{bonus}**",
              "",
              "Dynamic champion ensemble; models are reweighted after each newly observed draw.",
              "",
              "This is statistical research, not a guarantee of future lottery outcomes."]
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf-8")
    SIGNALS.write_text("# Candidate signals\n\n"+
        "\n".join(f"- {s}: development {dev[s]:.3f}; holdout {hold[s]:.3f}; Δ vs random {hold[s]-(5/36):+.3f}"
                    for s in sorted(merged,key=lambda x:-(hold[x]-.8)))+
        "\n\nThe next ticket is generated by a dynamic champion ensemble and evaluated after the next observed draw.\n",
        encoding="utf-8")
    print(f"AGGREGATED draws={len(rs)} holdout={H} p={p:.4f}")
    print(f"NEXT_TICKET={ticket} MODELS={selected}")
    if evaluated: print(f"EVALUATED_PREVIOUS={evaluated[0]['hits']}/5")

if __name__=="__main__":
    main()
