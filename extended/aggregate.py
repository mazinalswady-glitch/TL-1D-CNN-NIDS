"""Produce matched, descriptive summaries and the publication figure."""
import csv,json
from pathlib import Path
from statistics import mean,stdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
MODELS=('transfer','scratch','random_frozen','xgboost')

def read(name):
    p=ROOT/('..' if name=='primary' else '.')/('results.json' if name=='primary' else f'{name}.json')
    d=json.loads(p.read_text())
    assert [r['seed'] for r in d['runs']]==list(range(5)),name
    return d

def agg(records, key):
    v=[r[key] for r in records]
    return mean(v),stdev(v)

def run():
    data={n:read(n) for n in ('primary','reverse','labels','open','cic','leakage')}
    rows=[]
    for experiment,d in data.items():
        if experiment=='primary':
            settings={'1.0':d['runs']}
        else:
            settings={key:[r['comparisons'][key] for r in d['runs']] for key in d['runs'][0]['comparisons']}
        for setting,recs in settings.items():
            for model in recs[0]:
                if not isinstance(recs[0][model],dict) or ('macro_f1' not in recs[0][model] and 'auroc' not in recs[0][model]):
                    continue
                m=[r[model] for r in recs]
                keys=['macro_f1','accuracy','seconds'] if 'macro_f1' in m[0] else ['auroc','average_precision']
                if 'macro_f1' in m[0]:
                    assert all(0<=r['macro_f1']<=1 and len(r['confusion'])==len(r['confusion'][0]) for r in m)
                for key in keys:
                    avg,sd=agg(m,key);rows.append([experiment,setting,model,key,avg,sd])
                if experiment=='open':
                    prevalence=[r['unknown_test']/(r['known_test']+r['unknown_test']) for r in m]
                    rows.append([experiment,setting,model,'unknown_prevalence',mean(prevalence),stdev(prevalence)])
                    for fpr in ('0.01','0.05','0.1'):
                        for key in ('known_test_fpr','unknown_recall'):
                            avg,sd=agg([r['points'][fpr] for r in m],key)
                            rows.append([experiment,setting,model,key+'@'+fpr,avg,sd])
    with (ROOT/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['experiment','setting','model','metric','mean','sd']);w.writerows(rows)
    lookup={(e,s,m,k):(v,sd) for e,s,m,k,v,sd in rows}
    fig,axs=plt.subplots(1,2,figsize=(7.5,3.1),layout='constrained')
    colors=['#006AA6','#D55E00','#009E73','#6B508B']
    names=['Transfer','Scratch','Random frozen','XGBoost']
    for model,color,label in zip(MODELS,colors,names):
        vals=[lookup['labels',s,model,'macro_f1'] for s in ('0.01','0.05','0.1')]+[lookup['primary','1.0',model,'macro_f1']]
        axs[0].errorbar([1,5,10,100],[v[0] for v in vals],yerr=[v[1] for v in vals],marker='o',color=color,label=label,capsize=3)
    axs[0].set(xscale='log',xticks=[1,5,10,100],xticklabels=['1','5','10','100'],xlabel='Labelled target training data (%)',ylabel='Test macro-F1',title='A  Target label budget')
    axs[0].legend(fontsize=7)
    held=['generic+fuzzers','fuzzers','reconnaissance','dos']
    for i,(model,color,label) in enumerate(zip(MODELS[:2],colors,names)):
        vals=[lookup['open',s,model,'unknown_recall@0.05'] for s in held]
        axs[1].bar([j+(.17 if i else -.17) for j in range(4)],[v[0] for v in vals],width=.32,color=color,yerr=[v[1] for v in vals],capsize=3,label=label)
    axs[1].set(xticks=range(4),xticklabels=['Generic +\nFuzzers','Fuzzers','Recon.','DoS'],ylabel='Unknown recall',title='B  Nominal 5% validation FPR')
    axs[1].legend(fontsize=7)
    for ax in axs:ax.grid(axis='y',alpha=.22)
    fig.savefig(ROOT/'evaluation.png',dpi=300)
    fig.savefig(ROOT/'evaluation.pdf')
    return data,lookup

if __name__=='__main__':run()
