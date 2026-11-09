"""Additional paired experiments for the reconstructed 29-feature study."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from base import load, split, extractor, train_cnn, metrics, keras, XGBClassifier

MODELS=('transfer','scratch','random_frozen','xgboost')

def xgb(parts,k,seed):
    start=time.perf_counter()
    model=XGBClassifier(n_estimators=300,max_depth=8,learning_rate=.1,n_jobs=2,random_state=seed)
    model.fit(*parts[0])
    return metrics(parts[2][1],model.predict(parts[2][0]),k,time.perf_counter()-start)

def smaller(parts,fraction,seed):
    x,y=parts[0]; rng=np.random.default_rng(seed)
    rows=np.concatenate([rng.choice(np.flatnonzero(y==c),size=min(sum(y==c),max(5,round(sum(y==c)*fraction))),replace=False) for c in np.unique(y)])
    return [(x[rows],y[rows]),*parts[1:]]

def models(ext,parts,k,seed,epochs):
    keras.utils.set_random_seed(seed+100)
    fresh=extractor()
    scratch=extractor();scratch.set_weights(fresh.get_weights())
    return dict(transfer=train_cnn(ext,k,parts,epochs,True,seed+100),
                scratch=train_cnn(scratch,k,parts,epochs,False,seed+100),
                random_frozen=train_cnn(fresh,k,parts,epochs,True,seed+100),
                xgboost=xgb(parts,k,seed))

def open_split(target,classes,seed,held):
    # Split known records before scaling. Unknown records enter only the test set.
    k=[c for c in classes if c not in held]
    known=target[target.label.isin(k)]
    normal=known[known.label=='normal']; attack=known[known.label!='normal']
    known=pd.concat([normal.sample(n=min(len(normal),len(attack)),random_state=seed),attack])
    x=known.drop(columns='label').to_numpy(np.float32); y=pd.Categorical(known.label,categories=k).codes
    tr,tmp=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    va,te=train_test_split(tmp,test_size=2/3,stratify=y[tmp],random_state=seed)
    unknown=target[target.label.isin(held)].drop(columns='label').to_numpy(np.float32)
    x[:,:12]=np.log1p(x[:,:12]);unknown[:,:12]=np.log1p(unknown[:,:12])
    scaler=StandardScaler().fit(x[tr,:12])
    x[:,:12]=scaler.transform(x[:,:12]);unknown[:,:12]=scaler.transform(unknown[:,:12])
    return [(x[i],y[i]) for i in (tr,va,te)],unknown,k

def novelty(model,parts,unknown):
    def score(x):
        return 1-np.concatenate([model(x[i:i+256,:,None],training=False).numpy().max(1) for i in range(0,len(x),256)])
    val=score(parts[1][0]); test=score(parts[2][0]); unseen=score(unknown)
    y=np.r_[np.zeros(len(test)),np.ones(len(unseen))]; s=np.r_[test,unseen]
    out=dict(auroc=roc_auc_score(y,s),average_precision=average_precision_score(y,s),
             known_test=len(test),unknown_test=len(unseen),points={})
    for rate in (.01,.05,.10):
        threshold=float(np.quantile(val,1-rate,method='higher'))
        out['points'][str(rate)]=dict(threshold=threshold,known_test_fpr=float(np.mean(test>threshold)),
                                     unknown_recall=float(np.mean(unseen>threshold)))
    return out

def main():
    p=argparse.ArgumentParser()
    for n in ('ton','unsw','output'):p.add_argument(n,type=Path)
    p.add_argument('--mode',choices=['reverse','labels','open'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    p.add_argument('--epochs',type=int,default=12)
    a=p.parse_args()
    if a.mode=='reverse':
        s,sc,sa=load(a.unsw,'unsw',['dos','exploits','fuzzers','generic','normal','reconnaissance'])
        t,tc,ta=load(a.ton,'ton',['backdoor','ddos','dos','injection','normal','password','ransomware','scanning','xss'])
    else:
        s,sc,sa=load(a.ton,'ton'); t,tc,ta=load(a.unsw,'unsw')
    out=dict(mode=a.mode,epochs=a.epochs,source_classes=sc,target_classes=tc,source=sa,target=ta,
             sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.ton,a.unsw)},runs=[])
    for seed in a.seeds:
        keras.backend.clear_session();keras.utils.set_random_seed(seed)
        sp=split(s,sc,seed,False); tp=split(t,tc,seed,True)
        ext=extractor()
        row=dict(seed=seed,source=train_cnn(ext,len(sc),sp,a.epochs,False,seed),comparisons={})
        if a.mode=='reverse':
            row['comparisons']['1.0']=models(ext,tp,len(tc),seed,a.epochs)
        elif a.mode=='labels':
            for frac in (.01,.05,.10):
                sample=smaller(tp,frac,seed)
                row['comparisons'][str(frac)]=models(ext,sample,len(tc),seed,a.epochs)
                row['comparisons'][str(frac)]['train_size']=len(sample[0][1])
        else:
            for held in [('generic','fuzzers'),('fuzzers',),('reconnaissance',),('dos',)]:
                parts,unknown,k=open_split(t,tc,seed,held)
                # DoS and Reconnaissance may have source counterparts. Exclude their
                # corresponding source classes when they are the held-out family.
                excluded={'dos':'dos','reconnaissance':'scanning'}
                pre=ext
                if held[0] in excluded:
                    labels=[c for c in sc if c!=excluded[held[0]]]
                    masked=s[s.label.isin(labels)]
                    keras.utils.set_random_seed(seed)
                    pre=extractor()
                    train_cnn(pre,len(labels),split(masked,labels,seed,False),a.epochs,False,seed)
                keras.utils.set_random_seed(seed+100)
                scratch=extractor()
                c={}
                for name,extract,frozen in [('transfer',pre,True),('scratch',scratch,False)]:
                    _,model=train_cnn(extract,len(k),parts,a.epochs,frozen,seed+100,True)
                    c[name]=novelty(model,parts,unknown)
                row['comparisons']['+'.join(held)]=c
        out['runs'].append(row)
        a.output.write_text(json.dumps(out,indent=2))
        print(a.mode,seed,flush=True)

if __name__=='__main__':main()
