"""Two-source comparison on the IDS-ML CICIDS2017 sample."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from base import load, split, extractor, train_cnn, keras, VOC, NUM
from experiments import smaller,xgb

def align(path):
    d=pd.read_csv(path)
    x=pd.DataFrame(dict(duration=d['Flow Duration']/1e6,
        src_bytes=d['Total Length of Fwd Packets']+d['Fwd Header Length'],
        dst_bytes=d['Total Length of Bwd Packets']+d['Bwd Header Length'],
        src_pkts=d['Total Fwd Packets'],dst_pkts=d['Total Backward Packets'],
        http_depth=0,http_response=0))[NUM].clip(lower=0)
    def ratio(a,b):
        return np.divide(a,b,out=np.zeros(len(a),dtype=float),where=np.asarray(b)>0)
    for name,a,b in [('mean_src',x.src_bytes,x.src_pkts),('mean_dst',x.dst_bytes,x.dst_pkts),('packet_rate',x.src_pkts+x.dst_pkts,x.duration),('src_load',8*x.src_bytes,x.duration),('dst_load',8*x.dst_bytes,x.duration)]:
        x[name]=ratio(a,b)
    flag=d[['FIN Flag Count','SYN Flag Count','RST Flag Count','PSH Flag Count','ACK Flag Count','URG Flag Count']].gt(0).any(axis=1)
    window=d[['Init_Win_bytes_forward','Init_Win_bytes_backward']].ge(0).any(axis=1)
    state=np.select([d['RST Flag Count']>0,(d['SYN Flag Count']>0)&(d['ACK Flag Count']==0),(d['ACK Flag Count']>0)|(d['FIN Flag Count']>0)],['reset','attempt','established'],default='other')
    raw={'proto':np.where(flag|window,'tcp','other'),'service':np.full(len(d),'other'),'state':state}
    for key, levels in VOC.items():
        for level in levels:x[f'{key}_{level}']=(raw[key]==level).astype('uint8')
    x=x.astype(np.float32)
    y=d.Label.str.strip().str.lower()
    x=x.assign(label=y).loc[y!='infiltration']
    n=len(x)
    x=x.drop_duplicates()
    k=len(x)
    x=x.loc[~x.duplicated(subset=list(x.columns[:-1]),keep=False)].reset_index(drop=True)
    return x,sorted(x.label.unique()),dict(retained_before_dedup=n,unique_vector_label_pairs=k,conflicting_rows_removed=k-len(x),clean_rows=len(x))

def main():
    p=argparse.ArgumentParser()
    for n in ('ton','unsw','cic','output'):p.add_argument(n,type=Path)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    a=p.parse_args()
    s,sc,_=load(a.ton,'ton')
    u,uc,_=load(a.unsw,'unsw',['dos','exploits','fuzzers','generic','normal','reconnaissance'])
    t,tc,stats=align(a.cic)
    out=dict(epochs=a.epochs,classes=tc,target=stats,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.ton,a.unsw,a.cic)},runs=[])
    for seed in a.seeds:
        keras.backend.clear_session();keras.utils.set_random_seed(seed)
        ton=extractor(); train_cnn(ton,len(sc),split(s,sc,seed,False),a.epochs,False,seed)
        keras.utils.set_random_seed(seed)
        unsw=extractor();train_cnn(unsw,len(uc),split(u,uc,seed,False),a.epochs,False,seed)
        tp=split(t,tc,seed,False)
        row=dict(seed=seed,comparisons={})
        for frac in (.01,1.0):
            parts=tp if frac==1 else smaller(tp,frac,seed)
            keras.utils.set_random_seed(seed+100)
            random=extractor();scratch=extractor();scratch.set_weights(random.get_weights())
            row['comparisons'][str(frac)]=dict(train_size=len(parts[0][1]),
                ton_transfer=train_cnn(ton,len(tc),parts,a.epochs,True,seed+100),
                unsw_transfer=train_cnn(unsw,len(tc),parts,a.epochs,True,seed+100),
                scratch=train_cnn(scratch,len(tc),parts,a.epochs,False,seed+100),
                random_frozen=train_cnn(random,len(tc),parts,a.epochs,True,seed+100),
                xgboost=xgb(parts,len(tc),seed))
        out['runs'].append(row)
        a.output.write_text(json.dumps(out,indent=2))
        print('cic',seed,flush=True)
if __name__=='__main__':main()
