"""Controlled positive control for test-record exposure at fixed training size."""
import argparse,json,time
from pathlib import Path
import numpy as np
from base import load,split,extractor,train_cnn,keras
from experiments import xgb

def main():
    p=argparse.ArgumentParser()
    for n in ('ton','unsw','output'):p.add_argument(n,type=Path)
    p.add_argument('--epochs',type=int,default=12)
    a=p.parse_args()
    s,sc,_=load(a.ton,'ton');t,tc,_=load(a.unsw,'unsw')
    out=dict(epochs=a.epochs,classes=tc,exposure_fraction=.20,runs=[])
    for seed in range(5):
        keras.backend.clear_session();keras.utils.set_random_seed(seed)
        ext=extractor()
        train_cnn(ext,len(sc),split(s,sc,seed,False),a.epochs,False,seed)
        tr,va,te=split(t,tc,seed,True)
        rng=np.random.default_rng(seed)
        donor=[];exposed=[]
        for label in range(len(tc)):
            pool=np.flatnonzero(te[1]==label)
            seen=rng.choice(pool,size=max(1,round(.2*len(pool))),replace=False)
            exposed.extend(seen)
            donor.extend(rng.choice(np.flatnonzero(tr[1]==label),size=len(seen),replace=True))
        assert np.array_equal(tr[1][donor],te[1][exposed])
        row=dict(seed=seed,exposed=len(exposed),test_size=len(te[1]),comparisons={})
        for tag,extra in [('training_duplicates',(tr[0][donor],tr[1][donor])),('test_exposure',(te[0][exposed],te[1][exposed]))]:
            parts=[(np.concatenate([tr[0],extra[0]]),np.r_[tr[1],extra[1]]),va,te]
            row['comparisons'][tag]=dict(transfer=train_cnn(ext,len(tc),parts,a.epochs,True,seed+100),xgboost=xgb(parts,len(tc),seed))
        out['runs'].append(row)
        a.output.write_text(json.dumps(out,indent=2))
        print('leakage',seed,flush=True)
if __name__=='__main__':main()
