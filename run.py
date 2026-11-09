"""Reconstruction of the ToN-IoT to UNSW-NB15 29-feature comparison."""
import argparse, hashlib, json, os, time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier

NUM=['duration','src_bytes','dst_bytes','src_pkts','dst_pkts','http_depth','http_response']
VOC={'proto':['tcp','udp','icmp','other'], 'service':['-','dns','http','ssl','ftp','ftp-data','smtp','ssh','other'],
     'state':['established','attempt','reset','other']}
STATES={'ton':{'SF':'established','S1':'established','S2':'established','S3':'established','S0':'attempt','REJ':'attempt','RSTO':'reset','RSTR':'reset','RSTOS0':'reset','RSTRH':'reset'},
        'unsw':{'FIN':'established','CON':'established','ACC':'established','REQ':'attempt','RST':'reset'}}

def load(path, domain):
    d=pd.read_csv(path,low_memory=False)
    source=domain=='ton'
    columns=['duration','src_ip_bytes','dst_ip_bytes','src_pkts','dst_pkts','http_trans_depth','http_response_body_len'] if source else ['dur','sbytes','dbytes','spkts','dpkts','trans_depth','response_body_len']
    x=d[columns].apply(pd.to_numeric,errors='coerce').fillna(0).clip(lower=0)
    x.columns=NUM
    def ratio(a,b):
        return np.divide(a,b,out=np.zeros(len(a),dtype=float),where=np.asarray(b)>0)
    for name,a,b in [('mean_src',x.src_bytes,x.src_pkts),('mean_dst',x.dst_bytes,x.dst_pkts),('packet_rate',x.src_pkts+x.dst_pkts,x.duration),('src_load',8*x.src_bytes,x.duration),('dst_load',8*x.dst_bytes,x.duration)]:
        x[name]=ratio(a,b)
    raw={'proto':d.proto.str.lower(),'service':d.service.str.lower(),'state':d['conn_state' if source else 'state'].map(STATES[domain]).fillna('other')}
    for key, levels in VOC.items():
        values=raw[key].where(raw[key].isin(levels),'other')
        for level in levels:x[f'{key}_{level}']=(values==level).astype('uint8')
    assert x.shape[1]==29
    x=x.astype(np.float32)
    y=d['type' if source else 'attack_cat'].str.strip().str.lower()
    classes=['ddos','dos','scanning'] if source else ['dos','fuzzers','generic','normal','reconnaissance']
    x=x.assign(label=y).loc[y.isin(classes)]
    raw_n=len(x)
    x=x.drop_duplicates()
    counts=x.label.value_counts()
    unique_n=len(x)
    # Remove every vector with conflicting labels before splitting.
    conflict=x.duplicated(subset=list(x.columns[:-1]),keep=False)
    x=x.loc[~conflict].reset_index(drop=True)
    return x,classes,dict(retained_before_dedup=raw_n,unique_vector_label_pairs=unique_n,
                          conflicting_rows_removed=int(conflict.sum()),clean_rows=len(x),
                          pre_conflict_counts=counts.to_dict())

def split(d, classes, seed, target):
    if target:
        normal=d[d.label=='normal']; attacks=d[d.label!='normal']
        d=pd.concat([normal.sample(n=min(len(normal),len(attacks)),random_state=seed),attacks])
    y=pd.Categorical(d.label,categories=classes).codes
    x=d.drop(columns='label').to_numpy(np.float32)
    assert len(pd.DataFrame(x).drop_duplicates())==len(x)
    tr,tmp=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
    va,te=train_test_split(tmp,test_size=2/3,stratify=y[tmp],random_state=seed)
    x[:,:12]=np.log1p(x[:,:12])
    scaler=StandardScaler().fit(x[tr,:12])
    x[:,:12]=scaler.transform(x[:,:12])
    return [(x[i],y[i]) for i in (tr,va,te)]

def extractor():
    inp=keras.Input((29,1));x=inp
    for n in (32,64):
        x=keras.layers.Conv1D(n,10,padding='same')(x)
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.ReLU()(x)
        x=keras.layers.MaxPooling1D(2)(x)
    return keras.Model(inp,keras.layers.Flatten()(x))

def train_cnn(ext, classes, parts, epochs, frozen, seed):
    keras.utils.set_random_seed(seed)
    ext.trainable=not frozen
    inp=keras.Input((29,1)); x=ext(inp,training=False) if frozen else ext(inp)
    for n in (128,64):
        x=keras.layers.Dense(n,activation='relu')(x)
        x=keras.layers.Dropout(.3)(x)
    model=keras.Model(inp,keras.layers.Dense(classes,activation='softmax')(x))
    model.compile(optimizer=keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy')
    def dataset(part):
        x,y=part
        options=tf.data.Options()
        options.threading.private_threadpool_size=2
        return tf.data.Dataset.from_tensor_slices((x[:,:,None],y)).batch(256).with_options(options)
    x,y=parts[0]
    options=tf.data.Options()
    options.threading.private_threadpool_size=2
    training=tf.data.Dataset.from_tensor_slices((x[:,:,None],y)).shuffle(len(y),seed=seed).batch(256).with_options(options)
    start=time.perf_counter()
    history=model.fit(training,validation_data=dataset(parts[1]),epochs=epochs,verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
    duration=time.perf_counter()-start
    test,y=parts[2]
    pred=np.concatenate([model(test[i:i+256,:,None],training=False).numpy().argmax(1) for i in range(0,len(test),256)])
    return metrics(y,pred,classes,duration,len(history.epoch))

def metrics(y,pred,classes,duration,epochs=None):
    labels=list(range(classes))
    return dict(macro_f1=f1_score(y,pred,labels=labels,average='macro',zero_division=0),
                accuracy=accuracy_score(y,pred),seconds=duration,epochs=epochs,
                confusion=confusion_matrix(y,pred,labels=labels).tolist())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ton','unsw','output'):p.add_argument(name,type=Path)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4])
    a=p.parse_args()
    s,sc,sa=load(a.ton,'ton');t,tc,ta=load(a.unsw,'unsw')
    results=dict(source=sa,target=ta,source_classes=sc,target_classes=tc,features=list(t.columns[:-1]),
                 epochs=a.epochs,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.ton,a.unsw)},runs=[])
    print(sa,ta,flush=True)
    for seed in a.seeds:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        sp=split(s,sc,seed,False);tp=split(t,tc,seed,True)
        ext=extractor()
        assert ext.count_params()==21280
        row=dict(seed=seed,sizes=[len(y) for _,y in tp],source=train_cnn(ext,3,sp,a.epochs,False,seed))
        initial=extractor()
        row['transfer']=train_cnn(ext,5,tp,a.epochs,True,seed+100)
        row['random_frozen']=train_cnn(initial,5,tp,a.epochs,True,seed+100)
        fresh=extractor()
        fresh.set_weights(initial.get_weights())
        row['scratch']=train_cnn(fresh,5,tp,a.epochs,False,seed+100)
        start=time.perf_counter()
        clf=XGBClassifier(n_estimators=300,max_depth=8,learning_rate=.1,n_jobs=2,random_state=seed)
        clf.fit(*tp[0])
        dur=time.perf_counter()-start
        row['xgboost']=metrics(tp[2][1],clf.predict(tp[2][0]),5,dur)
        results['runs'].append(row)
        a.output.write_text(json.dumps(results,indent=2))
        print(seed,{k:round(row[k]['macro_f1'],4) for k in ('source','transfer','scratch','random_frozen','xgboost')},flush=True)

if __name__=='__main__':main()
