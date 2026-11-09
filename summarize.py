"""Validate run outputs and summarise the reconstruction descriptively."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

d=json.loads(Path('results.json').read_text())
assert len(d['features'])==29
assert len({row['seed'] for row in d['runs']})==len(d['runs'])
models=('transfer','scratch','random_frozen','xgboost')
for row in d['runs']:
    for name in models:
        r=row[name]
        assert sum(map(sum,r['confusion']))==row['sizes'][-1]
        assert 0<=r['macro_f1']<=1 and 0<=r['accuracy']<=1
with open('summary.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['model','completed_seeds','macro_f1_mean','macro_f1_sd','accuracy_mean','fit_seconds_mean'])
    for name in models:
        rows=[r[name] for r in d['runs']]
        values=[r['macro_f1'] for r in rows]
        w.writerow([name,len(rows),mean(values),stdev(values) if len(values)>1 else '',
                    mean(r['accuracy'] for r in rows),mean(r['seconds'] for r in rows)])
with open('paired_results.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['seed',*models,'transfer_minus_scratch','transfer_minus_random_frozen'])
    for row in d['runs']:
        values=[row[m]['macro_f1'] for m in models]
        w.writerow([row['seed'],*values,values[0]-values[1],values[0]-values[2]])
