import pandas as pd, numpy as np, json
ep=pd.Timestamp('2025-01-01',tz='UTC')
te=pd.read_csv('data/sim_fleet/SAT_2/resampled_30s_test.csv.gz',parse_dates=['timestamp'])
d=(te.timestamp-ep).dt.total_seconds()/86400
n={'channel_2':'rw2_speed','channel_6':'rw2_cur','channel_1':'rw1_speed','channel_3':'rw3_speed','channel_4':'rw4_speed','channel_12':'att_err'}
rows=[]
for lo,hi,nm in [(200,205,'d200-205 healthy'),(296,300,'d296-300'),(330,335,'d330-335'),(350,355,'d350-355'),(355,356.1,'d355-356.1'),(356.11,356.2,'d356.11-.2 just failed'),(357,358,'d357-358'),(362,365,'d362-365')]:
    x=te[(d>=lo)&(d<hi)]
    r={'win':nm}
    for c,k in n.items():
        r[k+'_medabs']=round(x[c].abs().median(),3); 
    r['att_err_p99']=round(x.channel_12.abs().quantile(.99),3); r['rw2cur_p99']=round(x.channel_6.abs().quantile(.99),3)
    rows.append(r)
print(pd.DataFrame(rows).to_string())
R=json.load(open('fed/out_final/results.json'))
T={}
for s in ['SAT_1','SAT_2','SAT_3','SAT_4']:
    t=pd.read_csv(f'data/sim_fleet/{s}/truth/events_truth.csv',parse_dates=['PhysicalStart']); t['day']=(t.PhysicalStart-ep).dt.total_seconds()/86400; T[s]=t.set_index('ID').day
for arm,sel in [('solo','local'),('fed','federated'),('pers','federated')]:
    e=pd.DataFrame(R[arm]['SAT_2'][sel]['events'])
    print(arm,sel,'SAT_2 by code', {c:f"{int(g.detected.sum())}/{len(g)}" for c,g in e.groupby('code')}, 'W rows', e[e.code.isin(['W2','W3'])][['code','detected','delay_min']].to_dict('records'))
    for s in T:
        e=pd.DataFrame(R[arm][s][sel]['events']); e['day']=e.id.map(T[s])
        a=e[e.day<296]; b=e[e.day>=296]
        print('  ',s,'before d296',f"{int(a.detected.sum())}/{len(a)}",'| d296+',f"{int(b.detected.sum())}/{len(b)}", '| F0.5 all %.3f anom %.3f falarms %s'%(R[arm][s][sel]['all']['global_EW_F_0.50'],R[arm][s][sel]['anomalies']['global_EW_F_0.50'],R[arm][s][sel]['n_false_alarm_events']))
