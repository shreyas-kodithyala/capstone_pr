import pandas as pd, json
pd.set_option('display.width',220); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',80)
ep=pd.Timestamp('2025-01-01',tz='UTC')
R=json.load(open('fed/out_final/results.json'))
print(R['solo'].keys())
for a in R:
    print(a,'fleet',R[a]['fed_selection'])
    for s in ['SAT_1','SAT_2','SAT_3','SAT_4']:
        c=R[a][s]['local']['cfg']; print('  own',s,{k:c.get(k) for k in ['thr','W','spans','mask','smask','latch','close','minlen','L']}, 'val',round(R[a][s]['local']['val'],3))
arms=[('solo','local'),('solo','federated'),('fed','federated'),('pers','federated')]
for s in ['SAT_1','SAT_4']:
    t=pd.read_csv(f'data/sim_fleet/{s}/truth/events_truth.csv',parse_dates=['PhysicalStart']).set_index('ID')
    base=pd.DataFrame(R['solo'][s]['local']['events'])[['id','code','family']]
    base['day']=base.id.map(((t.PhysicalStart-ep).dt.total_seconds()/86400).round(1)); base['comp']=base.id.map(t.Component)
    for a,sel in arms:
        e=pd.DataFrame(R[a][s][sel]['events']).set_index('id')
        base[a+'_'+sel[:3]]=base.id.map(e.detected)
    print(s); print(base.to_string())
for s in ['SAT_1','SAT_2','SAT_3','SAT_4']:
    for a,sel in arms:
        x=R[a][s][sel]
        print(s,a,sel,'all %.3f anom %.3f det %d/%d FA %d alarmfrac %.3f'%(x['all']['global_EW_F_0.50'],x['anomalies']['global_EW_F_0.50'],x['n_detected'],x['n_anomalies'],x['n_false_alarm_events'],x['alarm_rows_frac']))
