import pandas as pd
t=pd.read_csv('data/sim_fleet/SAT_2/truth/events_truth.csv',parse_dates=['PhysicalStart','FirstVisible','LastVisible'])
ep=pd.Timestamp('2025-01-01',tz='UTC')
w=t[t.Code.isin(['W1','W2','W3'])].copy()
for c in ['PhysicalStart','FirstVisible','LastVisible']: w[c+'_d']=(w[c]-ep).dt.total_seconds()/86400
print(w[['ID','Code','Split','Phase','PhysicalStart_d','FirstVisible_d','LastVisible_d','NChannels','Params']].round(2).to_string())
h=pd.read_csv('data/sim_fleet/SAT_2/truth/health.csv'); print(h.columns.tolist()); print(h.iloc[::max(1,len(h)//14)].to_string())
