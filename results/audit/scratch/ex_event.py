import pandas as pd, numpy as np
pd.set_option('display.width',200); pd.set_option('display.max_columns',40)
ch=pd.read_csv('data/sim_fleet/SAT_2/channels.csv'); print(ch.head(3).to_string()); 
import json
m=None
df=pd.read_csv('data/sim_fleet/SAT_2/resampled_30s_train.csv.gz',parse_dates=['timestamp'],nrows=70000)
print(df.columns.tolist()[:26])
ts=df.timestamp.dt.tz_localize(None) if df.timestamp.dt.tz is not None else df.timestamp
s,e=pd.Timestamp('2025-01-18 11:36'),pd.Timestamp('2025-01-18 12:17')
during=df[(ts>=s)&(ts<e)]; before=df[(ts>=s-pd.Timedelta('2h'))&(ts<s)]
nm=['channel_3','channel_7','channel_11','channel_12','channel_1','channel_2','channel_4']
for c in nm:
    print(c, 'before med %.4f std %.4f | during med %.4f max|.| %.4f'%(before[c].median(),before[c].std(),during[c].median(),during[c].abs().max()))
t=pd.read_csv('data/sim_fleet/SAT_2/truth/events_truth.csv')
print(t.iloc[0][['Class','Component','PhysicalStart','PhysicalEnd','FirstVisible','LastVisible','NChannels','Channels','Params']].to_string())
allv=[]
for s_ in ['SAT_1','SAT_2','SAT_3','SAT_4']:
    x=pd.read_csv(f'data/sim_fleet/{s_}/truth/events_truth.csv'); x['sat']=s_; allv.append(x)
a=pd.concat(allv)
print(a.groupby(['Category','Code']).size().to_string())
print(a.groupby('Category').size())
print('invisible',(~a.Visible).sum(), a[~a.Visible].Code.value_counts().to_dict())
print('---names', dict(zip(ch.Channel,ch.Name)))
lab=pd.read_csv('data/sim_fleet/SAT_2/labels.csv'); print(lab[lab.ID=='id_1'].to_string())
# rw3 current 2h before and 41 min during: robust numbers
print(df.loc[(ts>=s-pd.Timedelta('2min'))&(ts<s+pd.Timedelta('3min')),['timestamp','channel_3','channel_7','channel_12']].to_string())
