import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp, pearsonr
s=pd.read_parquet('data/sample.parquet'); E=np.load('data/emb_minilm.npy')
CATS=sorted(s.primary.unique()); y=s.primary.map({c:i for i,c in enumerate(CATS)}).values
months=sorted(s.month.unique())
tr=(s.month<'2018-01').values; ca=((s.month>='2018-01')&(s.month<'2019-01')).values
clf=LogisticRegression(C=1.0,max_iter=3000).fit(E[tr],y[tr]); P=clf.predict_proba(E)
score=1-P[np.arange(len(y)),y]; conf=P.max(1)
n=ca.sum(); q0=np.sort(score[ca])[int(np.ceil((n+1)*0.9))-1]
Cset=(1-P)<=q0
mu_cal=E[ca].mean(0)
rows=[]
for m in months:
    if m<'2018-01': continue
    mm=(s.month==m).values
    cov=Cset[mm][np.arange(mm.sum()),y[mm]].mean()
    rows.append(dict(month=m, cov=cov,
        ks_conf=ks_2samp(conf[mm],conf[ca]).statistic,
        mean_conf=conf[mm].mean(),
        est_cov=(P[mm]*Cset[mm]).sum(1).mean(),   # model's own expected coverage
        set_size=Cset[mm].sum(1).mean(),
        frac_single=(Cset[mm].sum(1)==1).mean(),
        emb_shift=np.linalg.norm(E[mm].mean(0)-mu_cal),
        pred_entropy=(-(P[mm]*np.log(P[mm]+1e-12)).sum(1)).mean()))
df=pd.DataFrame(rows); te=df[df.month>='2019-01']
for c in ['ks_conf','mean_conf','est_cov','set_size','frac_single','emb_shift','pred_entropy']:
    r=pearsonr(te[c],te['cov'])[0]; print(f"{c:14s} corr with coverage r={r:+.3f}")
print(te.groupby(te.month.str[:4])[['cov','mean_conf','est_cov','emb_shift','ks_conf']].mean().round(3))
df.to_csv('monthly_stats.csv',index=False)
