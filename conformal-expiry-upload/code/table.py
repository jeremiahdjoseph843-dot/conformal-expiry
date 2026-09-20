import json, numpy as np
R=json.load(open('final_results.json'))
def stats(d,k,alpha=0.1):
    tm=d['test_months']; v=d['methods'][k]; c=np.array([v[m][0] for m in tm]); sz=np.array([v[m][1] for m in tm])
    last=np.array([v[m][0] for m in tm if m[:4]=='2026'])
    return c.mean(),np.abs(c-(1-alpha)).mean(),c.min(),(c<(1-alpha-0.05)).mean(),sz.mean(),last.mean()
for key in ['minilm_2018','minilm_2021','tfidf_2018','minilm_2018_a05']:
    d=R[key]; al=0.05 if 'a05' in key else 0.1
    print(f"\n=== {d['name']} T={d['T']:.3f} delta={d['delta']:.4f} q0={d['q0']:.3f} ntr={d['n_train']} ncal={d['n_cal']} nte={d['n_test']} cal_acc={d['cal_acc']:.3f}")
    tm=d['test_months']; yrs=sorted(set(m[:4] for m in tm))
    print('acc:',{yy:round(np.mean([d['acc'][m] for m in tm if m[:4]==yy]),3) for yy in yrs})
    print('static cov:',{yy:round(np.mean([d['methods']['static'][m][0] for m in tm if m[:4]==yy]),3) for yy in yrs})
    print('est cov  :',{yy:round(np.mean([d['est_static'][m] for m in tm if m[:4]==yy]),3) for yy in yrs})
    tc=np.array([d['methods']['static'][m][0] for m in tm]); ec=np.array([d['est_static'][m] for m in tm])
    print('corr est vs true monthly: %.3f  MAE %.4f'%(np.corrcoef(tc,ec)[0,1],np.abs(tc-ec).mean()))
    print('selftrig months:',d['trig'],'budget %.3f'%(len(d['trig'])/len(tm)))
    print(f"{'method':14s} {'cov':>6s} {'|dev|':>6s} {'min':>6s} {'fail':>5s} {'size':>5s} {'2026':>6s}")
    for k in ['static','mondrian','sliding_1','sliding_12','weighted_0.9','aci_0.01','periodic_3','periodic_6','periodic_12','selfcal','selftrig']:
        print(f"{k:14s} "+" ".join(f"{x:6.3f}" for x in stats(d,k,al)))
    for b in [0.05,0.1,0.2,0.5]:
        a=np.array([stats(d,f'random_{b}_{r}',al) for r in range(10)]).mean(0); print(f"random_{b:<7} "+" ".join(f"{x:6.3f}" for x in a))
d=R['minilm_2018']; C=R['cats']; tm=d['test_months']
print('\nper-class static cov 2019 -> 2026, share')
for k,c in enumerate(C):
    a=np.nanmean([d['percls'][m][k] if d['percls'][m][k] is not None else np.nan for m in tm if m[:4]=='2019']); b=np.nanmean([d['percls'][m][k] if d['percls'][m][k] is not None else np.nan for m in tm if m[:4]=='2026'])
    s19=sum(d['prior'][m][k] for m in tm if m[:4]=='2019')/sum(sum(d['prior'][m]) for m in tm if m[:4]=='2019'); s26=sum(d['prior'][m][k] for m in tm if m[:4]=='2026')/sum(sum(d['prior'][m]) for m in tm if m[:4]=='2026')
    print(f"{c} {a:.3f} {b:.3f} {s19:.3f} {s26:.3f}")
