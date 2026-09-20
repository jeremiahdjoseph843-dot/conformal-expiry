import numpy as np, pandas as pd, json, time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from scipy.stats import pearsonr
ALPHA=0.1; EPS=0.02
s=pd.read_parquet('data/sample.parquet'); E=np.load('data/emb_minilm.npy')
CATS=sorted(s.primary.unique()); y=s.primary.map({c:i for i,c in enumerate(CATS)}).values
months=sorted(s.month.unique()); K=len(CATS)
def conf_q(sc,alpha=ALPHA):
    n=len(sc); k=int(np.ceil((n+1)*(1-alpha))); return np.inf if k>n else np.sort(sc)[k-1]
def wconf_q(sc,w,alpha=ALPHA):
    o=np.argsort(sc); sc=sc[o]; ww=w[o]/(w.sum()+1.0); cs=np.cumsum(ww); i=np.searchsorted(cs,1-alpha)
    return sc[i] if i<len(sc) else np.inf
def est_cov(P,q): C=(1-P)<=q; return (P*C).sum(1).mean()
def selfcal_q(P,target):
    cand=np.unique(np.round(1-P,4)); lo,hi=0,len(cand)-1
    if est_cov(P,cand[hi])<target: return cand[hi]
    while lo<hi:
        mid=(lo+hi)//2
        if est_cov(P,cand[mid])>=target: hi=mid
        else: lo=mid+1
    return cand[lo]
def fit_T(L,yy):
    return minimize_scalar(lambda T:-log_softmax(L/T,axis=1)[np.arange(len(yy)),yy].mean(),bounds=(0.05,20),method='bounded').x
def ev(P,yy,q): C=(1-P)<=q; return float(C[np.arange(len(yy)),yy].mean()),float(C.sum(1).mean())

def run(train_end,cal_year,feats,name,alpha=ALPHA):
    tr=(s.month<train_end).values; ca=((s.month>=f'{cal_year}-01')&(s.month<f'{cal_year+1}-01')).values
    clf=LogisticRegression(C=1.0,max_iter=3000).fit(feats[tr],y[tr]); L=clf.decision_function(feats)
    T=fit_T(L[ca],y[ca]); P=softmax(L/T,axis=1); pred=P.argmax(1)
    score=1-P[np.arange(len(y)),y]
    q0=conf_q(score[ca],alpha); delta=(1-alpha)-est_cov(P[ca],q0)
    cal_m=[m for m in months if f'{cal_year}-01'<=m<f'{cal_year+1}-01']; test_m=[m for m in months if m>=f'{cal_year+1}-01']
    hist=cal_m+test_m
    M={}; rec=lambda k,m,v: M.setdefault(k,{}).__setitem__(m,v)
    acc={m:float((pred[(s.month==m).values]==y[(s.month==m).values]).mean()) for m in test_m}
    est_static={}; percls={}
    qk=[conf_q(score[ca&(y==k)],alpha) for k in range(K)]
    q1=q0; qt=q0; trig=[]; qp={k:q0 for k in [3,6,12]}
    rnd={(b,r):(np.random.RandomState(100*r+int(b*10)),q0) for b in [0.05,0.1,0.2,0.5] for r in range(10)}
    for j,m in enumerate(test_m):
        mm=(s.month==m).values; Pm=P[mm]; ym=y[mm]
        rec('static',m,ev(Pm,ym,q0)); est_static[m]=float(est_cov(Pm,q0)+delta)
        C=(1-Pm)<=q0; percls[m]=[float(C[ym==k,k].mean()) if (ym==k).sum()>0 else None for k in range(K)]
        Cm=np.stack([(1-Pm[:,k])<=qk[k] for k in range(K)],1); rec('mondrian',m,(float(Cm[np.arange(len(ym)),ym].mean()),float(Cm.sum(1).mean())))
        hh=hist[:len(cal_m)+j]
        for W in [1,12]:
            sel=s.month.isin(hh[-W:]).values; rec(f'sliding_{W}',m,ev(Pm,ym,conf_q(score[sel],alpha)))
        sel=s.month.isin(hh).values; w=0.9**(np.array([months.index(m)-months.index(x) for x in s.month.values[sel]]))
        rec('weighted_0.9',m,ev(Pm,ym,wconf_q(score[sel],w,alpha)))
        for k in qp: rec(f'periodic_{k}',m,ev(Pm,ym,qp[k]))
        for k in qp:
            if (j+1)%k==0: qp[k]=conf_q(score[mm],alpha)
        for key,(rr,q) in rnd.items():
            rec(f'random_{key[0]}_{key[1]}',m,ev(Pm,ym,q))
            if rr.rand()<key[0]: rnd[key]=(rr,conf_q(score[mm],alpha))
        rec('selfcal',m,ev(Pm,ym,q1)); rec('selftrig',m,ev(Pm,ym,qt))
        if est_cov(Pm,qt)+delta<1-alpha-EPS: trig.append(m); qt=conf_q(score[mm],alpha)
        q1=selfcal_q(Pm,1-alpha-delta)
    # ACI per-sample
    calsc=np.sort(score[ca]); order=np.where((s.month>=f'{cal_year+1}-01').values)[0]
    a=alpha; covs=np.zeros(len(order)); sz=np.zeros(len(order))
    for i,idx in enumerate(order):
        if a<=0: q=np.inf
        elif a>=1: q=-np.inf
        else: q=conf_q(calsc,a)
        C=(1-P[idx])<=q; err=1-float(C[y[idx]]); covs[i]=1-err; sz[i]=C.sum(); a=a+0.01*(alpha-err)
    mo=s.month.values[order]
    for m in test_m: rec('aci_0.01',m,(float(covs[mo==m].mean()),float(sz[mo==m].mean())))
    return dict(name=name,T=float(T),delta=float(delta),q0=float(q0),methods=M,acc=acc,est_static=est_static,percls=percls,trig=trig,
                test_months=test_m,cal_months=cal_m,n_train=int(tr.sum()),n_cal=int(ca.sum()),n_test=int(len(order)),
                cal_acc=float((pred[ca]==y[ca]).mean()),
                prior={m:np.bincount(y[(s.month==m).values],minlength=K).tolist() for m in test_m})
out={'cats':CATS}
t=time.time()
out['minilm_2018']=run('2018-01',2018,E,'MiniLM / cal 2018'); print('1',time.time()-t,flush=True)
out['minilm_2021']=run('2021-01',2021,E,'MiniLM / cal 2021'); print('2',time.time()-t,flush=True)
tv=TfidfVectorizer(max_features=50000,ngram_range=(1,2),sublinear_tf=True,min_df=3); tr=(s.month<'2018-01').values
X=tv.fit(s.text[tr]).transform(s.text)
out['tfidf_2018']=run('2018-01',2018,X,'TF-IDF / cal 2018'); print('3',time.time()-t,flush=True)
out['minilm_2018_a05']=run('2018-01',2018,E,'MiniLM / cal 2018 / alpha=0.05',alpha=0.05); print('4',time.time()-t,flush=True)
json.dump(out,open('final_results.json','w'))
print('saved')
