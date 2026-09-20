import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd
plt.rcParams.update({'font.size':7.5,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.5,'xtick.major.width':0.5,'ytick.major.width':0.5,'grid.color':'#e3e3e0','grid.linewidth':0.4,
    'legend.frameon':False,'legend.fontsize':6.8,'axes.labelsize':7.5,'axes.titlesize':8,'pdf.fonttype':42})
BLUE,ORANGE,AQUA,YELLOW,MAG,VIOLET,RED='#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#4a3aa7','#e34948'
INK,INK2='#0b0b0b','#52514e'
R=json.load(open('final_results.json')); C=R['cats']
d=R['minilm_2018']; tm=d['test_months']; x=pd.to_datetime(tm)
def roll(a,w=3): return pd.Series(a).rolling(w,min_periods=1,center=True).mean().values
def series(k): return np.array([d['methods'][k][m][0] for m in tm])
def sizes(k): return np.array([d['methods'][k][m][1] for m in tm])
COLW=3.45  # IEEE column width inches

# ---------- Fig 1: expiry + estimator + per-class
fig,(a1,a2)=plt.subplots(2,1,figsize=(COLW,3.6),gridspec_kw={'height_ratios':[1.25,1]})
a1.axhline(0.9,color=INK2,lw=0.6,ls=(0,(4,2))); a1.text(x[0],0.902,'target 1$-\\alpha$ = 0.90',fontsize=6.5,color=INK2,va='bottom')
a1.plot(x,series('static'),color=BLUE,lw=0.6,alpha=0.35)
a1.plot(x,roll(series('static')),color=BLUE,lw=1.6,label='True coverage (static split CP)')
a1.plot(x,roll([d['est_static'][m] for m in tm]),color=ORANGE,lw=1.4,ls='--',label='Label-free estimate $\\hat{c}_t$ (ours)')
a1.plot(x,roll([d['acc'][m] for m in tm]),color=INK2,lw=1.0,ls=':',label='Top-1 accuracy of frozen classifier')
a1.set_ylim(0.68,0.95); a1.set_ylabel('Fraction'); a1.grid(axis='y'); a1.legend(loc='lower left',ncol=1)
a1.set_title('(a) Static threshold calibrated on 2018 (3-mo. rolling)',loc='left',fontsize=7.5)
# per-class
y19=[];y26=[]
for k,c in enumerate(C):
    y19.append(np.nanmean([d['percls'][m][k] if d['percls'][m][k] is not None else np.nan for m in tm if m[:4]=='2019']))
    y26.append(np.nanmean([d['percls'][m][k] if d['percls'][m][k] is not None else np.nan for m in tm if m[:4]=='2026']))
order=np.argsort(np.array(y26)-np.array(y19))
idx=np.arange(len(C)); w=0.38
a2.bar(idx-w/2,np.array(y19)[order],w,color='#9ec5f4',label='2019'); a2.bar(idx+w/2,np.array(y26)[order],w,color=BLUE,label='2026')
a2.axhline(0.9,color=INK2,lw=0.6,ls=(0,(4,2)))
a2.set_xticks(idx); a2.set_xticklabels([C[i].replace('cs.','') for i in order],fontsize=6.5); a2.set_ylim(0.5,1.0); a2.set_ylabel('Class-cond. coverage')
a2.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=2,handlelength=1.2,columnspacing=1.0); a2.grid(axis='y'); a2.set_ylim(0.5,1.06)
a2.set_title('(b) Per-category coverage of the same threshold',loc='left',fontsize=7.5)
fig.tight_layout(h_pad=1.0); fig.savefig('paper/fig1_expiry.pdf'); plt.close()

# ---------- Fig 2: methods over time (coverage + set size)
fig,(a1,a2)=plt.subplots(2,1,figsize=(COLW,3.3),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
meth=[('static','Static (no labels)',INK2,':'),('sliding_1','Sliding-1 (all labels)',AQUA,'-'),('aci_0.01','ACI (all labels)',VIOLET,'-'),('selfcal','SelfCal, ours (no labels)',ORANGE,'-')]
for k,l,c,ls in meth:
    a1.plot(x,roll(series(k)),color=c,lw=1.5 if k!='static' else 1.3,ls=ls,label=l)
    a2.plot(x,roll(sizes(k)),color=c,lw=1.5 if k!='static' else 1.3,ls=ls)
a1.axhline(0.9,color=INK2,lw=0.6,ls=(0,(4,2))); a1.set_ylim(0.8,0.95); a1.set_ylabel('Monthly coverage'); a1.grid(axis='y'); a1.legend(loc='lower left',ncol=2,handlelength=1.6,columnspacing=0.8)
a2.set_ylabel('Mean set size'); a2.grid(axis='y'); a2.set_ylim(1.2,1.85)
a1.set_title('Deployment A (MiniLM, cal. 2018), 3-mo. rolling',loc='left',fontsize=7.5)
fig.tight_layout(h_pad=0.6); fig.savefig('paper/fig2_methods.pdf'); plt.close()

# ---------- Fig 3: label budget vs mean |dev|, both deployments (stacked)
fig,axes=plt.subplots(2,1,figsize=(COLW,3.1),sharex=True)
for ax,key,title in zip(axes,['minilm_2018','minilm_2021'],['Deployment A (cal. 2018)','Deployment B (cal. 2021)']):
    dd=R[key]; tmm=dd['test_months']
    def st(k):
        c=np.array([dd['methods'][k][m][0] for m in tmm]); return np.abs(c-0.9).mean()
    per={'periodic_3':1/3,'periodic_6':1/6,'periodic_12':1/12}
    ax.scatter(list(per.values()),[st(k) for k in per],marker='s',s=18,color=BLUE,label='Periodic-$k$ (k=12,6,3)',zorder=3)
    rp=[(b,np.mean([st(f'random_{b}_{r}') for r in range(10)])) for b in [0.05,0.1,0.2,0.5]]
    ax.plot([p[0] for p in rp],[p[1] for p in rp],'-o',ms=3.5,lw=1,color=INK2,label='Random-$p$ (mean of 10 seeds)',zorder=2)
    ax.scatter([1.0],[st('sliding_1')],marker='D',s=18,color=AQUA,label='Sliding-1',zorder=3)
    bt=len(dd['trig'])/len(tmm)
    ax.scatter([bt],[st('selftrig')],marker='^',s=32,color=ORANGE,label='SelfTrig (ours)',zorder=4)
    ax.scatter([0.0],[st('selfcal')],marker='*',s=60,color=ORANGE,label='SelfCal (ours)',zorder=4)
    ax.scatter([0.0],[st('static')],marker='x',s=24,color=RED,label='Static',zorder=4)
    ax.set_title(title,loc='left',fontsize=7.5); ax.grid(axis='y'); ax.set_xlim(-0.04,1.04)
    ax.set_ylabel('Mean |cov. $-$ 0.90|')
axes[1].set_xlabel('Fraction of stream months whose labels are requested')
axes[0].legend(loc='upper right',ncol=2,fontsize=6.3,handlelength=1.2,columnspacing=0.8)
axes[0].set_ylim(0.012,0.034); axes[1].set_ylim(0.012,0.036)
fig.tight_layout(h_pad=0.8); fig.savefig('paper/fig3_budget.pdf'); plt.close()
print('figs done')
