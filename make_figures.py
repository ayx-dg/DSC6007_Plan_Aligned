from pathlib import Path
import json,hashlib,csv,shutil
import numpy as np
from scipy.stats import t as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent; (O/'figures').mkdir(exist_ok=True)
R=json.loads((O/'core/results.json').read_text()); X=json.loads((O/'robustness/results.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
names=['random','disjoint','hybrid','contextual_greedy','epsilon_greedy','window','logistic']
labels=[R['display_labels'][n] for n in names]
def wilson(k,n):
    z=1.95996398454; p=k/n; den=1+z*z/n
    a=(p+z*z/(2*n))/den; d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return a-d,a+d
fig,ax=plt.subplots(1,2,figsize=(12.0,4.1),constrained_layout=True)
vals=[R['test'][n]['replay_ctr']*100 for n in names]
cis=[wilson(R['test'][n]['clicks'],R['test'][n]['matched']) for n in names]
ax[0].errorbar(vals,range(len(names)),xerr=[[v-100*c[0] for v,c in zip(vals,cis)],[100*c[1]-v for v,c in zip(vals,cis)]],fmt='o',capsize=3,color='#397990')
ax[0].set_yticks(range(len(names)),labels); ax[0].invert_yaxis()
ax[0].set(xlabel='Matched replay CTR (%)',title='Original protocol / nominal Wilson ranges'); ax[0].grid(axis='x',alpha=.2)
seed_files=[('random','random_seeds.csv','#849cb0'),
            ('contextual_greedy','contextual_greedy_seeds.csv','#318c83'),
            ('epsilon_greedy','epsilon_greedy_seeds.csv','#b05c3c')]
for i,(key,filename,color) in enumerate(seed_files):
    a=np.array([float(v['replay_ctr']) for v in csv.DictReader((O/'core'/filename).open())])*100
    ax[1].scatter(np.full(len(a),i),a,s=16,alpha=.4,color=color)
    ax[1].errorbar(i+.12,a.mean(),yerr=a.std(ddof=1),fmt='D',capsize=5,color='black')
ax[1].set_xticks(range(3),['Random','contextual greedy','ε-greedy (context-free)'])
ax[1].set(ylabel='Matched replay CTR (%)',title='30 seeds on the SAME fixed log / mean ± SD'); ax[1].grid(axis='y',alpha=.2)
fig.savefig(O/'figures/yahoo_results.pdf'); fig.savefig(O/'figures/yahoo_results.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9.4,3.8),constrained_layout=True)
for j,name in enumerate(['Random','LinUCB (disjoint)','Window LinUCB (300)']):
    arrays=[np.array([a['dynamic_regret'] for a in R['simulation'] if a['scenario']==s and a['policy']==name]) for s in ['stationary','abrupt','gradual']]
    ax.bar(np.arange(3)+(j-1)*.24,[a.mean() for a in arrays],.24,yerr=[st.ppf(.975,9)*a.std(ddof=1)/np.sqrt(10) for a in arrays],capsize=3,label=name)
ax.set_xticks(range(3),['Stationary','Abrupt','Gradual']);ax.set_ylabel('Dynamic pseudo-regret');ax.legend();ax.grid(axis='y',alpha=.15)
fig.savefig(O/'figures/poster_simulation.pdf');fig.savefig(O/'figures/poster_simulation.png',dpi=180);plt.close(fig)
for n in ['replication_curves.pdf','window_ablation.pdf']:
    shutil.copy2(O/'robustness'/n,O/'figures'/n)
for n in ['sparsity.pdf','sparsity.png']:
    src=O/'sparsity'/n
    if src.exists():shutil.copy2(src,O/'figures'/n)
