"""Robustness study for the Hybrid LinUCB aligned report.

python extend_and_audit.py --yahoo-project reproduction
Without --yahoo-project only the independent synthetic replication is run.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,csv,hashlib,json,platform,sys,time
import numpy as np
import scipy
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn,joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reproduction'))
import experiments as E
OUT=ROOT/'robustness'; OUT.mkdir(exist_ok=True)
SEEDS=list(range(100,120)); T=3000; N=len(SEEDS)
CONFIG=[('Random',None,.25,True,0.),('LinUCB (disjoint)',None,.25,False,0.),
        ('Window-100',100,.25,False,0.),('Window-300',300,.25,False,0.),
        ('Window-600',600,.25,False,0.),('contextual greedy',None,0.,False,.05)]
def dumpcsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def simulate():
    rows=[]; curves={}
    for scenario in ['stationary','abrupt','gradual']:
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            x=rng.normal(size=(T,6,6)); x/=np.linalg.norm(x,axis=2,keepdims=True)
            theta=rng.normal(size=(6,6)); theta/=np.linalg.norm(theta,axis=1,keepdims=True)
            w=np.zeros(T)
            if scenario=='abrupt': w[T//2:]=1
            if scenario=='gradual': w=np.clip((np.arange(T)-T//3)/(T/3),0,1)
            coeff=(1-2*w[:,None,None])*theta
            mu=.5+.35*np.einsum('tkd,tkd->tk',x,coeff)
            assert mu.min()>=.15-1e-12 and mu.max()<=.85+1e-12
            reward=rng.random((T,6))<mu
            for features in ['original6','intercept7']:
                X=x if features=='original6' else np.concatenate([np.ones((T,6,1)),x],axis=2)
                for name,window,alpha,random,eps in CONFIG:
                    p=E.LinUCBDisjoint(6,X.shape[-1],alpha=alpha,window=window,
                                     random=random,epsilon=eps,seed=50000+seed)
                    reg=np.empty(T); clicks=0
                    for t in range(T):
                        a=p.choose(np.arange(6),X[t]); rr=int(reward[t,a]); p.update(a,X[t,a],rr)
                        reg[t]=mu[t].max()-mu[t,a]; clicks+=rr
                    row={'scenario':scenario,'feature_map':features,'seed':seed,'policy':name,
                         'dynamic_regret':float(reg.sum()),'click_rate':clicks/T,
                         'second_half_regret':float(reg[T//2:].sum())}
                    rows.append(row); curves.setdefault((scenario,features,name),[]).append(np.cumsum(reg))
        print('Replication completed',scenario,flush=True)
    critical=float(student_t.ppf(.975,N-1)); adjusted=float(student_t.ppf(1-.05/(2*6),N-1))
    summary={}; contrasts={}
    for scenario in ['stationary','abrupt','gradual']:
        summary[scenario]={}; contrasts[scenario]={}
        for feature in ['original6','intercept7']:
            summary[scenario][feature]={}
            for name,*_ in CONFIG:
                a=np.array([v['dynamic_regret'] for v in rows if v['scenario']==scenario and v['feature_map']==feature and v['policy']==name])
                sd=float(a.std(ddof=1)); er=critical*sd/np.sqrt(N)
                summary[scenario][feature][name]={'mean':float(a.mean()),'sd':sd,'mean_ci95':[float(a.mean()-er),float(a.mean()+er)]}
            a=np.array([v['dynamic_regret'] for v in rows if v['scenario']==scenario and v['feature_map']==feature and v['policy']=='Window-300'])
            b=np.array([v['dynamic_regret'] for v in rows if v['scenario']==scenario and v['feature_map']==feature and v['policy']=='LinUCB (disjoint)'])
            delta=a-b; half=adjusted*delta.std(ddof=1)/np.sqrt(N)
            contrasts[scenario][feature]={'mean_window_minus_linucb':float(delta.mean()),'family_ci95':[float(delta.mean()-half),float(delta.mean()+half)]}
    dumpcsv(OUT/'simulation_per_seed.csv',rows)
    np.savez_compressed(OUT/'curves.npz',**{'__'.join(k):np.array(v) for k,v in curves.items()})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12.8,6.4),constrained_layout=True)
    for i,features in enumerate(['original6','intercept7']):
        for j,scenario in enumerate(['stationary','abrupt','gradual']):
            ax=axes[i,j]
            for name in ['Random','LinUCB (disjoint)','Window-300']:
                arr=np.array(curves[scenario,features,name]); mean=arr.mean(0); err=critical*arr.std(0,ddof=1)/np.sqrt(N)
                ax.plot(np.arange(1,T+1),mean,label=name); ax.fill_between(np.arange(1,T+1),mean-err,mean+err,alpha=.15)
            if scenario=='abrupt':ax.axvline(1500,color='grey',ls='--',lw=.8)
            ax.set(title=f'{scenario.capitalize()} / {features}',xlabel='Round',ylabel='Cumulative pseudo-regret'); ax.grid(alpha=.15)
    axes[0,-1].legend(fontsize=8)
    fig.savefig(OUT/'replication_curves.pdf'); fig.savefig(OUT/'replication_curves.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.4),constrained_layout=True)
    labels=['LinUCB (disjoint)','Window-100','Window-300','Window-600','contextual greedy']
    for ax,scenario in zip(axes,['stationary','abrupt','gradual']):
        vals=[summary[scenario]['original6'][n] for n in labels]
        ax.bar(range(5),[a['mean'] for a in vals],yerr=[a['mean_ci95'][1]-a['mean'] for a in vals],capsize=3,color='#3d8b86')
        ax.set_xticks(range(5),['All history','W=100','W=300','W=600','contextual greedy'],rotation=25,ha='right')
        ax.set(title=scenario.capitalize(),ylabel='Total pseudo-regret'); ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'window_ablation.pdf'); fig.savefig(OUT/'window_ablation.png',dpi=170); plt.close(fig)
    return {'seeds':SEEDS,'horizon':T,'policies':CONFIG,'summary':summary,'paired':contrasts,
            'mean_interval_critical':critical,'family_interval_critical':adjusted,
            'family_size':6,'trajectories':len(rows),'chosen_reward_only':True,
            'policy_seed_offset':50000,'feature_maps':['original6','intercept7']}

def audit_yahoo(project):
    E.ROOT=project; rows,af,ids,i,j,stats=E.load_data()
    assert stats['log_sha256']=='45a5ce2a1f64875389d5a5bcc65be8e95c3b6a91b53757e90b055da49915525f'
    membership=all(logged in arms for _,u,logged,r,arms in rows)
    unique=all(len(set(arms))==len(arms) for *_,arms in rows)
    users=np.array([v[1] for v in rows]); ts=np.array([v[0] for v in rows])
    seed_runs={}
    for method,filename,label in [('contextual_greedy','epsilon_seeds.csv','contextual greedy'),
                                  ('epsilon_greedy','context_free_epsilon_seeds.csv','ε-greedy (context-free)')]:
        runs=[]
        for seed in range(2026,2056):
            p=E.make_policy(method,.05,len(af),seed=seed)
            E.replay(rows[:j],af,p);v=E.replay(rows[j:],af,p)
            runs.append({'seed':seed,'matched':v['matched'],'clicks':v['clicks'],'replay_ctr':v['replay_ctr']})
            if (seed-2025)%10==0:print('Yahoo',label,'seeds completed',seed-2025,flush=True)
        dumpcsv(OUT/filename,runs)
        seed_runs[method]=runs
    summary={}
    for method,runs in seed_runs.items():
        arr=np.array([v['replay_ctr'] for v in runs])
        summary[method]={'n':30,'seed_range':[2026,2055],'mean':float(arr.mean()),
            'sd':float(arr.std(ddof=1)),
            'interpretation':'algorithm randomness on one fixed log; not independent users or population CI'}
    return {'data':stats,'logged_action_in_candidates':membership,'candidate_ids_unique':unique,
            'split_timestamps':{'development_last':int(ts[i-1]),'validation_first':int(ts[i]),
              'validation_last':int(ts[j-1]),'test_first':int(ts[j]),'test_last':int(ts[-1])},
            'user_feature_min':users.min(0).tolist(),'user_feature_max':users.max(0).tolist(),
            'article_feature_min':af.min(0).tolist(),'article_feature_max':af.max(0).tolist(),
            'epsilon_seed_summary':summary['contextual_greedy'],
            'seed_summaries':summary}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--yahoo-project',type=Path); args=ap.parse_args()
    start=time.time(); result={'simulation':simulate()}
    if args.yahoo_project: result['yahoo_audit']=audit_yahoo(args.yahoo_project.resolve())
    result['manifest']={'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'sklearn':sklearn.__version__,
        'joblib':joblib.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'experiments_sha256':hashlib.sha256((ROOT/'reproduction/experiments.py').read_bytes()).hexdigest(),
        'runtime_seconds':time.time()-start,'replication_seeds_exclude_poster_seeds':list(range(10))}
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Extension complete',result['manifest']['runtime_seconds'],flush=True)

if __name__=='__main__':main()
