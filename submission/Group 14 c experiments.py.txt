"""Chronological Yahoo replay, supervised baseline, and controlled drift study.

Run directly in PyCharm. Paths are relative to this file, never the working directory.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,copy,csv,hashlib,json,platform,time,sys
from collections import Counter,deque
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score,roc_auc_score,log_loss,
 brier_score_loss,precision_recall_curve,precision_score,recall_score,f1_score,confusion_matrix)
from Yahoo_recommendation.runner import load_articles,iter_log_lines

ROOT=Path(__file__).resolve().parent

class LinUCBDisjoint:
    """Li et al. (2010) Algorithm 1: disjoint LinUCB, one ridge model per article on the 6D user context."""
    def __init__(self,n,d,alpha=1.,epsilon=0.,seed=2026,window=None,random=False):
        self.inv=np.repeat(np.eye(d)[None],n,axis=0);self.b=np.zeros((n,d))
        self.theta=np.zeros((n,d));self.alpha=alpha;self.epsilon=epsilon
        self.rng=np.random.default_rng(seed);self.window=window;self.history=deque()
        self.random=random;self.t=0;self.n=n;self.d=d
    def expire(self):
        if self.window is None:return
        while self.history and self.history[0][0]<self.t-self.window:
            _,a,x,r=self.history.popleft();v=self.inv[a]@x;den=1-x@v
            if den<=1e-9:
                # Stable rebuild when a downdate is poorly conditioned.
                A=np.eye(self.d);b=np.zeros(self.d)
                for _,aa,xx,rr in self.history:
                    if aa==a:A+=np.outer(xx,xx);b+=rr*xx
                self.inv[a]=np.linalg.inv(A);self.b[a]=b
            else:self.inv[a]+=np.outer(v,v)/den;self.b[a]-=r*x
            self.theta[a]=self.inv[a]@self.b[a]
    def choose(self,arms,x):
        self.t+=1;self.expire()
        if self.random or self.rng.random()<self.epsilon:return int(self.rng.integers(len(arms)))
        means=np.einsum('ij,ij->i',self.theta[arms],x)
        var=np.einsum('ij,ijk,ik->i',x,self.inv[arms],x)
        return int(np.argmax(means+self.alpha*np.sqrt(np.maximum(var,0))))
    def update(self,a,x,r):
        v=self.inv[a]@x;self.inv[a]-=np.outer(v,v)/(1+x@v)
        self.b[a]+=r*x;self.theta[a]=self.inv[a]@self.b[a]
        if self.window is not None:self.history.append((self.t,a,x.copy(),r))

class LinUCBHybrid:
    """Li et al. (WWW 2010) Algorithm 2, Hybrid LinUCB: per-arm theta on user x, shared beta on interaction z.

    x is the six-dimensional user feature.  Following Li et al. (2010),
    Sec. 5.2.2, z is the flattened outer product of the user and article
    six-vectors (6 x 6 = 36 dimensions); it therefore carries user-article
    interaction information that the disjoint context x does not have.

    Score = x^T theta_a + z^T beta
            + alpha * sqrt(x^T A_a^{-1} x
                           - 2 x^T A_a^{-1} B_a S^{-1} z
                           + z^T S^{-1} z),
    with A_a = I + sum x x^T, B_a = sum x z^T, b_a = sum r x,
    A_0 = I + sum z z^T, b_0 = sum r z,
    S = A_0 - sum_a B_a^T A_a^{-1} B_a,
    beta = S^{-1}(b_0 - sum_a B_a^T A_a^{-1} b_a),
    theta_a = A_a^{-1}(b_a - B_a beta).
    Only matched feedback updates state, exactly as in the disjoint policy.
    """
    def __init__(self,n,d=6,k=36,alpha=1.,epsilon=0.,seed=2026):
        self.n=n;self.d=d;self.k=k
        self.inv=np.repeat(np.eye(d)[None],n,axis=0)
        self.b=np.zeros((n,d));self.B=np.zeros((n,d,k))
        self.A0=np.eye(k);self.b0=np.zeros(k);self.beta=np.zeros(k)
        self.theta=np.zeros((n,d));self.S_inv=np.eye(k)
        self.alpha=alpha;self.epsilon=epsilon;self.rng=np.random.default_rng(seed)
    def _refresh(self):
        # inv[n]=A_a^{-1} (d,d); B[n]=B_a (d,k); Schur complement S=A_0-sum B^T A^{-1} B.
        # BD[n]=A_a^{-1} B_a (d,k) using symmetry of A_a^{-1}.
        BD=np.einsum('nik,nij->njk',self.B,self.inv)
        S=self.A0-np.einsum('njk,njm->nkm',BD,self.B).sum(0)
        self.S_inv=np.linalg.inv(S)
        invb=np.einsum('nij,nj->ni',self.inv,self.b)
        g=self.b0-np.einsum('nik,ni->nk',self.B,invb).sum(0)
        self.beta=self.S_inv@g
        self.theta=np.einsum('nij,nj->ni',self.inv,self.b-np.einsum('nik,k->ni',self.B,self.beta))
    def choose(self,arms,x,z):
        if self.rng.random()<self.epsilon:return int(self.rng.integers(len(arms)))
        inv=self.inv[arms];B=self.B[arms]
        means=np.einsum('ij,ij->i',self.theta[arms],x)+z@self.beta
        # Li et al. 2010 Algorithm 2 line 13:
        # s = x^T A^{-1}x + (z - B^T A^{-1}x)^T S^{-1}(z - B^T A^{-1}x)
        #   = z^T S^{-1}z - 2z^T S^{-1}B^T A^{-1}x + x^T A^{-1}x + x^T A^{-1}B S^{-1}B^T A^{-1}x
        ax=np.einsum('ij,ijk,ik->i',x,inv,x)
        A_inv_B=np.einsum('ijk,ikl->ijl',inv,B)
        w=np.einsum('ij,ijk->ik',x,A_inv_B)          # B_a^T A_a^{-1} x
        diff=z-w
        var=ax+np.einsum('ik,kl,il->i',diff,self.S_inv,diff)
        return int(np.argmax(means+self.alpha*np.sqrt(np.maximum(var,0))))
    def update(self,a,x,z,r):
        v=self.inv[a]@x;self.inv[a]-=np.outer(v,v)/(1+x@v)
        self.b[a]+=r*x;self.B[a]+=np.outer(x,z)
        self.A0+=np.outer(z,z);self.b0+=r*z
        self._refresh()

class ContextualGreedy(LinUCBDisjoint):
    """Contextual-greedy comparator: disjoint ridge means, zero UCB width, epsilon exploration.

    This is the contextual epsilon-greedy variant used in the original project,
    renamed to avoid confusing it with the paper's context-free epsilon-greedy.
    """
    def __init__(self,n,d=6,epsilon=.05,seed=2026):
        super().__init__(n,d,alpha=0.,epsilon=epsilon,seed=seed)

class EpsilonGreedy:
    """Li et al. (2010), Sections 2.2 and 5.3: the original context-free epsilon-greedy.

    It ignores all user/article features.  It keeps an empirical click mean for
    each article, exploits the candidate with the highest mean with probability
    1-epsilon, and chooses a uniformly random candidate with probability epsilon.
    """
    def __init__(self,n,epsilon=.05,seed=2026):
        self.n=n;self.epsilon=epsilon;self.rng=np.random.default_rng(seed)
        self.counts=np.zeros(n);self.clicks=np.zeros(n);self.means=np.zeros(n)
    def choose(self,arms,x=None,z=None):
        if self.rng.random()<self.epsilon:return int(self.rng.integers(len(arms)))
        return int(np.argmax(self.means[arms]))
    def update(self,a,x,r):
        self.counts[a]+=1;self.clicks[a]+=r;self.means[a]=self.clicks[a]/self.counts[a]

def load_data():
    data=ROOT/'Yahoo_recommendation/data';articles=load_articles(data/'webscope-articles.txt')
    ids=sorted(articles);index={a:i for i,a in enumerate(ids)};af=np.array([articles[a] for a in ids])
    raw=sorted(iter_log_lines(data/'webscope-logs.txt'),key=lambda r:(r[1],r[0]))
    rows=[]
    for _,t,u,a,r,c in raw:
        arms=np.array([index[v] for v in c]);u=np.array(u)
        rows.append((t,u,index[a],r,arms))
    ts=np.array([r[0] for r in rows]);i=int(np.searchsorted(ts,ts[int(.70*len(ts))],side='left'))
    j=int(np.searchsorted(ts,ts[int(.85*len(ts))],side='left'))
    if not 0<i<j<len(rows):raise ValueError('Insufficient distinct timestamps for split')
    if not np.isfinite(af).all() or not all(np.isfinite(r[1]).all() for r in rows):
        raise ValueError('Nonfinite user/article features')
    stats={'records':len(rows),'articles':len(ids),'split_counts':[i,j-i,len(rows)-j],
      'label_counts':dict(Counter(str(r[3]) for r in rows)),
      'candidate_size_counts':dict(Counter(str(len(r[4])) for r in rows)),
      'time_span_seconds':int(ts[-1]-ts[0]),'logged_ctr':float(np.mean([r[3] for r in rows])),
      'log_sha256':hashlib.sha256((data/'webscope-logs.txt').read_bytes()).hexdigest(),
      'article_sha256':hashlib.sha256((data/'webscope-articles.txt').read_bytes()).hexdigest()}
    return rows,af,ids,i,j,stats

def context(u,arms,af,hybrid=False):
    """Hybrid returns (6D user context, 36D user x article interaction); else 6D user context.

    Li et al. (2010), Sec. 5.2.2 define the shared feature z_{t,a} as the
    outer product of the six-dimensional user and article features, giving
    36 entries that encode user-article affinity.  The disjoint context x
    keeps only the six user coordinates.
    """
    x=np.broadcast_to(u,(len(arms),6))
    if hybrid:
        z=np.einsum('i,nj->nij',u,af[arms]).reshape(len(arms),36)
        return x,z
    return x

def replay(rows,af,policy,hybrid=False,model=None):
    clicks=matches=0;actions=Counter();matched_actions=Counter();trace=[];bc=bm=0
    for k,(t,u,logged,r,arms) in enumerate(rows,1):
        ctx=context(u,arms,af,hybrid)
        x,z=ctx if hybrid else (ctx,None)
        if model is None:q=policy.choose(arms,x,z) if hybrid else policy.choose(arms,x)
        else:q=int(np.argmax(model.predict_proba(features(u,af[arms]))[:,1]))
        a=int(arms[q]);actions[a]+=1
        if a==logged:
            matches+=1;clicks+=r;bm+=1;bc+=r;matched_actions[a]+=1
            if model is None:policy.update(a,x[q],z[q],r) if hybrid else policy.update(a,x[q],r)
        if k%1000==0 or k==len(rows):
            trace.append({'events':k,'matched':matches,'clicks':clicks,'ctr':clicks/matches if matches else 0,
                          'block_ctr':bc/bm if bm else None,'block_matched':bm});bc=bm=0
    return {'processed':len(rows),'matched':matches,'clicks':clicks,
      'replay_ctr':clicks/matches if matches else 0,'match_rate':matches/len(rows),
      'trace':trace,'action_counts':dict(actions),'matched_actions':dict(matched_actions)}

def features(u,a):
    u=np.broadcast_to(u,(len(a),6)) if np.ndim(u)==1 else u
    return np.c_[u,a,(u[:,:,None]*a[:,None,:]).reshape(len(a),36)]

def metrics(y,p,threshold):
    pred=p>=threshold
    return {'average_precision':float(average_precision_score(y,p)),
      'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
      'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(brier_score_loss(y,p)),
      'precision':float(precision_score(y,pred,zero_division=0)),
      'recall':float(recall_score(y,pred,zero_division=0)),
      'f1':float(f1_score(y,pred,zero_division=0)),
      'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def supervised(rows,af,i,j,out):
    u=np.array([r[1] for r in rows]);a=af[[r[2] for r in rows]]
    X=features(u,a);y=np.array([r[3] for r in rows]);grid=[];best=None
    for c in [.001,.01,.1,1.,10.]:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,random_state=2026))
        m.fit(X[:i],y[:i]);p=m.predict_proba(X[i:j])[:,1];ap=average_precision_score(y[i:j],p)
        grid.append({'C':c,'validation_average_precision':float(ap)})
        if best is None or ap>best[0]:best=(ap,m,c,p)
    _,m,c,p=best;precision,recall,thresholds=precision_recall_curve(y[i:j],p)
    f=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-15)
    threshold=float(thresholds[np.argmax(f)])
    joblib.dump(m,out/'logistic_model.joblib')
    np.savez_compressed(out/'test_predictions.npz',y=y[j:],probability=m.predict_proba(X[j:])[:,1])
    return m,{'C':c,'threshold_validation_f1':threshold,'grid':grid,
      'validation':metrics(y[i:j],p,threshold),'test':metrics(y[j:],m.predict_proba(X[j:])[:,1],threshold),
      'test_click_prevalence':float(y[j:].mean()),'features':'user6 + article6 + outer36; no history leakage',
      'test_protocol':'frozen training-only model; validation selects C and threshold'}

def synthetic(out,seeds=10,horizon=3000):
    allrows=[]
    for scenario in ['stationary','abrupt','gradual']:
        for seed in range(seeds):
            rng=np.random.default_rng(seed);x=rng.normal(size=(horizon,6,6));x/=np.maximum(np.linalg.norm(x,axis=2,keepdims=True),1e-8)
            theta=rng.normal(size=(6,6));theta/=np.linalg.norm(theta,axis=1,keepdims=True)
            other=-theta;weight=np.zeros(horizon)
            if scenario=='abrupt':weight[horizon//2:]=1
            if scenario=='gradual':weight=np.clip((np.arange(horizon)-horizon//3)/(horizon/3),0,1)
            mu=.5+.35*np.einsum('tkd,tkd->tk',x,(1-weight[:,None,None])*theta+weight[:,None,None]*other)
            uniforms=rng.random((horizon,6));rewards=uniforms<mu
            for name,p in [('Random',LinUCBDisjoint(6,6,random=True,seed=seed)),
                ('LinUCB (disjoint)',LinUCBDisjoint(6,6,alpha=.25,seed=seed)),
                ('Window LinUCB (300)',LinUCBDisjoint(6,6,alpha=.25,seed=seed,window=300))]:
                regret=clicks=post=0
                for t in range(horizon):
                    a=p.choose(np.arange(6),x[t]);r=int(rewards[t,a]);p.update(a,x[t,a],r)
                    regret+=float(mu[t].max()-mu[t,a]);clicks+=r
                    if t>=horizon//2:post+=r
                allrows.append({'scenario':scenario,'seed':seed,'policy':name,'ctr':clicks/horizon,
                    'dynamic_regret':regret,'post_half_ctr':post/(horizon-horizon//2)})
    # Static-policy OPE sanity study: known uniform propensities, independent q-model.
    rng=np.random.default_rng(981);n=20000;k=4;X=rng.normal(size=(n,k,3));theta=np.array([.3,-.2,.1])
    mu=np.clip(.5+X@theta*.1,.05,.95);target=np.argmax(X@theta,axis=1)
    logged=rng.integers(k,size=n);reward=rng.random(n)<mu[np.arange(n),logged]
    w=(logged==target)*k; q=np.full((n,k),.5)
    ope={'source':'synthetic randomized logger only; fixed target; q=0.5 fixed independently',
      'known_propensity':1/k,'oracle_mean':float(mu[np.arange(n),target].mean()),
      'IPS':float(np.mean(w*reward)),'SNIPS':float(np.sum(w*reward)/w.sum()),
      'DR':float(np.mean(q[np.arange(n),target]+w*(reward-q[np.arange(n),logged])))}
    return allrows,ope

def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plots(result,out):
    tests=result['test'];names=list(tests)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(names,[tests[n]['replay_ctr']*100 for n in names]);ax.set_ylabel('Test replay CTR (%)');ax.set_title('Descriptive replay; adaptive bandits vs frozen LR');ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(out/'ctr_comparison.png',dpi=180);plt.close(fig)
    for metric,label in [('clicks','Cumulative matched clicks (unequal matched exposure)'),('block_ctr','CTR per 1,000 log events')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for n,z in tests.items():ax.plot([v['events'] for v in z['trace']],[v[metric] for v in z['trace']],label=n)
        ax.set(xlabel='Test log events',ylabel=label);ax.legend();fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ['disjoint','hybrid']:
        values=[z for z in result['validation_grid'] if z['method']==method]
        ax.plot([z['value'] for z in values],[z['replay_ctr']*100 for z in values],'-o',label=method)
    ax.set(xlabel='Alpha (validation only)',ylabel='Validation replay CTR (%)',xscale='log');ax.legend();fig.tight_layout();fig.savefig(out/'alpha_search.png',dpi=180);plt.close(fig)
    s=result['simulation'];fig,ax=plt.subplots(figsize=(9,5));locations=np.arange(3)
    for k,name in enumerate(['Random','LinUCB (disjoint)','Window LinUCB (300)']):
        means=[];errs=[]
        for scenario in ['stationary','abrupt','gradual']:
            v=[z['dynamic_regret'] for z in s if z['policy']==name and z['scenario']==scenario]
            means.append(np.mean(v));errs.append(np.std(v,ddof=1)/np.sqrt(len(v))*1.96)
        ax.bar(locations+(k-1)*.24,means,width=.24,yerr=errs,label=name)
    ax.set_xticks(locations,['Stationary','Abrupt','Gradual']);ax.set_ylabel('Dynamic pseudo-regret (mean, 95% seed-mean normal interval)');ax.legend();fig.tight_layout();fig.savefig(out/'drift_regret.png',dpi=180);plt.close(fig)

METHOD_GRIDS=[('disjoint',[.5,1,2,5,10]),('hybrid',[.25,.5,1,2.1,5]),
              ('contextual_greedy',[.05,.1,.2]),('epsilon_greedy',[.05,.1,.2]),
              ('window',[1000,5000,10000])]
DISPLAY_LABELS={'random':'Random','disjoint':'LinUCB (disjoint)','hybrid':'LinUCB (hybrid)',
    'contextual_greedy':'contextual greedy','epsilon_greedy':'ε-greedy (context-free)',
    'window':'Window LinUCB','logistic':'Logistic regression'}

def make_policy(method,value,n,seed=2026):
    """Create a paper-aligned policy for a validation-grid value."""
    if method=='disjoint':return LinUCBDisjoint(n,6,alpha=value,seed=seed)
    if method=='hybrid':return LinUCBHybrid(n,alpha=value,seed=seed)
    if method=='contextual_greedy':return ContextualGreedy(n,epsilon=value,seed=seed)
    if method=='epsilon_greedy':return EpsilonGreedy(n,epsilon=value,seed=seed)
    if method=='window':return LinUCBDisjoint(n,6,alpha=1.,window=value,seed=seed)
    if method=='random':return LinUCBDisjoint(n,6,random=True,seed=seed)
    raise ValueError('Unknown policy method: '+method)

def run(args):
    out=Path(args.output).resolve() if args.output else ROOT/'results';out.mkdir(exist_ok=True,parents=True)
    rows,af,ids,i,j,stats=load_data();print('Data:',stats,flush=True)
    grid=[];chosen={};test={}
    for method,values in METHOD_GRIDS:
        best=None
        for value in values:
            p=make_policy(method,value,len(af))
            replay(rows[:i],af,p,method=='hybrid');z=replay(rows[i:j],af,p,method=='hybrid')
            row={'method':method,'value':value,'matched':z['matched'],'clicks':z['clicks'],'replay_ctr':z['replay_ctr']};grid.append(row)
            print('validation',row,flush=True)
            if best is None or z['replay_ctr']>best[0]:best=(z['replay_ctr'],value)
        chosen[method]=best[1]
    for method,value in chosen.items():
        p=make_policy(method,value,len(af))
        replay(rows[:j],af,p,method=='hybrid');test[method]=replay(rows[j:],af,p,method=='hybrid')
        joblib.dump(p,out/(method+'_policy.joblib'))
    seed_summaries={};seed_runs={}
    for method in ['random','contextual_greedy','epsilon_greedy']:
        value=0 if method=='random' else chosen[method];runs=[]
        for seed in range(2026,2056):
            p=make_policy(method,value,len(af),seed=seed)
            replay(rows[:j],af,p);z=replay(rows[j:],af,p)
            runs.append({'seed':seed,'matched':z['matched'],'clicks':z['clicks'],'replay_ctr':z['replay_ctr']})
            if seed==2026:test[method]=z
        v=np.array([z['replay_ctr'] for z in runs])
        seed_runs[method]=runs
        seed_summaries[method]={'n':len(runs),'seed_range':[2026,2055],'mean':float(v.mean()),
            'sd':float(v.std(ddof=1)),'interpretation':'algorithm randomness on one fixed log; not population uncertainty'}
    print('Training supervised baseline...',flush=True)
    model,supervision=supervised(rows,af,i,j,out);test['logistic']=replay(rows[j:],af,None,model=model)
    print('Running controlled drift and OPE experiments...',flush=True)
    sim,ope=synthetic(out,args.sim_seeds,args.sim_horizon)
    result={'data':stats,'selected_validation_parameters':chosen,'validation_grid':grid,'test':test,
      'display_labels':DISPLAY_LABELS,'seed_summaries':seed_summaries,
      'random_test_seeds':seed_runs['random'],'contextual_greedy_test_seeds':seed_runs['contextual_greedy'],
      'epsilon_greedy_test_seeds':seed_runs['epsilon_greedy'],
      'random_seed_summary':seed_summaries['random'],
      'supervised':supervision,'simulation':sim,'synthetic_ope':ope,
      'protocol':'Timestamp-boundary 70/15/15. Adaptive bandits: warm up before test, then matched-only updates during test. Frozen LR trained on development only. No Yahoo IPS/DR.',
      'environment':{'python':platform.python_version(),'numpy':np.__version__}}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    csvwrite(out/'validation_grid.csv',grid);csvwrite(out/'random_seeds.csv',seed_runs['random'])
    csvwrite(out/'contextual_greedy_seeds.csv',seed_runs['contextual_greedy'])
    csvwrite(out/'epsilon_greedy_seeds.csv',seed_runs['epsilon_greedy']);csvwrite(out/'simulation.csv',sim)
    for name,z in test.items():
        csvwrite(out/(name+'_trace.csv'),z['trace']);csvwrite(out/(name+'_actions.csv'),[{'article_id':ids[int(a)],'selected':n} for a,n in z['action_counts'].items()])
    plots(result,out)
    lines=['# Completed experiment results','',result['protocol'],'','## Held-out replay','',
      '|Policy|Matched|Clicks|Replay CTR|','|---|---:|---:|---:|']
    lines += [f"|{DISPLAY_LABELS[n]}|{z['matched']}|{z['clicks']}|{z['replay_ctr']:.3%}|" for n,z in test.items()]
    lines += ['','## Thirty algorithm seeds on the fixed log']+[
        f"- {DISPLAY_LABELS[k]}: mean {v['mean']:.3%}, SD {v['sd']*100:.3f} pp"
        for k,v in seed_summaries.items()]
    lines+=['','## Logistic regression',json.dumps(supervision,indent=2),'','## Synthetic known-propensity OPE',json.dumps(ope,indent=2),
      '', '## Interpretation limits','Replay is descriptive until logger/preprocessing assumptions are verified. Different policies accept different events. Random-seed variation is not population uncertainty. Synthetic drift does not establish real-user improvement. No unobserved candidate labels or test thresholds were used. Average precision (AP) is reported rather than trapezoidal PR-AUC. Simulation settings were fixed, not optimized on Yahoo test results.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('DONE:',out,flush=True)

if __name__=='__main__':
    # Keep policy checkpoints importable outside a direct script launch.
    sys.modules['experiments']=sys.modules[__name__]
    for cls in (LinUCBDisjoint,LinUCBHybrid,ContextualGreedy,EpsilonGreedy):
        cls.__module__='experiments'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--sim-seeds',type=int,default=10);p.add_argument('--sim-horizon',type=int,default=3000)
    run(p.parse_args())
