"""Sparse-data diagnostic for the paper-aligned Hybrid LinUCB reproduction.

The paper reports that the benefit of contextual (especially hybrid) models is
concentrated in sparse learning buckets and with a large article pool.  This
diagnostic keeps the Yahoo test split fixed and varies only the amount of
development+validation data used to warm up the policies.  A second condition
enlarges each logged candidate set to the 50 most frequent articles; that
condition is explicitly a sensitivity diagnostic because it changes the logged
candidate distribution.

Run through ./run.sh after core/results.json exists.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import csv,hashlib,json,platform,sys,time
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'reproduction'))
import experiments as E

OUT=ROOT/'sparsity'; OUT.mkdir(exist_ok=True)
FRACTIONS=[1.0,0.10,0.05,0.01]
POOL_SIZE=50
SEED=2026
POLICIES=['disjoint','hybrid','contextual_greedy','epsilon_greedy']

def dumpcsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def warmup_rows(rows,i,j,fraction,seed=SEED):
    """Uniform chronological subsample of development+validation, kept in time order."""
    if fraction>=1.0:return rows[:j],j
    k=max(1,int(round(fraction*j)))
    chosen=np.sort(np.random.default_rng(seed).choice(j,size=k,replace=False))
    return [rows[int(t)] for t in chosen],k

def enlarge_candidates(rows,n,top_ids,size=POOL_SIZE):
    """Add frequent articles to each logged candidate set, preserving order.

    The original logger's candidate order is kept so that the logged action
    keeps its original random position.  Sorting the union would place the
    smallest (never-logged) article first and make deterministic UCB policies
    degenerate: with all arms tied they always choose index zero and can never
    observe a match.  Appending the extra articles avoids that artefact.
    """
    out=[]
    for t,u,logged,r,arms in rows:
        chosen=[int(a) for a in arms]
        if int(logged) not in chosen:chosen.append(int(logged))
        for a in top_ids:
            if len(chosen)>=size:break
            if int(a) not in chosen:chosen.append(int(a))
        for a in range(n):
            if len(chosen)>=size:break
            if int(a) not in chosen:chosen.append(int(a))
        out.append((t,u,logged,r,np.array(chosen,dtype=int)))
    return out

def main():
    start=time.time()
    core=json.loads((ROOT/'core/results.json').read_text(encoding='utf-8'))
    chosen=core['selected_validation_parameters']
    params={p:(0 if p=='random' else chosen[p]) for p in POLICIES}
    rows,af,ids,i,j,stats=E.load_data()
    counts=Counter(int(r[2]) for r in rows)
    top_ids=[a for a,_ in counts.most_common(POOL_SIZE)]
    results=[]
    for fraction in FRACTIONS:
        warm,k=warmup_rows(rows,i,j,fraction)
        for pool in ['native','top50']:
            test_rows=rows[j:] if pool=='native' else enlarge_candidates(rows[j:],len(af),top_ids)
            warm_eval_rows=warm if pool=='native' else enlarge_candidates(warm,len(af),top_ids)
            for method in POLICIES:
                p=E.make_policy(method,params[method],len(af),seed=SEED)
                w=E.replay(warm_eval_rows,af,p,method=='hybrid')
                z=E.replay(test_rows,af,p,method=='hybrid')
                upd=np.array(list(w['matched_actions'].values()),dtype=int)
                results.append({'fraction':fraction,'pool':pool,'policy':method,
                    'label':E.DISPLAY_LABELS[method],'parameter':params[method],
                    'warmup_events':k,'warmup_matches':w['matched'],'warmup_clicks':w['clicks'],
                    'arms_with_updates':int(len(upd)),
                    'min_updates_per_updated_arm':int(upd.min()) if len(upd) else 0,
                    'test_events':len(test_rows),'matched':z['matched'],'clicks':z['clicks'],
                    'replay_ctr':z['replay_ctr'],'match_rate':z['match_rate']})
            print('sparsity',fraction,pool,'done',flush=True)
    dumpcsv(OUT/'sparsity_results.csv',results)
    # Paired diagnostic: hybrid minus disjoint in matched-replay CTR percentage points.
    index={(r['pool'],r['fraction'],r['policy']):r for r in results}
    contrasts=[]
    for pool in ['native','top50']:
        for fraction in FRACTIONS:
            h=index[pool,fraction,'hybrid']['replay_ctr']
            d=index[pool,fraction,'disjoint']['replay_ctr']
            contrasts.append({'pool':pool,'fraction':fraction,'hybrid_minus_disjoint_pp':(h-d)*100})
    dumpcsv(OUT/'hybrid_contrasts.csv',contrasts)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(12.6,3.8),constrained_layout=True)
    colors={'disjoint':'#2b6f8a','hybrid':'#b55236','contextual_greedy':'#7b8c5a','epsilon_greedy':'#8a6f9b'}
    labels={'disjoint':'LinUCB (disjoint)','hybrid':'LinUCB (hybrid)',
            'contextual_greedy':'contextual greedy','epsilon_greedy':'ε-greedy (context-free)'}
    for panel,pool in enumerate(['native','top50']):
        for method in POLICIES:
            r=[index[pool,f,method] for f in FRACTIONS]
            y=[(q['replay_ctr']*100 if q['matched'] else float('nan')) for q in r]
            ax[panel].plot([q['fraction']*100 for q in r],y,
                           '-o',label=labels[method],color=colors[method])
        if pool=='top50':
            ax[panel].text(.98,.05,'LinUCB: 0 matches\n(CTR undefined)',transform=ax[panel].transAxes,
                           ha='right',va='bottom',fontsize=8,color='#555555')
        ax[panel].set_xscale('log');ax[panel].set_xticks([f*100 for f in FRACTIONS])
        ax[panel].set_xticklabels(['100','10','5','1'])
        ax[panel].set(xlabel='Warmup data (% of development+validation)',ylabel='Matched replay CTR (%)',
                      title='Native 20–21 candidates' if pool=='native' else 'Enlarged 50-article pool')
        ax[panel].grid(alpha=.2)
    ax[1].legend(fontsize=8)
    r=[index['native',f,'disjoint'] for f in FRACTIONS]
    ax[2].plot([q['fraction']*100 for q in r],[q['arms_with_updates'] for q in r],'-o',color='#2b6f8a',label='disjoint updates')
    r=[index['native',f,'hybrid'] for f in FRACTIONS]
    ax[2].plot([q['fraction']*100 for q in r],[q['arms_with_updates'] for q in r],'-o',color='#b55236',label='hybrid updates')
    ax[2].set_xscale('log');ax[2].set_xticks([f*100 for f in FRACTIONS]);ax[2].set_xticklabels(['100','10','5','1'])
    ax[2].set(xlabel='Warmup data (%)',ylabel='Articles with ≥1 matched update',title='Warmup coverage');ax[2].grid(alpha=.2);ax[2].legend(fontsize=8)
    fig.savefig(OUT/'sparsity.pdf');fig.savefig(OUT/'sparsity.png',dpi=180);plt.close(fig)
    manifest={'python':platform.python_version(),'numpy':np.__version__,
        'matplotlib':matplotlib.__version__,'seed':SEED,'fractions':FRACTIONS,
        'pool_size':POOL_SIZE,'policies':POLICIES,'selected_parameters':params,
        'experiments_sha256':hashlib.sha256((ROOT/'reproduction/experiments.py').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'log_sha256':stats['log_sha256'],'runtime_seconds':time.time()-start,
        'note':'Fixed chronological test; only warmup size and candidate-pool construction vary. Enlarged pool changes the logged candidate distribution and is descriptive.'}
    payload={'manifest':manifest,'results':results,'contrasts':contrasts}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('Sparsity diagnostic complete',manifest['runtime_seconds'],flush=True)

if __name__=='__main__':main()
