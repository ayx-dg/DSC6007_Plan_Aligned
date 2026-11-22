"""Consistency audit: results, sources, report, PDF, review and deliverables.

Called at the end of ./run.sh and directly when the report changes.
"""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parent
errors=[];notes=[]

def check(condition,message):
    if not condition:errors.append(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

# 1. Executable correctness tests.
res=subprocess.run([sys.executable,str(ROOT/'reproduction/test_experiments.py')],
                   cwd=ROOT/'reproduction',capture_output=True,text=True)
check(res.returncode==0,'Correctness tests failed: '+res.stderr[-500:])
m=re.search(r'Ran (\d+) tests',res.stderr)
ntests=int(m.group(1)) if m else 0
check(ntests>=14,f'Expected at least 14 correctness tests, found {ntests}')
notes.append(f'correctness_tests={ntests}')

# 2. Required artifacts and JSON schemas.
required=['run.sh','build_report.sh','report.tex','report.pdf','slides.tex','slides.pdf',
          'AGENTS.md','project_desc.pdf','review_paper_judge.json','paper_judge_review.md',
          'core/results.json','robustness/results.json','sparsity/results.json',
          'reproduction/experiments.py','references/li2010_contextual_bandit_news.pdf',
          'references/li2011_unbiased_offline_eval.pdf']
for rel in required:check((ROOT/rel).exists(),'Missing '+rel)
if any((ROOT/r).exists() is False for r in required):
    print('\n'.join(errors));sys.exit(1)

core=json.loads((ROOT/'core/results.json').read_text())
rob=json.loads((ROOT/'robustness/results.json').read_text())
spa=json.loads((ROOT/'sparsity/results.json').read_text())
review=json.loads((ROOT/'review_paper_judge.json').read_text())

for key in ['random','disjoint','hybrid','contextual_greedy','epsilon_greedy','window','logistic']:
    check(key in core['test'],'core test missing '+key)
for key in ['random','contextual_greedy','epsilon_greedy']:
    check(len(core[key+'_test_seeds'])==30,'30-seed sweep missing for '+key)
check(len(core['simulation'])==90,'Poster simulation must have 3 policies x 3 scenarios x 10 seeds')
check(rob['manifest']['source_sha256']==sha(ROOT/'extend_and_audit.py'),'Extension source hash mismatch')
check(rob['manifest']['experiments_sha256']==sha(ROOT/'reproduction/experiments.py'),'Experiments hash mismatch in robustness')
check(spa['manifest']['experiments_sha256']==sha(ROOT/'reproduction/experiments.py'),'Experiments hash mismatch in sparsity')
check(spa['manifest']['source_sha256']==sha(ROOT/'sparsity/run_sparsity.py'),'Sparsity source hash mismatch')
check(len(spa['results'])==32,'Sparsity diagnostic must have 4 fractions x 2 pools x 4 policies')
check(core['data']['split_counts']==[66204,18010,15786],'Chronological split counts changed')
check(core['data']['log_sha256']=='45a5ce2a1f64875389d5a5bcc65be8e95c3b6a91b53757e90b055da49915525f','Log hash mismatch')

# 3. Report identity and numbers.
tex=(ROOT/'report.tex').read_text()
identity=[r'Contextual Bandits for News Recommendation\\under Changing User Preferences',
          'Chronological Yahoo Replay and Controlled Drift Experiments',
          'DSC6007 Dynamic Programming and Reinforcement Learning --- Group 14',
          r'LI Yizhuo \quad ZHANG Tianyi \quad GOU Zhijie \quad AN Yuxian \quad XIAO Yuhang',
          'Semester A 2026']
for s in identity:check(s in tex,'Report identity string missing: '+s)
for n,z in core['test'].items():
    check(f"{z['matched']:,}" in tex or str(z['matched']) in tex,f'Report missing matched count for {n}')
    check(f"{z['clicks']}" in tex,f'Report missing clicks for {n}')
    check(f"{z['replay_ctr']*100:.3f}" in tex,f'Report missing replay CTR for {n}')
for h in [core['data']['log_sha256'],core['data']['article_sha256'],
          sha(ROOT/'reproduction/experiments.py'),sha(ROOT/'extend_and_audit.py')]:
    check(h in tex,'Report missing current source/data hash: '+h[:12])
check('⊮' not in tex,'Report contains undefined glyph')
check('\\mathbb' not in tex and 'amssymb' not in tex,'Report uses unsupported math package/glyph')
review_ids={c.get('id') for c in review.get('criteria',[])}
for key in ['C1','C2','C3','C4','C5']:
    check(key in review_ids,'paper-judge review missing '+key)

# 4. PDF pages and warnings.
if shutil.which('pdfinfo'):
    info=subprocess.run(['pdfinfo',str(ROOT/'report.pdf')],capture_output=True,text=True).stdout
    pm=re.search(r'Pages:\s+(\d+)',info)
    pages=int(pm.group(1)) if pm else 99
else:
    raw=(ROOT/'report.pdf').read_bytes();pages=len(re.findall(rb'/Type\s*/Page\b',raw))
check(pages<=15,f'Report has {pages} pages (>15)')
notes.append(f'report_pages={pages}')
log=Path('/tmp/hybrid_report_pass2.log')
if log.exists():
    txt=log.read_text(errors='ignore')
    for token in ['Overfull \\hbox','Undefined control sequence','LaTeX Warning: Citation']:
        if token in txt:errors.append('LaTeX warning: '+token)

review_hash=review.get('reviewed_pdf_sha256')
if review_hash!=sha(ROOT/'report.pdf'):
    notes.append('review_pdf_hash_stale_or_unset: refresh paper-judge after report edits')
else:notes.append('review_pdf_hash_matches=true')

if errors:
    print('CONSISTENCY CHECK FAILED')
    for e in errors:print(' -',e)
    sys.exit(1)
print('CONSISTENCY CHECK PASSED',' '.join(notes))
