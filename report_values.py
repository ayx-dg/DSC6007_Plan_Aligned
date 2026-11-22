"""Print machine-generated numbers needed in report.tex and slides.tex."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent
c=json.loads((R/'core/results.json').read_text())
r=json.loads((R/'robustness/results.json').read_text())
s=json.loads((R/'sparsity/results.json').read_text())
print('SELECTED',c['selected_validation_parameters'])
for n,z in c['test'].items():
    print(f"TEST {n:18s} matched={z['matched']} clicks={z['clicks']} ctr={z['replay_ctr']*100:.3f}")
for k,v in c['seed_summaries'].items():
    print(f"SEED {k:18s} mean={v['mean']*100:.3f} sd={v['sd']*100:.3f}")
print('SUPERVISED', {k:v for k,v in c['supervised'].items() if k not in ['grid','validation','test']})
print('SIM',[(x['policy'],x['scenario'],round(x['dynamic_regret'],2)) for x in c['simulation']][:3])
print('SPARSITY')
for x in s['results']:
    print(f"  {x['pool']:6s} {x['fraction']:4.2f} {x['policy']:18s} ctr={x['replay_ctr']*100:.3f} matched={x['matched']} clicks={x['clicks']} warm={x['warmup_matches']} arms={x['arms_with_updates']}")
