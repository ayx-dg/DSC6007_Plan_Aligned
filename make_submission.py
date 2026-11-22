"""Assemble course-submission-named files and the required Group 14 zip."""
from pathlib import Path
import shutil,zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'submission';OUT.mkdir(exist_ok=True)
CODE=[Path('reproduction/experiments.py'),Path('reproduction/test_experiments.py'),
      Path('reproduction/Yahoo_recommendation/runner.py'),Path('extend_and_audit.py'),Path('sparsity/run_sparsity.py'),
      Path('make_figures.py'),Path('verify_consistency.py'),Path('run.sh'),Path('build_report.sh')]

def main():
    shutil.copy2(ROOT/'report.pdf',OUT/'Group 14 report.pdf')
    shutil.copy2(ROOT/'slides.pdf',OUT/'Group 14 slides.pdf')
    for src in CODE:
        p=ROOT/src
        if p.exists():
            shutil.copy2(p,OUT/('Group 14 c '+src.name+'.txt'))
    zpath=OUT/'Group 14.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.name=='Group 14.zip':continue
            z.write(p,p.name)
    print('Submission package:',zpath)

if __name__=='__main__':main()
