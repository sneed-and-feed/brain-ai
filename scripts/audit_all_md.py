import os
from audit_gfm_math import audit_file

files = ['README.md', 'docs/ARCHITECTURE_SPEC.md', 'docs/RESEARCH_PLAN.md', 'docs/preprint/README.md']
all_pass = True
for f in files:
    if os.path.exists(f):
        res = audit_file(f)
        if res != 0:
            all_pass = False

if all_pass:
    print("\nALL MARKDOWN FILES ARE 100% KATEX / GFM COMPLIANT!")
else:
    print("\nSOME FILES FAILED COMPLIANCE!")
    exit(1)
