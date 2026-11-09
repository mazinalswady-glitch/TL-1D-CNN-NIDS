"""Check the bundled five-seed results and output inventory."""
import hashlib,json
from pathlib import Path

root=Path(__file__).resolve().parent
datasets=('results.json','extended/reverse.json','extended/labels.json','extended/open.json','extended/cic.json','extended/leakage.json')
for name in datasets:
    d=json.loads((root/name).read_text())
    assert [r['seed'] for r in d['runs']]==list(range(5)),name
for line in (root/'SHA256SUMS.txt').read_text().splitlines():
    expected,name=line.split('  ',1)
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print('Verified six experiment records with five seeds each and all manifest hashes.')
