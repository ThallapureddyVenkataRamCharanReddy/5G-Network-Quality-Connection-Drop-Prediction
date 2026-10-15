import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None

p = Path('5g_network_eda_analysis_2025.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
ns = {'plt': plt}

for i, cell in enumerate(nb['cells'], 1):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if not src.strip():
        continue
    print(f'\n--- Executing cell {i} ---')
    try:
        exec(src, ns)
        print('OK')
    except Exception as e:
        print('ERROR:', type(e).__name__, e)
        raise

print('\nAll code cells executed successfully!')
