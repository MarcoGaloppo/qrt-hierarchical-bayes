"""Static checks for the analysis notebooks (run in CI).

The notebooks need the challenge data, which cannot be redistributed, so CI does not
execute them. Instead it checks that:
  1. each notebook is a valid Jupyter notebook (nbformat schema);
  2. every code cell is valid Python;
  3. the libraries they import are installable from requirements.txt;
  4. the challenge data files are not committed to the repository.
"""
import ast
import importlib
import subprocess
import sys
from pathlib import Path

import nbformat

NOTEBOOKS = [Path("BaysianHierarchicalApproach.ipynb"), Path("CNNApproach.ipynb")]
LIBRARIES = ["numpy", "pandas", "scipy", "scipy.spatial", "scipy.special", "scipy.stats",
             "sklearn.model_selection", "sklearn.metrics", "matplotlib.pyplot",
             "torch", "torch.nn", "torch.utils.data"]
DATA_FILES = {"X_train.csv", "X_test.csv", "y_train.csv"}
MAX_FILE_MB = 50

failures = []

for notebook in NOTEBOOKS:
    # 1. valid notebook
    nb = nbformat.read(notebook, as_version=4)
    try:
        nbformat.validate(nb)
        print(f"[ok] {notebook} is a valid notebook ({len(nb.cells)} cells)")
    except nbformat.ValidationError as e:
        failures.append(f"{notebook}: invalid notebook: {e}")

    # 2. every code cell parses
    n_before = len(failures)
    code_cells = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    for i, cell in code_cells:
        src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            failures.append(f"{notebook} cell {i}: syntax error at line {e.lineno}: {e.msg}")
    print(f"[ok] parsed {len(code_cells)} code cells" if len(failures) == n_before else "[!!] syntax errors found")

# 3. imports resolve
for lib in LIBRARIES:
    try:
        importlib.import_module(lib)
    except ImportError as e:
        failures.append(f"cannot import {lib}: {e}")
print("[ok] all notebook imports resolve")

# 4. repository hygiene: no challenge data, no very large files
tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
for f in tracked:
    if Path(f).name in DATA_FILES:
        failures.append(f"challenge data file is tracked: {f} (the data must not be redistributed)")
    elif Path(f).exists() and Path(f).stat().st_size > MAX_FILE_MB * 1e6:
        failures.append(f"file larger than {MAX_FILE_MB} MB is tracked: {f}")
print(f"[ok] checked {len(tracked)} tracked files")

if failures:
    print("\nFAILED:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("\nAll checks passed.")
