import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else '.'
for dirpath, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '_audit')]
    for f in sorted(files):
        if f.endswith(('.py', '.txt', '.md', '.yml', '.example')) or f == 'Dockerfile':
            p = os.path.join(dirpath, f)
            try:
                n = sum(1 for _ in open(p, encoding='utf-8', errors='replace'))
            except OSError:
                n = -1
            print(f'{n:6d}  {p}')