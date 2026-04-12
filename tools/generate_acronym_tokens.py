#!/usr/bin/env python3
"""Generate deduplicated tokens using same curated rules and output token-only list.
"""
import os
import re
from collections import defaultdict

EXCLUDE_SUBSTRS = ['/.git/', '/node_modules/', '/__pycache__/', '/db/chroma/', '/.venv/', '/venv/', '/frontend/.next/', '/models/prediction/saved_models/']
INCLUDE_EXTS = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.ini', '.cfg', '.ts', '.tsx', '.js', '.jsx', '.html', '.htm', '.css', '.env', '.sql', '.mdx', '.rst'}
ALLOWED_PATH_KEYWORDS = ['api/', 'agents/', 'coordinator/', 'data/', 'docs/', 'models/', 'storage/', 'utils/', 'tools/', 'db/', 'tests/', 'launch.py', 'README.md', 'frontend/src/']

PATTERNS = [
    re.compile(r"\b[A-Z]{2,10}\b"),
    re.compile(r"\b[A-Z]+[0-9]{1,4}\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b"),
    re.compile(r"\bS&P[0-9]{0,3}\b"),
    re.compile(r"\bSP[0-9]{2,}\b"),
]
HEX_RE = re.compile(r"^[0-9A-F]{6,}$")
COMMON_STOPWORDS = set(['AND','THE','FOR','WITH','FROM','IN','TO','BY','OF','IS','ARE','AS','IT','AT','THIS','THAT','A','I','BE','WILL','HAS','HAVE','NOT','OR','IF','ELSE'])


def walk_repo(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            pathnorm = fpath.replace('\\', '/')
            if any(s in pathnorm for s in EXCLUDE_SUBSTRS):
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in INCLUDE_EXTS:
                continue
            yield fpath


def in_allowed_path(files):
    for f in files:
        pn = f.replace('\\', '/')
        if any(k in pn for k in ALLOWED_PATH_KEYWORDS) or os.path.basename(pn) in ['README.md', 'launch.py', 'requirements.txt', 'config.yaml']:
            return True
    return False


acrs = defaultdict(lambda: defaultdict(int))
for f in walk_repo('.'):
    try:
        with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
            text = fh.read()
    except Exception:
        continue
    matches = set()
    for p in PATTERNS:
        for m in p.findall(text):
            token = m if isinstance(m, str) else ''.join(m) if isinstance(m, tuple) else m
            token = token.strip()
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
                continue
            matches.add(token)
    for token in matches:
        acrs[token][f] += text.count(token)

# filter
cleaned = {}
for token, files in acrs.items():
    # filter noise
    if len(token) > 12:
        continue
    if token.upper() in COMMON_STOPWORDS:
        continue
    if any(ch.isdigit() for ch in token) and sum(ch.isdigit() for ch in token) > len(token) / 2:
        continue
    if HEX_RE.match(token):
        continue
    if all(('package-lock.json' in f or '/.next/' in f or '/node_modules/' in f or 'vendor-chunks' in f) for f in files):
        continue
    if not in_allowed_path(list(files.keys())):
        continue
    cleaned[token] = files

# write tokens
with open('acronyms_curated_tokens.txt', 'w', encoding='utf-8') as out:
    for t in sorted(cleaned.keys(), key=lambda s: s.lower()):
        out.write(t + '\n')

print(f'Wrote {len(cleaned)} tokens to acronyms_curated_tokens.txt')
