#!/usr/bin/env python3
"""Curated list of acronyms across the repo.

This script focuses on tokens in source, docs, and config files and avoids build artifacts and saved model ids.

Usage:
    python tools/list_acronyms_curated.py > acronyms_curated_report.txt
"""
import os
import re
import sys
from collections import defaultdict

EXCLUDE_SUBSTRS = ['/.git/', '/node_modules/', '/__pycache__/', '/db/chroma/', '/.venv/', '/venv/', '/frontend/.next/', '/models/prediction/saved_models/']
INCLUDE_EXTS = {
    '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.ini', '.cfg', '.ts', '.tsx', '.js', '.jsx', '.html', '.htm', '.css', '.env', '.sql', '.mdx', '.rst'
}

# Only consider tokens found in these source directories or root files
ALLOWED_PATH_KEYWORDS = ['api/', 'agents/', 'coordinator/', 'data/', 'docs/', 'models/', 'storage/', 'utils/', 'tools/', 'db/', 'tests/', 'launch.py', 'README.md', 'frontend/src/']

PATTERNS = [
    re.compile(r"\b[A-Z]{2,10}\b"),
    re.compile(r"\b[A-Z]+[0-9]{1,4}\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b"),  # camel/pascal with at least 2 caps
    re.compile(r"\bS&P[0-9]{0,3}\b"),
    re.compile(r"\bSP[0-9]{2,}\b"),
]

HEX_RE = re.compile(r"^[0-9A-F]{6,}$")
COMMON_STOPWORDS = set(['AND','THE','FOR','WITH','FROM','IN','TO','BY','OF','IS','ARE','AS','IT','AT','THIS','THAT','A','I','BE','WILL','HAS','HAVE','NOT','OR','IF','ELSE'])


def walk_repo(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            # Exclude by substring
            pathnorm = fpath.replace('\\', '/')
            if any(s in pathnorm for s in EXCLUDE_SUBSTRS):
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in INCLUDE_EXTS:
                continue
            yield fpath


def token_in_allowed_path(files):
    # Return True if at least one file is in an allowed path
    for f in files:
        pn = f.replace('\\', '/')
        if any(k in pn for k in ALLOWED_PATH_KEYWORDS):
            return True
        # also allow top-level files
        if os.path.basename(pn) in ['README.md', 'launch.py', 'requirements.txt', 'config.yaml']:
            return True
    return False


def looks_noisy(token, files):
    if len(token) > 12:
        return True
    if token.upper() in COMMON_STOPWORDS:
        return True
    if any(ch.isdigit() for ch in token) and sum(ch.isdigit() for ch in token) > len(token) / 2:
        return True
    if HEX_RE.match(token):
        return True
    # exclude tokens that only appear in package-lock or build dirs
    if all(('package-lock.json' in f or '/.next/' in f or '/node_modules/' in f or 'vendor-chunks' in f) for f in files):
        return True
    # require at least one result in allowed path
    if not token_in_allowed_path(files):
        return True
    return False


def collect_acronyms(root):
    acronyms = defaultdict(lambda: defaultdict(int))
    for f in walk_repo(root):
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
                # Exclude dates like YYYY-MM-DD
                if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
                    continue
                matches.add(token)

        for token in matches:
            acronyms[token][f] += text.count(token)

    cleaned = {}
    for token, files in acronyms.items():
        if looks_noisy(token, list(files.keys())):
            continue
        cleaned[token] = files
    return cleaned


def print_report(acrs):
    unique = sorted(acrs.keys(), key=lambda s: s.lower())
    print('# Unique Curated Acronym-like Tokens found: ', len(unique))
    print()
    for acr in unique:
        files = list(acrs[acr].keys())
        print(f'{acr}  — {len(files)} file(s)')
        for f in files:
            print(f'  {f}')
        print()


if __name__ == '__main__':
    root = '.'
    if len(sys.argv) > 1:
        root = sys.argv[1]
    acrs = collect_acronyms(root)
    print_report(acrs)
