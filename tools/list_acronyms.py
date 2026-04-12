#!/usr/bin/env python3
"""List acronyms and acronym-like tokens across the repo.

Usage:
    python tools/list_acronyms.py > acronyms_report.txt
"""
import os
import re
import sys
from collections import defaultdict

EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', 'db/chroma', 'frontend/.next', '.venv', 'venv'}
INCLUDE_EXTS = {
    '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.ini', '.cfg', '.ts', '.tsx', '.js', '.jsx', '.html', '.htm', '.css', '.env', '.sql', '.mdx', '.rst', '.ipynb'
}

# Patterns to match Acronyms and MixedCase tokens
PATTERNS = [
    re.compile(r"\b[A-Z0-9_]{2,}\b"),                    # ALL_UPPER (incl underscores and digits)
    re.compile(r"\bAcronym:[ ]*([A-Z0-9_]{2,})\b"),       # 'Acronym: ABC' typed patterns
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b"),  # MixedCamelCase (e.g., FinBERT, RoBERTa, LangGraph)
    re.compile(r"\bS&P[0-9]{0,3}\b"),                    # S&P, S&P500
    re.compile(r"\bSP[0-9]{2,}\b"),                      # SP500, SP5000
]


def is_text_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in INCLUDE_EXTS


def walk_repo(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip excludes
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            yield fpath


def collect_acronyms(root):
    acronyms = defaultdict(lambda: defaultdict(int))  # acr -> {file:count}
    for f in walk_repo(root):
        if not is_text_file(f):
            continue
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
        except Exception:
            continue

        matches = set()
        for p in PATTERNS:
            for m in p.findall(text):
                # Some patterns use groups
                token = m if isinstance(m, str) else ''.join(m) if isinstance(m, tuple) else m
                # Make sure token length > 1 and not purely numeric
                token = token.strip()
                if len(token) < 2:
                    continue
                if token.isdigit():
                    continue
                # Exclude timestamps like YYYY-MM-DD if matched by uppercase pattern
                if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
                    continue
                matches.add(token)

        for token in matches:
            acronyms[token][f] += text.count(token)

    return acronyms


def print_report(acrs):
    unique = sorted(acrs.keys(), key=lambda s: s.lower())
    print('# Unique Acronym Tokens found: ', len(unique))
    print()
    for acr in unique:
        # Print only token and file list summary
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
