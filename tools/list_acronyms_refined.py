#!/usr/bin/env python3
"""Refined list of acronyms and acronym-like tokens across the repo.

Usage:
    python tools/list_acronyms_refined.py > acronyms_refined_report.txt
"""
import os
import re
import sys
from collections import defaultdict

EXCLUDE_DIRS = {'.git', 'node_modules', '__pycache__', 'db/chroma', 'frontend/.next', '.venv', 'venv', 'models/prediction/saved_models'}
INCLUDE_EXTS = {
    '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.ini', '.cfg', '.ts', '.tsx', '.js', '.jsx', '.html', '.htm', '.css', '.env', '.sql', '.mdx', '.rst', '.ipynb'
}

# Patterns to match Acronyms and MixedCase tokens
PATTERNS = [
    re.compile(r"\b[A-Z]{2,10}\b"),  # ALL_UPPER alpha-only tokens of reasonable length (2-10)
    re.compile(r"\b[A-Z]+[0-9]{1,4}\b"),  # Alpha plus numbers like SP500, B2B
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b"),  # MixedCamelCase (FinBERT, RoBERTa)
    re.compile(r"\bS&P[0-9]{0,3}\b"),  # S&P, S&P500
    re.compile(r"\bSP[0-9]{2,}\b"),  # SP500, SP5000
    re.compile(r"\b[A-Z]{2,}-[A-Z]{2,}\b"),  # e.g., XG-BOOST-like tokens w/ hyphen
]

HEX_RE = re.compile(r"^[0-9A-F]{6,}$")  # probable hex-like tokens


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


def looks_noisy(token, files):
    # Filter out tokens that are long or hex-like; or tokens that appear only in saved model jsons
    if len(token) > 12:
        return True
    if any(ch.isdigit() for ch in token) and sum(ch.isdigit() for ch in token) > len(token) / 2:
        # Too many digits -> likely an id
        return True
    if HEX_RE.match(token):
        return True
    # If token seen only inside saved model jsons, exclude
    if all('models/prediction/saved_models' in f.replace('\\', '/') for f in files):
        return True
    return False


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
                token = m if isinstance(m, str) else ''.join(m) if isinstance(m, tuple) else m
                token = token.strip()
                if len(token) < 2:
                    continue
                if token.isdigit():
                    continue
                # Exclude dates like YYYY-MM-DD
                if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
                    continue
                # Exclude short single letters and small words
                if len(token) <= 1:
                    continue
                matches.add(token)

        for token in matches:
            acronyms[token][f] += text.count(token)

    # post-filter
    cleaned = {}
    for token, files in acronyms.items():
        if looks_noisy(token, list(files.keys())):
            continue
        cleaned[token] = files

    return cleaned


def print_report(acrs):
    unique = sorted(acrs.keys(), key=lambda s: s.lower())
    print('# Unique Acronym-like Tokens found: ', len(unique))
    print()
    for acr in unique:
        files = list(acrs[acr].keys())
        if len(files) == 0:
            continue
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
