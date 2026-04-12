#!/usr/bin/env python3
"""Extract tokens from acronyms_curated_report.txt and print deduplicated token list.
"""
import re

infile = 'acronyms_curated_report.txt'

with open(infile, 'r', encoding='utf-8', errors='ignore') as fh:
    content = fh.read()

tokens = []
count_lines = 0
first_line = None
for i, line in enumerate(content.splitlines()):
    if 'file(s)' in line:
        count_lines += 1
        if first_line is None:
            first_line = line

print(f'Found {count_lines} lines with file(s). First example: {repr(first_line)}')

for line in content.splitlines():
    # token lines are top-level (not indented) and contain 'file(s)'
    if 'file(s)' in line and line.strip() and not line.startswith(' '):
        tok = line.strip().split()[0]
        tokens.append(tok)

# Dump tokens
with open('acronyms_curated_tokens.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(sorted(tokens)))

print(f'Wrote {len(tokens)} tokens to acronyms_curated_tokens.txt')
