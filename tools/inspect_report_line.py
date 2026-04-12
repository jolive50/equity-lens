#!/usr/bin/env python3
with open('acronyms_curated_report.txt','r', encoding='utf-8', errors='ignore') as fh:
    for line in fh:
        line = line.rstrip('\n')
        if line.strip() == '' or line.startswith('#'):
            continue
        print('LINE_REPR:', repr(line))
        for i, ch in enumerate(line):
            print(i, ch, hex(ord(ch)))
        break
