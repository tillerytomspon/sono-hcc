from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from smc_lud import index_paths, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit-smc')
    audit.add_argument('root', type=Path)
    audit.add_argument('--subset', choices=('all', 'clean'), required=True)
    audit.add_argument('--patients', type=Path)
    build = commands.add_parser('prepare-smc')
    build.add_argument('root', type=Path)
    build.add_argument('--subset', choices=('all', 'clean'), required=True)
    build.add_argument('--profiles', type=Path, required=True)
    build.add_argument('--patient-split', type=Path)
    build.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'audit-smc':
        paths = [p.relative_to(args.root).as_posix() for p in args.root.rglob('*') if p.is_file()]
        records, summary = index_paths(paths, args.subset)
        if args.patients:
            counts = Counter((row['patient'], row['label']) for row in records)
            with args.patients.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(('patient', 'label', 'images', 'split'))
                writer.writerows((patient, label, count, '') for (patient, label), count in sorted(counts.items()))
        print(json.dumps(summary, sort_keys=True))
    else:
        summary = prepare(args.root, args.subset, args.profiles, args.output, args.patient_split)
        print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
