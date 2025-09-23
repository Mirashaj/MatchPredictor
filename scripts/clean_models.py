"""Clean up noisy backup files in the models/ folder.

Usage examples:
  python scripts/clean_models.py --dry-run
  python scripts/clean_models.py --archive --out models/archive
  python scripts/clean_models.py --delete --force

Modes:
  --dry-run   : list matching files without changing anything
  --archive   : move matches to archive folder (default: models/archive)
  --delete    : permanently delete matches (requires --force)

By default this looks for patterns like:
  - elo_matches_YYYYMMDDTHHMMSSZ.csv
  - elo_matches*.csv (timestamped)
  - elo_ratings_*.json
  - files with repeated duplicates eliding the canonical names

Script is defensive: it will never delete files unless --delete and --force are provided.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

PATTERNS = [
    r"elo_matches_\d{8}T\d{6}Z\.csv$",
    r"elo_matches_.*\.csv$",
    r"elo_ratings_.*\.json$",
    r"elo_\d{4}_matches\.csv$",
    r"elo_\d{4}_ratings\.json$",
]


def find_candidates(models_dir: Path):
    candidates = []
    for p in models_dir.iterdir():
        if not p.is_file():
            continue
        for pat in PATTERNS:
            if re.search(pat, p.name):
                candidates.append(p)
                break
    # exclude canonical stable filenames
    canonical = {"elo_matches.csv", "elo_ratings.json", "elo_2025_matches.csv", "elo_2025_ratings.json",
                 "ensemble_with_elo.joblib", "feature_encoders_with_elo.joblib", "scaler_with_elo.joblib",
                 "ensemble_model.joblib", "feature_encoders.joblib", "scaler.joblib"}
    filtered = [p for p in candidates if p.name not in canonical]
    return sorted(filtered)


def archive_files(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in files:
        dest = out_dir / p.name
        shutil.move(str(p), str(dest))
        moved.append(dest)
    return moved


def delete_files(files):
    removed = []
    for p in files:
        p.unlink()
        removed.append(p)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default='models')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--archive', action='store_true')
    ap.add_argument('--out', default=None, help='Archive destination (used with --archive)')
    ap.add_argument('--delete', action='store_true')
    ap.add_argument('--force', action='store_true', help='Require to actually delete files')
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        logging.error('models directory not found: %s', models_dir)
        return 1

    candidates = find_candidates(models_dir)
    if not candidates:
        logging.info('No candidate backup files found in %s', models_dir)
        return 0

    logging.info('Found %d candidate files:', len(candidates))
    for p in candidates:
        logging.info('  %s', p.name)

    if args.dry_run:
        logging.info('Dry run; no changes made')
        return 0

    if args.archive:
        out_dir = Path(args.out) if args.out else models_dir / 'archive'
        moved = archive_files(candidates, out_dir)
        logging.info('Archived %d files to %s', len(moved), out_dir)
        return 0

    if args.delete:
        if not args.force:
            logging.error('Refusing to delete files without --force')
            return 2
        removed = delete_files(candidates)
        logging.info('Deleted %d files', len(removed))
        return 0

    logging.info('No action requested. Use --dry-run, --archive, or --delete --force')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
