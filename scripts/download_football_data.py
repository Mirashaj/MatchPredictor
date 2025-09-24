#!/usr/bin/env python3
"""Download CSV files linked from a football-data.co.uk page.

Usage (from repo root):
    .venv\Scripts\Activate
    python scripts\download_football_data.py \
      --url "https://www.football-data.co.uk/englandm.php" \
      --out data/football-data \
      --since-days 365

Features:
- Respects robots.txt (best-effort).
- Skips files older than --min-year or --since-days (tries Last-Modified header first,
  falls back to reading CSV 'Date' column).
- Organizes downloads into competition subfolders by filename prefix (EC.csv -> EC/EC.csv).
"""
from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import csv
import io
import email.utils as eut

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
HEADERS = {"User-Agent": "MatchPredictor/1.0 (+https://github.com/your-repo)"}
TIMEOUT = 15
PROBE_BYTES = 64 * 1024  # bytes to read when probing CSV content


def allowed_by_robots(base_url: str, path: str) -> bool:
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        r = requests.get(robots_url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return True
        lines = r.text.splitlines()
        ua = None
        disallows = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("user-agent"):
                ua = line.split(":", 1)[1].strip()
            elif line.lower().startswith("disallow") and ua in ("*", "*:"):
                d = line.split(":", 1)[1].strip()
                if d:
                    disallows.append(d)
        for d in disallows:
            if path.startswith(d):
                logging.warning("Access disallowed by robots.txt: %s%s", base_url, path)
                return False
    except Exception:
        logging.debug("Could not fetch robots.txt, continuing")
    return True


def parse_csv_links(page_url: str) -> list[str]:
    r = requests.get(page_url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".csv"):
            links.append(urljoin(page_url, href))
    return sorted(set(links))


def parse_http_date(hdr: str) -> float | None:
    try:
        t = eut.parsedate_to_datetime(hdr)
        return t.timestamp()
    except Exception:
        return None


def probe_csv_for_max_year(url: str) -> int | None:
    """Try to download first chunk of CSV and return max year found in Date column.
       Returns None on failure / no dates found.
    """
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    buf.extend(chunk)
                    if len(buf) >= PROBE_BYTES:
                        break
            text = buf.decode(errors="replace")
            # use csv reader on the chunk - handle both comma and semicolon just in case
            sample = io.StringIO(text)
            sniffer = csv.Sniffer()
            dialect = None
            try:
                dialect = sniffer.sniff(sample.read(2048))
                sample.seek(0)
            except Exception:
                sample.seek(0)
            reader = csv.reader(sample, dialect=dialect) if dialect else csv.reader(sample)
            headers = None
            max_year = None
            for i, row in enumerate(reader):
                if i == 0:
                    headers = [h.strip().lower() for h in row]
                    # find date column index
                    if "date" in headers:
                        date_idx = headers.index("date")
                    else:
                        # try first column fallback
                        date_idx = 1 if len(row) > 1 else 0
                    continue
                if not row:
                    continue
                try:
                    raw = row[date_idx].strip()
                    # common format on football-data: DD/MM/YY or DD/MM/YYYY
                    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
                        try:
                            dt = datetime.strptime(raw, fmt)
                            year = dt.year
                            if max_year is None or year > max_year:
                                max_year = year
                            break
                        except Exception:
                            continue
                except Exception:
                    continue
                # stop early if we scanned enough rows
                if i > 200:
                    break
            return max_year
    except Exception as e:
        logging.debug("Probe failed for %s: %s", url, e)
        return None


def competition_from_filename(fn: str) -> str:
    # e.g. EC.csv -> EC ; E0.csv -> E0 ; if filename contains season tokens, pick the short prefix
    stem = Path(fn).stem
    # common pattern: prefix letters (competition code); take leading letters
    prefix = ''.join([c for c in stem if c.isalpha()])
    if prefix:
        return prefix.upper()
    # fallback to whole stem
    return stem.upper()


def download_if_new(url: str, out_dir: Path, threshold_dt: datetime | None) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = Path(urlparse(url).path).name
    comp = competition_from_filename(fn)
    comp_dir = out_dir / comp
    comp_dir.mkdir(parents=True, exist_ok=True)
    target = comp_dir / fn

    # HEAD to get metadata
    try:
        h = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except Exception as e:
        logging.error("HEAD failed for %s: %s", url, e)
        return False

    if h.status_code >= 400:
        logging.warning("Skipping %s (HEAD %s)", url, h.status_code)
        return False

    remote_len = h.headers.get("Content-Length")
    remote_lm = h.headers.get("Last-Modified")
    remote_ts = parse_http_date(remote_lm) if remote_lm else None

    # check recency using Last-Modified if present
    if threshold_dt and remote_ts:
        remote_dt = datetime.fromtimestamp(remote_ts)
        if remote_dt < threshold_dt:
            logging.info("Remote file older than threshold (Last-Modified): %s -> %s", fn, remote_dt.date())
            return False

    if target.exists():
        local_mtime = target.stat().st_mtime
        if remote_ts and local_mtime >= remote_ts:
            logging.info("Up-to-date: %s", fn)
            return False
        if remote_len and str(target.stat().st_size) == remote_len:
            logging.info("Same size, skipping: %s", fn)
            # still may be old; if threshold provided and remote_ts missing, check content probe below
            if not threshold_dt:
                return False

    # if we don't have Last-Modified and threshold is set, probe CSV content for dates
    if threshold_dt and not remote_ts:
        max_year = probe_csv_for_max_year(url)
        if max_year is not None:
            try:
                if max_year < threshold_dt.year:
                    logging.info("Remote CSV max date year %s is before threshold %s -> skipping %s", max_year, threshold_dt.year, fn)
                    return False
            except Exception:
                pass

    # GET and save
    logging.info("Downloading: %s -> %s", url, target)
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            tmp = target.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            tmp.replace(target)
            if remote_ts:
                os.utime(target, (remote_ts, remote_ts))
        return True
    except Exception as e:
        logging.error("Failed to download %s: %s", url, e)
        if target.exists():
            try:
                target.unlink()
            except Exception:
                pass
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Page URL that links CSVs (e.g. https://www.football-data.co.uk/englandm.php)")
    ap.add_argument("--out", default="data/football-data", help="Output directory for CSVs")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads")
    ap.add_argument("--since-days", type=int, default=365, help="Only download files newer than this many days (uses Last-Modified or CSV Date when needed). Use 0 to disable.")
    ap.add_argument("--min-year", type=int, help="Alternative: only download files with data >= this year (e.g. 2022)")
    ap.add_argument("--include", help="Comma-separated competition codes to include (e.g. D1,SP1,F1,I1). If set, only these comps are downloaded.")
    ap.add_argument("--exclude", help="Comma-separated competition codes to exclude (e.g. E2,E3)")
    args = ap.parse_args()

    page = args.url
    base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(page))
    parsed_path = urlparse(page).path
    if not allowed_by_robots(base, parsed_path):
        logging.error("Robots disallows scraping this page. Aborting.")
        return 1

    try:
        links = parse_csv_links(page)
    except Exception as e:
        logging.error("Failed to parse page: %s", e)
        return 2

    if not links:
        logging.info("No CSV links found on %s", page)
        return 0

    # parse include/exclude lists
    include_set = {s.strip().upper() for s in args.include.split(",")} if args.include else None
    exclude_set = {s.strip().upper() for s in args.exclude.split(",")} if args.exclude else None

    # filter links by competition code derived from filename
    filtered = []
    for link in links:
        fn = Path(urlparse(link).path).name
        comp = competition_from_filename(fn)
        if include_set and comp not in include_set:
            logging.debug("Skipping %s (comp %s not in include list)", fn, comp)
            continue
        if exclude_set and comp in exclude_set:
            logging.debug("Skipping %s (comp %s in exclude list)", fn, comp)
            continue
        filtered.append(link)

    if not filtered:
        logging.info("No CSV links match include/exclude filters. Nothing to do.")
        return 0

    out_dir = Path(args.out)
    threshold_dt = None
    if args.min_year:
        threshold_dt = datetime(args.min_year, 1, 1)
    elif args.since_days and args.since_days > 0:
        threshold_dt = datetime.utcnow() - timedelta(days=args.since_days)

    downloaded = 0
    for link in filtered:
        ok = download_if_new(link, out_dir, threshold_dt)
        if ok:
            downloaded += 1
        time.sleep(args.delay)
    logging.info("Done. Downloaded %d new/updated files.", downloaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
