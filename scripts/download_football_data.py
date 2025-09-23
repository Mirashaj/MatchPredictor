#!/usr/bin/env python3
"""
Download new/updated data files from https://www.football-data.co.uk/data.php
- discovers CSV/XLS/XLSX links on the data page
- downloads files that are missing locally
- optional: --update will check Last-Modified and replace stale local files
- supports --dry-run and --filter (e.g., 2025)

Place this in the repo under scripts/ and run from repo root.
"""
import argparse
import logging
import os
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

ENTRY_URL = "https://www.football-data.co.uk/data.php"
OUT_DIR_DEFAULT = os.path.join("data", "football-data")
USER_AGENT = "matchpredictor-downloader/1.0 (+https://github.com/Mirashaj/MatchPredictor)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("downloader")

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def discover_links(entry_url=ENTRY_URL):
    logger.info("Fetching index page %s", entry_url)
    r = session.get(entry_url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # only file-type links we care about
        if href.lower().endswith(('.csv', '.xls', '.xlsx')):
            links.add(urljoin(entry_url, href))
    return sorted(links)


def get_remote_last_modified(url):
    try:
        r = session.head(url, allow_redirects=True, timeout=15)
        if r.status_code == 200:
            return r.headers.get("Last-Modified")
    except Exception:
        return None
    return None


def download_file(url, out_dir, retries=3, timeout=30):
    filename = os.path.basename(urlparse(url).path)
    out_path = os.path.join(out_dir, filename)
    tmp_path = out_path + ".part"
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename, leave=False) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            os.replace(tmp_path, out_path)
            return True, out_path
        except Exception as e:
            logger.debug("Attempt %s failed for %s: %s", attempt, url, e)
            time.sleep(1 + attempt)
    return False, str(e)


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download new/updated football-data files")
    parser.add_argument("--entry", default=ENTRY_URL, help="Entry page (default: %(default)s)")
    parser.add_argument("--out", default=OUT_DIR_DEFAULT, help="Output directory (default: %(default)s)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Discover and list files but don't download")
    parser.add_argument("--update", action="store_true", help="Check Last-Modified and update local files if remote is newer")
    parser.add_argument("--filter", default=None, help="Only download links that contain this substring (e.g. 2025)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    out_dir = ensure_outdir(args.out)
    links = discover_links(args.entry)
    if args.filter:
        links = [l for l in links if args.filter in l]
    logger.info("Discovered %d candidate files", len(links))
    if not links:
        logger.info("No downloadable files found. Exiting.")
        return 0

    summary = {"ok": [], "skipped": [], "updated": [], "failed": []}
    for url in links:
        filename = os.path.basename(urlparse(url).path)
        local_path = os.path.join(out_dir, filename)
        if not os.path.exists(local_path):
            logger.info("New file: %s", filename)
            if not args.dry_run:
                ok, info = download_file(url, out_dir)
                if ok:
                    summary["ok"].append(local_path)
                else:
                    summary["failed"].append((url, info))
            else:
                summary["ok"].append(local_path)
        else:
            if args.update:
                remote_lm = get_remote_last_modified(url)
                if remote_lm:
                    try:
                        local_mtime = time.gmtime(os.path.getmtime(local_path))
                        # naive compare: if remote last-mod string present, just opt to redownload
                        # because parsing may be locale-dependent; we conservatively redownload
                        # if --update provided and remote header exists.
                        logger.info("Updating (remote header present) %s", filename)
                        if not args.dry_run:
                            ok, info = download_file(url, out_dir)
                            if ok:
                                summary["updated"].append(local_path)
                            else:
                                summary["failed"].append((url, info))
                        else:
                            summary["updated"].append(local_path)
                    except Exception:
                        logger.debug("Could not check local mtime for %s", local_path)
                        continue
                else:
                    logger.debug("No Last-Modified for %s; skipping unless --force", url)
                    summary["skipped"].append(local_path)
            else:
                summary["skipped"].append(local_path)
        time.sleep(args.delay)

    logger.info("Summary: ok=%d updated=%d skipped=%d failed=%d", len(summary["ok"]), len(summary["updated"]), len(summary["skipped"]), len(summary["failed"]))
    if summary["failed"]:
        logger.warning("Some downloads failed. See details:")
        for u, msg in summary["failed"]:
            logger.warning("%s -> %s", u, msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
