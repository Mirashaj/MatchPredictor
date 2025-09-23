import glob, pandas as pd, os

files = glob.glob(os.path.join('data','**','*.csv'), recursive=True)
out_rows = []
start = pd.Timestamp('2024-07-01')
end = pd.Timestamp('2025-06-30')
print('Scanning', len(files), 'CSV files under data/...')
for f in files:
    norm = os.path.normpath(f)
    if os.path.normpath(os.path.join('data','predictions')) in norm:
        continue
    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception:
        # skip unreadable
        continue
    if df.empty:
        continue
    cols = {c.lower(): c for c in df.columns}
    # find date-like column
    date_col = None
    for cand in ['date','match_date','kickoff','matchday','day']:
        if cand in cols:
            date_col = cols[cand]
            break
    if date_col is None:
        for c in df.columns:
            if 'date' in c.lower() or 'day' in c.lower():
                date_col = c
                break
    # robust date parsing: try several common formats first, then fall back to
    # dayfirst/daylast parsing while suppressing pandas' format-inference warnings.
    def try_parse_dates(series):
        # work on a string copy to avoid type issues
        s = series.astype(str).replace({'nan': '', 'None': ''})
        formats = [
            '%Y-%m-%d', '%Y/%m/%d',
            '%d/%m/%Y', '%d.%m.%Y', '%d-%m-%Y',
            '%m/%d/%Y', '%m-%d-%Y',
            '%d %b %Y', '%d %B %Y'
        ]
        for fmt in formats:
            try:
                parsed = pd.to_datetime(s, format=fmt, errors='coerce')
            except Exception:
                parsed = pd.to_datetime(s, format=fmt, errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed

        # last-resort: try dayfirst True/False but silence warnings from pandas
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parsed = pd.to_datetime(s, dayfirst=True, errors='coerce', infer_datetime_format=False)
            if parsed.notna().sum() > 0:
                return parsed
            parsed = pd.to_datetime(s, dayfirst=False, errors='coerce', infer_datetime_format=False)
            return parsed

    if date_col is None:
        df['__date__'] = pd.NaT
    else:
        df['__date__'] = try_parse_dates(df[date_col])
    # find home/away columns
    home_col = None; away_col = None
    for c in df.columns:
        if 'home' in c.lower() and home_col is None:
            home_col = c
        if 'away' in c.lower() and away_col is None:
            away_col = c
    if home_col is None or away_col is None:
        continue
    # filter by date range
    sel = df['__date__'].notna() & (df['__date__'] >= start) & (df['__date__'] <= end)
    if sel.any():
        filtered = df.loc[sel, :].copy()
        # normalize column names used later by trainer: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Div
        mapping = {}
        mapping[home_col] = 'HomeTeam'
        mapping[away_col] = 'AwayTeam'
        if 'fthg' in cols:
            mapping[cols['fthg']] = 'FTHG'
        if 'ftag' in cols:
            mapping[cols['ftag']] = 'FTAG'
        if 'ftr' in cols:
            mapping[cols['ftr']] = 'FTR'
        if 'div' in cols:
            mapping[cols['div']] = 'Div'
        mapping[date_col if date_col else '__date__'] = 'Date'
        filtered = filtered.rename(columns=mapping)
    # ensure Date column exists and formatted using the same robust parser
    filtered['Date'] = try_parse_dates(filtered['Date'])
    out_rows.append(filtered[[c for c in ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Div'] if c in filtered.columns]])
# concat
if not out_rows:
    print('No matches found for 2024-07-01 to 2025-06-30')
else:
    out_df = pd.concat(out_rows, ignore_index=True)
    out_path = os.path.join('data','training_2024_2025.csv')
    out_df.to_csv(out_path, index=False)
    print('Wrote', len(out_df), 'rows to', out_path)
