# src/io_utils.py
import os, zipfile
import pandas as pd

def extract_zips_to_dir(data_dir):
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.zip'):
            zpath = os.path.join(data_dir, fname)
            outdir = os.path.join(data_dir, os.path.splitext(fname)[0])
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
                with zipfile.ZipFile(zpath, 'r') as z:
                    z.extractall(outdir)

def find_time_series_files(data_dir):
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def read_table(path, nrows=None):
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, low_memory=False, nrows=nrows)
    else:
        return pd.read_excel(path, nrows=nrows)
