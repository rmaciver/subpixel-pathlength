import sys
sys.path.append("../src")

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import worker
from pathlib import Path
import matplotlib.pyplot as plt

import cld, worker

n_workers = 10  
run_all = True

xcal, ycal = 14, 14
n_lines = 1000
n_files = 100
runs = ["1", "2", "3"]
sample_names = ["sample1", "sample4"]
sample_input_paths = ["../sample1/bin", "../sample4/bin"]
sample_output_paths = ["../sample1", "../sample4"]

# ---- run cld for all samples with workers
for run in runs: 
    for pathin, pathout, sample_name in zip(sample_input_paths,sample_input_paths, sample_names): 
        os.makedirs(f"{pathout}/stats-{run}", exist_ok=True)
        files = sorted(list(Path(pathin).glob("*tif")))
        
        files = np.random.choice(files, n_files, replace=False) # sampling n_files from the list

        if n_workers == 1: 
            print(f"Run in loop")
            for idj, file in tqdm(enumerate(files)): 
                cld.run_cld(file, idj, run, pathout, xcal, ycal, n_lines, simple_output=True)
        else: 
            print("Run parallel worker process")
            kwargs = {
                "run": run, 
                "pathout": pathout, 
                "xcal": xcal, 
                "ycal": ycal, 
                "n_lines": n_lines
            }
            worker.run_func(files, n_workers, cld.run_cld, kwargs)

# ---- cat the results
for run in runs: 
    for sample_name in ["sample1", "sample4"]:
        pathin = f"../{sample_name}/stats-{run}"
        df = pd.concat([pd.read_csv(f) for f in Path(pathin).glob("*.zip")])
        df.to_csv(f"../run-{run}-{sample_name}-cld-stats.zip")
