"""

"""
from copy import deepcopy
from pathlib import Path
from threading import Thread
from queue import Queue
import time
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress

def _producer(q_in, filenames):
    """ """
    for idx, filename in enumerate(filenames):
        q_in.put((filename, idx))
    q_in.put((None, None))
 
def _consumer(q_in, q_out, func, func_kwargs):    
    """
    """
    while True:
        filename, idx = q_in.get() 
        if idx is None: 
            q_in.put((None, None)) 
            return   
        out = func(filename, idx, **func_kwargs)
        q_out.put(out)
        
def _monitor(q_out:Queue, n:int, objs:dict): 
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing ...", total=n)
        while (not progress.finished): # | (not q_out.empty()):            
            out = q_out.get()
            progress.update(task, advance=1)
            #objs[out["image_idx"]] = out["idxs"]
            objs.append(out)

def run_func(files, n_threads, func, func_kwargs, q_size_max=100, **kwargs): 
    """ """   
    print(f"Setting up multiprocessing for {len(files)} files")
    n = len(files)
    q_in = Queue(maxsize=q_size_max)
    q_out = Queue()
    if len(files) == 0: 
        raise ValueError(f"No files to process")
        
    for i in range(n_threads):
        worker = Thread(target=_consumer, args=(q_in, q_out, func, func_kwargs))
        worker.setDaemon(True)
        worker.start()

    objs = list()
    mon = Thread(target=_monitor, args=(q_out, n, objs))
    mon.start()
    producer = Thread(target=_producer, args=(q_in, files))
    producer.start()
    
    producer.join()
    mon.join()
    print('Done')
    return objs
    