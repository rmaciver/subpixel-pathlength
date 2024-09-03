"""
MIT License

Copyright (c) 2024 M. Ryan MacIver

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__author__ = "M. Ryan MacIver"
__copyright__ = "Copyright 2024, M. Ryan MacIver"
__credits__ = ["M. Ryan MacIver"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "M. Ryan MacIver"
__email__ = "rmaciver@chemetc.com"
__status__ = "Development"

from pathlib import Path
import re
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import worker

import matplotlib.pyplot as plt


def get_face_pts(img: np.ndarray) -> list: 
    """ 
    Generate indices on the edges of an image.
    
        Parameters:
        ----------
            img (np.ndarray): The input image.
        Returns:
        ----------
            list: A list of arrays containing the 
                generated points on the faces of the image.
    """
    shape = img.shape
    # generate points on faces
    # --- x0 -> x1, y1
    xy = np.meshgrid(np.arange(0, shape[0]-1), [shape[1]-1])
    S1 = np.vstack([xy[0].ravel(), xy[1].ravel()]).T
    # --- x0 --> x1, y0
    xy = np.meshgrid(np.arange(0, shape[0]-1), [0])
    S2 = np.vstack([xy[0].ravel(), xy[1].ravel()]).T
    # --- x0, y0 -> y1
    xy = np.meshgrid([0], np.arange(0, shape[1]-1))
    S3 = np.vstack([xy[0].ravel(), xy[1].ravel()]).T
    # --- x1, y0 -> y1
    xy = np.meshgrid([shape[0]-1], np.arange(0, shape[1]-1))
    S4 = np.vstack([xy[0].ravel(), xy[1].ravel()]).T

    return [S1, S2, S3, S4]

def get_line_ends(face_pts: list) -> np.ndarray: 
    """
    Get the end points of a line defined by two random points on the faces of an image.


        Parameters:
        ----------
            face_pts (list): A list of arrays containing the 
                generated points on the faces of the image.
        Returns:
        ----------
            np.ndarray: An array containing the end points of a line defined by two 
                random points on the faces of an image.
    """
    i = np.random.choice([0,1,2,3], 2, replace=False)
    p1 = face_pts[i[0]][np.random.randint(0, len(face_pts[i[0]]))]
    p2 = face_pts[i[1]][np.random.randint(0, len(face_pts[i[1]]))]
    idx = np.argsort([p1[0], p2[0]]) # ensure x is increasing
    return np.array([p1, p2])[idx]

def cal_subpx_path(pts, xcal:float=1, ycal:float=1): 
    """ 
    Calculate the path length of a line through each pixel in an image.
        
        Parameters:
        ----------
            pts (np.ndarray): An array containing the end points of a line.
            xcal (float): The x dimension / size of the pixel
            ycal (float): The y dimension/ size of the pixel

        Returns:
        ----------
            tuple: A tuple containing the indices of pixels crossed in the image,
                the path length through each pixel, and the x and y vectors for each pixel.
    """
    d = np.diff(pts, axis=0).ravel().astype(np.int64)
    
    if pts[0][0] == pts[1][0]:
        return None, None, None # not handling straight lines
    if pts[0][1] == pts[1][1]:
        return None, None, None # not handling straight lines
    
    tan = d[1]/d[0]

    # take x spacing, find where y crossed
    xs = np.arange(pts[0][0], pts[1][0])
    ysatx = pts[0][1] + (xs-xs[0])*tan
    ysatx[0], ysatx[-1], pts[:,1]
    atx = np.vstack([xs, ysatx]).T

    # take y spacing, find where x crossed
    if pts[0][1] > pts[1][1]: 
        step = -1
    else: 
        step = 1
    
    ys = np.arange(pts[0][1], pts[1][1], step)
    xsaty = pts[0][0] + (ys-ys[0])*(1/tan)
    aty = np.vstack([xsaty, ys]).T

    apts = np.vstack([atx, aty]) 
    ix = np.argsort(apts[:,0])
    # sort and remove duplicates
    apts_v = apts[ix][1:-1] 
    # indices of pixels crossed in image
    idx = np.floor(apts_v).astype(int)
    # x and y vectors for each pixel
    dl = np.diff(apts[ix][1:], axis=0)*np.array([xcal, ycal])
    # path length through pixel
    dists = np.sqrt(dl[:,0]**2+dl[:,1]**2)
    return idx, dists, apts

# pick a random point on the faces
def run_cld(
        filename: str, 
        run: int, 
        pathout: str, 
        xcal: float, 
        ycal: float, 
        n_lines: int, 
        simple_output: bool=True
    ): 
    """ A wrapper function to apply the chord length distribution analysis to an image
    """
    img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
    face_pts = get_face_pts(img)
    shape = img.shape
    vall, pall, idxall, lineidx = [],[],[], []
    sli_idx = int(re.search(r'\d+', filename.name.split("Z")[1]).group())
    
    for ilx, n in enumerate(range(n_lines)): 
        pts = get_line_ends(face_pts)
        if pts.shape != (2,2): continue #print(f"pts shape {pts.shape}")
        idx, dists, apts = cal_subpx_path(pts, xcal=xcal, ycal=ycal)
        if idx is None: 
            continue
        arr = img[idx[:,0], idx[:,1]] > 0
        stops = np.where(np.diff(arr))[0]
        asplit = np.hsplit(arr, indices_or_sections=stops+1) #, stops
        p = [a[0] for a in asplit]
        dsplit = np.hsplit(dists, indices_or_sections=stops+1) #, stops
        v = [a.sum() for a in dsplit]
        vall.extend(v)
        pall.extend(p)
        idxall.extend(np.repeat(sli_idx, len(v)))
        lineidx.extend(np.repeat(ilx, len(v)))

    if simple_output: 
        df = pd.DataFrame({"length": vall, "phase-1pore": pall, "line_idx": lineidx, "image_idx": idxall }, index=range(len(vall)))
        df = df[df["length"]>0]
        df.to_csv(f"{pathout}/stats-{run}/stats-{sli_idx:04d}.zip", index=False)
    else: 
        df = pd.DataFrame({"length": vall, "phase": pall}, index=range(len(vall)))
        df = df[df["length"]>0]
        df["length"] = df["length"].astype(np.int16)
        df.to_csv(f"{pathout}/stats-{run}/stats-{sli_idx:04d}.zip", index=False)

