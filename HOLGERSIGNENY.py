"""Drift correlation and plotting"""

import rasterio
from rasterio.plot import show,adjust_band
from matplotlib.widgets import Button
import pandas as pd
import os
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
import Crop_Sentinel2 as sentinel_crop
import Crop_ICESat2 as icesat_crop
from mpl_point_clicker import clicker
import tkinter as tk
from tkinter import messagebox
import re 
import csv

def tiff_to_np_RGB(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        gain = 1.5 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

def tiff_to_np(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([src.read(i) for i in (4,3,2,1)])
        """
        1: blue, 2. green, 3: red, 4: NIR
        """
        band1 = src.read(1)
        height = band1.shape[0]
        width = band1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons= np.array(xs)
        lats = np.array(ys)
        print('lons shape', lons.shape)    
    return big_img,src.transform,src.crs,lons,lats

def transform_coords_to_utm(df,crs):
    """Transform the latitude and longitude to the image coordinates"""
    lat = df["lat_ph"]
    lon = df["lon_ph"]
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    Ice_x,Ice_y = transformer.transform(lat, lon)
    return Ice_x,Ice_y

def extract_datetime(file_name):
    """Extracts the datetime from the file name"""
    if file_name.endswith("tiff"):
        date_time_str = file_name.split("_")[2]
        return datetime.strptime(date_time_str, "%Y%m%dT%H%M%S")
    elif file_name.endswith("csv"):
        date_time_str = file_name.split("_")[2]  
        return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
    else:
        return None 

def datetime_diff(datetime1, datetime2):
    """Calculates the difference in seconds between two datetime objects"""
    return(datetime2.timestamp()-datetime1.timestamp())

# Folder containing input data
path = "C:/Users/holge/Downloads/meltpond_storage"

# Extract the files in the folder and sort them: 
# 1_ICESat2.csv, 1_depth.csv, 1_tiff.tiff, 2_ICESat2.csv, 2_depth.csv, 2_tiff.tiff, ...
files = os.listdir(path)
files_comb = []
for j in range(1,int((len(files)-1)/3+1)):
    print(j)
    meltpond = []
    for i in range(len(files)):
        if files[i].startswith(f"{j}_"):
            meltpond.append(files[i])
    files_comb.append(meltpond)

print(files_comb)

# Read the drift values
drift = pd.read_csv(os.path.join(path,"drift_values.csv"))

index = 1 # CHANGE ME!
# tiff_path = f"{path}/{files_comb[index][2]}"
# icesat_path = f"{path}/{files_comb[index][0]}"
# depth_path = f"{path}/{files_comb[index][1]}"
tiff_path = os.path.join(path,"2_T20XNS_20210628T193909_gt3rw.tiff")
icesat_path = os.path.join(path,"2_ATL03_20210628193330_00791204_006_01_gt3rw_0.csv")
depth_path = os.path.join(path,"2_ATL03_20210628193330_00791204_006_01_gt3rw_0.csv_depths.csv")

# Extract the time difference between the two files
#tiff_time = extract_datetime(files_comb[index][2])
#icesat_time = extract_datetime(files_comb[index][0])
tiff_time = extract_datetime("2_T20XNS_20210628T193909_gt3rw.tiff")
icesat_time = extract_datetime("2_ATL03_20210628193330_00791204_006_01_gt3rw_0.csv")
timediff = datetime_diff(tiff_time,icesat_time)
print(timediff)

# Open the tiff file and the icesat file
fig,(ax1,ax2) = plt.subplots(1,2)
img_1_RGB,transform_1,src = tiff_to_np_RGB(tiff_path)
img_1,transform_1,src,lons,lats = tiff_to_np(tiff_path)
print(img_1.shape)

icesat_df = pd.read_csv(icesat_path)
depth_df = pd.read_csv(depth_path)
depth_x,depth_y = transform_coords_to_utm(depth_df,src)
ice_x,ice_y = transform_coords_to_utm(icesat_df,src)
drift_xNy = depth_x + drift["xs_drift"][index]*timediff
drift_yNy = depth_y + drift["ys_drift"][index]*timediff

show(img_1_RGB,transform=transform_1,ax=ax1)
ax1.scatter(depth_x,depth_y)
ax1.scatter(drift_xNy,drift_yNy,c=depth_df["depth"],cmap="viridis")
ax2.scatter(depth_df["x_atc"],depth_df["depth"])

# Create button to save the data
button_ax_cut = fig.add_axes([0.85, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_cut = Button(button_ax_cut, 'append data')
plt.show()




