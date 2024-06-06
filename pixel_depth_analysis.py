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
import utm

def tiff_to_np_RGB(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        #np.array([sr.read(1),src-read(2],srac.read(3])
        gain = 1.2 # Adjust the gain (brightness) of the image
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
        lons= np.array(xs)[0,:]
        lats = np.array(ys)[:,0]
        print('lons shape', lons.shape)    
        
    return big_img,src.transform,src.crs,lons,lats

def transform_coords_to_utm(df,crs):
    """Transform the latitude and longitude to the image coordinates"""
    lat = df["lat_ph"]
    lon = df["lon_ph"]
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    Ice_x,Ice_y = transformer.transform(lat, lon)
    
    return Ice_x, Ice_y

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
path = "C:/Users/signe/Downloads/Meltponds_batch_1"

# Extract the files in the folder and sort them: 
# 1_ICESat2.csv, 1_depth.csv, 1_tiff.tiff, 2_ICESat2.csv, 2_depth.csv, 2_tiff.tiff, ...
files = os.listdir(path)
files_comb = []
for j in range(1,int((len(files)-1)/3+1)):
    print(j)
    k = [11,14,18,8,9]
    meltpond = []
    for i in range(len(files)):
        if files[i].startswith(f"{k[j-1]}_"):
            meltpond.append(files[i])
    files_comb.append(meltpond)

# Read the drift values
drift = pd.read_csv(os.path.join(path,"drift_values.csv"))

index = 4 # CHANGE ME
drift_constant = 1.0 # change me!!


tiff_path = f"{path}/{files_comb[index][2]}"
print(tiff_path)
icesat_path = f"{path}/{files_comb[index][0]}"
depth_path = f"{path}/{files_comb[index][1]}"
print(depth_path)

# Extract the time difference between the two files
tiff_time = extract_datetime(files_comb[index][2])
print(tiff_time)
icesat_time = extract_datetime(files_comb[index][0])
print(icesat_time)
timediff = datetime_diff(tiff_time,icesat_time) 
print(timediff)

# Open the tiff file and the icesat file
img_1_RGB,transform_1,src = tiff_to_np_RGB(tiff_path)
img_1,transform_1,src,lons,lats = tiff_to_np(tiff_path)
print(img_1.shape)

icesat_df = pd.read_csv(icesat_path)
depth_df = pd.read_csv(depth_path, sep=',')
depth_x,depth_y = transform_coords_to_utm(depth_df,src)
ice_x,ice_y = transform_coords_to_utm(icesat_df,src)
drift_xNy = depth_x + drift["xs_drift"][index]*timediff*drift_constant
drift_yNy = depth_y + drift["ys_drift"][index]*timediff*drift_constant


def nearest_pixel(ice_x, ice_y, sentinel_x, sentinel_y):
    """find nearest pixel to meltpond from icesat depth"""
    x_pond_pixels = np.zeros(len(ice_x),dtype=int)
    y_pond_pixels = np.zeros(len(ice_y),dtype=int)

    # x-direction
    for i in range(len(ice_x)):
        diff = abs((sentinel_x) - (ice_x[i])) 
        min_diff = np.argmin(diff)
        x_pond_pixels[i] = min_diff
        
    # y-direction
    for n in range(len(ice_y)):
        diff = abs((sentinel_y)- (ice_y[n]))
        min_diff = np.argmin(diff)
        y_pond_pixels[n] = min_diff
    
    return x_pond_pixels, y_pond_pixels

#print("UTM zone:", src.wkt)

def pixel_depth_data(index, x_pond_pixels, y_pond_pixels, sentinel_x, sentinel_y, pond_img, icesat_data, zone):
    """ Appending all data about pixels including corresponding depth measured by icesat to array"""
    
    pixel_information = np.zeros((len(x_pond_pixels),8))
   
    pixel_information[:,0] = np.zeros(len(x_pond_pixels))+index
   
    x_pixels = sentinel_x[x_pond_pixels]
    y_pixels = sentinel_y[y_pond_pixels]
    print(zone)
    pixel_lat, pixel_lon = utm.to_latlon(x_pixels,y_pixels,int(zone[0]),zone[1])
   
    pixel_information[:,[1,2]] = np.transpose([pixel_lat, pixel_lon])
   
    for i in range(pond_img.shape[0]):
        print(i)
        pixel_information[:,i+3] = pond_img[i,y_pond_pixels,x_pond_pixels]
    pixel_information[:,7] = depth_df['depth']
    
    return pixel_information, x_pixels, y_pixels


def final_df_setup(pixel_information, sentinel_datetime, icesat_datetime):
    """ Joins the pixel information with general information about the melt pond"""
    
    columns = ['Meltpond index','Latitude','Longtitude','b08 (NIR)','b02 (red)','b03 (green)','b04 (blue)','Depth [m]']
    pixel_df = pd.DataFrame(pixel_information, columns = columns)
    
    sentinel_time = []
    for i in range(pixel_information.shape[0]):
        sentinel_time.append(sentinel_datetime)
    
    icesat_time = []
    for i in range(pixel_information.shape[0]):
        icesat_time.append(icesat_datetime)
    
    d = {'Sentinel Time':sentinel_time, 'IceSat Time':icesat_time}
    df2 = pd.DataFrame(data=d)
    final_df = pixel_df.join(df2)
    
    return final_df


def save_pixel_file(file_path, final_data):
    """ Saves meltpond data as csv file.
    Creates file if path doesn't exist
    Appends to file if path exists.
    """
    #Prepare data for current meltpond
    data = final_data
    
    # Check if the CSV file exists
    file_exists = os.path.isfile(file_path)
    
    if file_exists:
        existing_data = pd.read_csv(file_path)
        new_data = pd.concat([existing_data, data])
        new_data.to_csv(file_path, sep=',', index=False)
    
    if not file_exists:
        data.to_csv(file_path, sep=',', index=False)
    
    
def check_coordinates(markers):
    for key, value in markers.items():
        if value.shape != (1, 2):
            return False  # If shape is not (1, 1), return False
    return True  # All arrays have shape (1, 1)

def get_positions(event):
    """ extracts positons from markers"""
    click_pos = clicker.get_positions()
    #print("Click positions:", click_pos)
    if check_coordinates(click_pos):
        for key in click_pos.keys():
            if click_pos[key].shape == (1,2):
                pos = click_pos[key][0] 
                clicker_coordinates.append(pos)
        print('Click coordinates:',clicker_coordinates)

zone = [src.wkt[26:28],src.wkt[28]]
x_pond_pixels, y_pond_pixels = nearest_pixel(drift_xNy, drift_yNy, lons, lats)
pixel_info_matrix, x_pixels, y_pixels = pixel_depth_data(index, x_pond_pixels, y_pond_pixels, lons, lats, img_1, depth_df, zone)
final_df = final_df_setup(pixel_info_matrix, tiff_time, icesat_time)

drift_xNy = depth_x + drift["xs_drift"][index]*timediff*drift_constant
drift_yNy = depth_y + drift["ys_drift"][index]*timediff*drift_constant

fig,(ax1,ax2) = plt.subplots(1,2)
img_2 = img_1_RGB
img_2[:,y_pond_pixels,x_pond_pixels] = 0
show(img_2,transform=transform_1,ax=ax1)
ax1.scatter(depth_x,depth_y)
ax1.scatter(drift_xNy,drift_yNy,c=depth_df["depth"],cmap="viridis")
ax1.scatter(max(lons),max(lats))
ax2.scatter(depth_df["x_atc"],depth_df["depth"])

clicker = clicker(ax1, ["1", "2"], markers = ["*", "*"])

clicker_coordinates = []
# Connect mouse click to marker positions
fig.canvas.mpl_connect('button_press_event', get_positions)

product_file_path = "C:/Users/signe/OneDrive/Dokumenter/GitHub/Fagprojekt_MeltpondsNY/pixel_data_final"

def append(event):
    """Appending pixel data to file if marker conditions matched"""
    
    x_click = [clicker_coordinates[0][0],clicker_coordinates[1][0]]
    y_click = [clicker_coordinates[0][1],clicker_coordinates[1][1]]
    click_lat, click_lon = utm.to_latlon(np.array(x_click),np.array(y_click),int(zone[0]),zone[1])
    
    y_in = (min(click_lat) < pixel_info_matrix[:, 1]) & (pixel_info_matrix[:, 1] < max(click_lat))
    y_in_coor = pixel_info_matrix[:,1][y_in]
    append_ready = final_df[final_df['Latitude'].isin(y_in_coor)]
    
    
    save_pixel_file(product_file_path, append_ready)
  

# Create button to save the data
button_ax_cut = fig.add_axes([0.85, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_cut = Button(button_ax_cut, 'append data')
button_cut.on_clicked(append)
plt.show()




