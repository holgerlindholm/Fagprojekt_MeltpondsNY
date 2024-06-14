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

segment_folder_path = "C:/Users/chrel/Documents/Fagprojekt_Lokal/Fagprojekt-Meltponds-master/Tile_10_07_1609"

storage_folder = "C:/Users/chrel/Documents/Fagprojekt_Lokal/Fagprojekt-Meltponds-master/Tile_10_07_1609/Meltponds"

#Error message pop up
def error_message_window(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Error", message)

#Pick next segment for display
def select_data(number, beam):

    #Create lists containing names of all files in folders
    segment_files = os.listdir(segment_folder_path)

    choosen_file_name = ""
    print(f'{beam}_{number}.tiff')

    for filename in segment_files:
        if filename.endswith(f'{beam}_{number}.tiff'):
            choosen_file_name = filename
            break  # Stop iterating once the first matching filename is found
    print(choosen_file_name)
    #List with all files ending with the same number
    segments = []

    for file_name in segment_files:
        parts = file_name.split('_')
        last_part = parts[-1]
        extracted_number = ''.join(filter(str.isdigit, last_part))

        # Check if the extracted number matches the target number and right beam
        if extracted_number == number and parts[-2] == beam:
            segments.append(file_name)

    if len(segments) == 3:
        return segments
    else:
        error_message_window(f"Could not find 3 files with name {beam}_{number}")

def extract_datetime(file_name):
        parts = file_name.split('_')
        if len(parts) >= 2 and file_name.endswith("tiff"):
            date_time_str = parts[1]  
            return datetime.strptime(date_time_str, "%Y%m%dT%H%M%S")
        elif len(parts) >= 2 and file_name.endswith("csv"):
            date_time_str = parts[1]  
            return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
        else:
            return None 


def tiff_to_np(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        gain = 1 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

def thin_ICE_data(df):
    """Thin the ICESat data"""
    return df.iloc[::100,:]

def transform_coords_to_utm(df,crs):
    """Transform the latitude and longitude to the image coordinates"""
    lat = df["lat_ph"]
    lon = df["lon_ph"]
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    Ice_x,Ice_y = transformer.transform(lat, lon)
    return Ice_x,Ice_y

def transform_coords_to_latlon(coords, crs):
    """Transform UTM coordinates back to latitude and longitude"""
    tiff_x = coords[0]  # UTM easting
    tiff_y = coords[1]  # UTM northing
    transformer = Transformer.from_crs(crs, 'EPSG:4326')
    lat, lon = transformer.transform(tiff_x, tiff_y)
    return lat, lon

def average_drift(drift_vector, time_diff_sec):
    drift_xy = [0,0]
    for key in drift_vector[0].keys():
        drift_xy[0] = drift_vector[0][key][0]+drift_xy[0]
        drift_xy[1] = drift_vector[0][key][1]+drift_xy[1]
    return [(drift_xy[0])/(3*time_diff_sec),(drift_xy[1])/(3*time_diff_sec)]

def find_meltpond_id():
    #Create lists containing names of all files in folders
    stored_files = os.listdir(storage_folder)
    regex_pattern = re.compile(r"\d+_")
    filtered_filenames = [filename for filename in stored_files if regex_pattern.match(filename)]

    highest_digit_value = 0

    for filename in filtered_filenames:
        parts = filename.split('_') 
        numeric_part = parts[0]
        numeric_value = int(numeric_part)
        if numeric_value > highest_digit_value:
            highest_digit_value = numeric_value

    #check if the folder is empty
    if not filtered_filenames:
        return 1
    else:
        return highest_digit_value + 1

def save_drift_csv(file_path, meltpond_id, drift_xy_sec):
    
    # Check if the CSV file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(['Meltpond_ID', 'xs_drift', 'ys_drift'])
        
        # Prepare the row data
        row_data = [meltpond_id, drift_xy_sec[0], drift_xy_sec[1]]
        
        # Write the row to the CSV file
        writer.writerow(row_data)

def transform_tiff_name(tiff_name, meltpond_id):
    parts = tiff_name.split('_')     
    return f"{meltpond_id}_{parts[0]}_{parts[1]}_{parts[7]}"    

def datetime_diff(datetime1, datetime2):
    return(datetime2.timestamp()-datetime1.timestamp())


def plot_images(Sentinel_name_1, Sentinel_name_2,ICE_track, ice_path):
    """Plots the two images"""
    fig, ax = plt.subplots(1,3, figsize=(12,6))
    img_1,transform_1,src = tiff_to_np(Sentinel_name_1)
    img_2,transform_2,_ = tiff_to_np(Sentinel_name_2)
    show(img_1,transform=transform_1,ax=ax[0])
    show(img_2,transform=transform_2,ax=ax[1])

    # Thin the ICE data to make it easier to plot on the image
    trimmed_df = thin_ICE_data(ICE_track)

    # Transform the coordinates to the image coordinates
    Ice_x,Ice_y = transform_coords_to_utm(trimmed_df,src)

    # Plot the ICESat-2 data
    ax[0].scatter(Ice_x,Ice_y,color='red',s=1,alpha=0.5)
    ax[1].scatter(Ice_x,Ice_y,color='red',s=1,alpha=0.5)
    ax[2].scatter(ICE_track["x_atc"],ICE_track["h_ph"],color='black',s=1,alpha=0.5)
    ax[2].set_ylim(10,20)

    tiff_1_name = Sentinel_name_1.split("\\")
    tiff_2_name = Sentinel_name_2.split("\\")
    csv_3_name = ice_path.split("\\")

    time_1_tiff = extract_datetime(tiff_1_name[-1])
    print(f"Time 1: {time_1_tiff}")
    time_2_tiff = extract_datetime(tiff_2_name[-1])
    print(f"Time 2: {time_2_tiff}")
    time_3_csv = extract_datetime(csv_3_name[-1])
    print(f"Time 3: {time_3_csv}")
    time_diff_tiff = datetime_diff(time_1_tiff,time_2_tiff)
    print(f"time diff between tiffs 2-1:{time_diff_tiff}")
    time_diff_ice = datetime_diff(time_1_tiff,time_3_csv)
    print(f"time diff between tiff-ice 3-1: {time_diff_ice}")

    ax[0].set_title('Time: 0', fontsize=14, weight='bold', color='green')
    if time_diff_tiff < 0:
        ax[1].set_title(f'Time: -{timedelta(seconds=abs(time_diff_tiff))}', fontsize=14, weight='bold', color='red')
    else:
        ax[1].set_title(f'Time: {timedelta(seconds=time_diff_tiff)}', fontsize=14, weight='bold', color='green')

    if time_diff_ice <= 0 :
        ax[2].set_title(f'Time: -{timedelta(seconds=abs(time_diff_ice))}', fontsize=14, weight='bold', color='red')
    else:
        ax[2].set_title(f'Time: {timedelta(seconds=time_diff_ice)}', fontsize=14, weight='bold', color='green')
    
    klicker_1 = clicker(ax[0],["1", "2", "3"], markers=["+", "+", "+"])

    klicker_2 = clicker(ax[1],["1", "2", "3"], markers=["+", "+", "+"])

    def check_coordinates(markers):
        for key, value in markers.items():
            if value.shape != (1, 2):
                return False  # If shape is not (1, 2), return False
        return True  # All arrays have shape (1, 2)

    def calculate_coordinate_drift(markers_1, markers_2):
        differences = {}
        for key in markers_1.keys():
            difference = markers_1[key][0] - markers_2[key][0]
            differences[key] = difference
        return differences

    tiff_bbox = []
    drift_vector = []
    new_ice_track = []

    def auto_zoom_icesat(event):
        tiff_xlim = ax[0].get_xlim()
        tiff_ylim = ax[0].get_ylim()
        tiff_coords_0 = [tiff_xlim[0],tiff_ylim[0]]
        tiff_coords_1 = [tiff_xlim[1],tiff_ylim[1]]

        tiff_lat_0, tiff_lon_0 = transform_coords_to_latlon(tiff_coords_0, src)
        tiff_lat_1, tiff_lon_1 = transform_coords_to_latlon(tiff_coords_1, src)

        cropped_ice = icesat_crop.crop_df_by_latlon(ICE_track,[[tiff_lat_0,tiff_lat_1],[tiff_lon_0,tiff_lon_1]])
        ax[2].clear()
        ax[2].scatter(cropped_ice["x_atc"], cropped_ice["h_ph"],color='black',s=1,alpha=0.5)
        ax[2].set_ylim(10,20)
        if time_diff_ice <= 0 :
            ax[2].set_title(f'Time: -{timedelta(seconds=abs(time_diff_ice))}', fontsize=14, weight='bold', color='red')
        else:
            ax[2].set_title(f'Time: {timedelta(seconds=time_diff_ice)}', fontsize=14, weight='bold', color='green')
        plt.show()


    # Function to print current xlim and ylim of each subplot
    def cut(event):
        if check_coordinates(klicker_1.get_positions()) and check_coordinates(klicker_2.get_positions()):
            drift_vector.append(calculate_coordinate_drift(klicker_1.get_positions(),klicker_2.get_positions()))
            
            for i, ax_i in enumerate(ax):
                xlim = ax_i.get_xlim()
                ylim = ax_i.get_ylim()
                bbox = [xlim[0], ylim[0], xlim[1], ylim[1]]
                tiff_bbox.append(bbox)

            tiff_xlim = ax[0].get_xlim()
            tiff_ylim = ax[0].get_ylim()
            tiff_coords_0 = [tiff_xlim[0],tiff_ylim[0]]
            tiff_coords_1 = [tiff_xlim[1],tiff_ylim[1]]

            tiff_lat_0, tiff_lon_0 = transform_coords_to_latlon(tiff_coords_0, src)
            tiff_lat_1, tiff_lon_1 = transform_coords_to_latlon(tiff_coords_1, src)

            new_ice_track.append(icesat_crop.crop_df_by_latlon(ICE_track,[[tiff_lat_0,tiff_lat_1],[tiff_lon_0,tiff_lon_1]]))

            plt.close(fig)
        else:
            error_message_window("Wrong number of drift markers placed.")        

    # Create a button to print bbox
    button_ax_cut = fig.add_axes([0.85, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
    button_cut = Button(button_ax_cut, 'cut')
    button_cut.on_clicked(cut)

    ax[0].callbacks.connect('xlim_changed', auto_zoom_icesat)
    ax[0].callbacks.connect('ylim_changed', auto_zoom_icesat)

    plt.show()
    return tiff_bbox, drift_vector, new_ice_track

def sort_segment_list(segment_list):

    tiff_list = []
    csv_name = ""
    final_list = []

    for element in segment_list:
        if element.endswith(".tiff"):
            tiff_list.append(element)
        elif element.endswith(".csv"):
            csv_name = element
        
    if abs(datetime_diff(extract_datetime(csv_name), extract_datetime(tiff_list[0]))) > abs(datetime_diff(extract_datetime(csv_name), extract_datetime(tiff_list[1]))):
        final_list = [csv_name,tiff_list[1], tiff_list[0]]
    else:
        final_list = [csv_name,tiff_list[0], tiff_list[1]]

    return final_list



def inspect_segment(segment_list):

    path = segment_folder_path

    sorted_segment_list = sort_segment_list(segment_list)

    #naming the files
    tiff1_path = os.path.join(path,sorted_segment_list[1])
    print(tiff1_path)
    tiff2_path = os.path.join(path,sorted_segment_list[2])
    print(tiff2_path)
    csv_path = os.path.join(path,sorted_segment_list[0])
    ICESat_df = pd.read_csv(csv_path)

    #find time difference
    time_1_tiff = extract_datetime(sorted_segment_list[1])
    time_2_tiff = extract_datetime(sorted_segment_list[2])
    time_diff_sec = abs(datetime_diff(time_1_tiff,time_2_tiff))

    # Plot the images
    tiff_bbox, drift_vector, new_ice_df_list = plot_images(tiff1_path,tiff2_path,ICESat_df,csv_path)
    
    new_ice_df = new_ice_df_list[0]

    drift_xy_sec = average_drift(drift_vector, time_diff_sec)

    new_tiff_1 = sentinel_crop.crop_tiff_by_bbox(tiff1_path, tiff_bbox[0])

    #find the id for the new meltpond
    meltpond_id = find_meltpond_id()

    #save csv
    drift_file_path = os.path.join(storage_folder+"\\drift_values.csv")
    save_drift_csv(drift_file_path, meltpond_id, drift_xy_sec)

    #give_icesat_name
    icesat_name = f"{meltpond_id}_{sorted_segment_list[0]}"

    #save tiff
    icesat_crop.save_df_to_csv(os.path.join(storage_folder,icesat_name),new_ice_df)

    #Give new names for tiff
    tiff_1_name = transform_tiff_name(sorted_segment_list[1],meltpond_id)

    #save tiff
    sentinel_crop.save_tiff(new_tiff_1[0], new_tiff_1[1], os.path.join(storage_folder,tiff_1_name))
    print(f"Meltpond Saved as meltpond_{meltpond_id}")


#Pick the number and beam

selected_data = select_data("0","gt3rw")   

inspect_segment(selected_data)

