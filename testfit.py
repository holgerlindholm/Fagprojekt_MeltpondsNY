import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scipy.optimize
from matplotlib.widgets import Button
import rasterio
from rasterio.plot import show,adjust_band
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import polyfit
import shutil
from datetime import datetime

"""
NEW DEPTH ALGORITHM
"""
def tiff_to_np_RGB(Sentinel_name):
    """Opens tiff file and converts to np array 
    - NB: Only reads RGB bands (3,2,1) to show image"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        print(src.read().shape)
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        gain = 1 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

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

def transform_coords_to_utm(df,crs):
    """Transform the latitude and longitude to the image coordinates"""
    lat = df["lat_ph"]
    lon = df["lon_ph"]
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    Ice_x,Ice_y = transformer.transform(lat, lon)
    return Ice_x,Ice_y
def trim_data_by_xatc(df, lower_xatc=None, upper_xatc=None):
    """Trim dataset based on along track distance (x_atc)"""
    mask = (df['x_atc'] >= lower_xatc) & (df['x_atc'] <= upper_xatc)
    return df[mask]

def trim_data_by_height(df,lower_height=None,upper_height=None):
    """Trim dataset based on height distance (h_ph)"""
    mask = (df['h_ph'] >= lower_height) & (df['h_ph'] <= upper_height)
    return df[mask]

def calculate_mode(data, bw_method=0.5):
    """Calculate mode"""
    kde = gaussian_kde(data, bw_method=bw_method)
    x = np.linspace(data.min(),data.max(),1000)
    pdf = kde(x)
    avg_mode = x[np.argmax(pdf)]
    return avg_mode,x,pdf

def refraction_correction(data):
    """Perform refraction correction"""
    n_air = 1.00029
    n_water = 1.34116
    return data * (n_air / n_water)

def calculate_depths(data,bin_width=5,max_height=-0.15):
    """Calculate depths"""
    min_x_atc = min(data["x_atc"])
    max_x_atc = max(data["x_atc"])

    # Arrays to store melt pond depths data
    depths_moving_avg = []
    lat_moving_avg = []
    lon_moving_avg = []
    x_atc_moving_avg = []
    depths_moving_avg_no_refraction = []

    for i in np.arange(min_x_atc, max_x_atc, bin_width):
        data = trim_data_by_xatc(df, i - 2 * bin_width, i + 3 * bin_width)
        data = data[(data["h_ph"] < max_height)]
    
        middle_dist = i+1/2*bin_width
        try:
            middle_index = (data['x_atc'] - middle_dist).abs().idxmin()
            lat_moving_avg.append(data['lat_ph'][middle_index])
            lon_moving_avg.append(data['lon_ph'][middle_index])
            avg_mode,_,_ = calculate_mode(data['h_ph'])
            #print(avg_mode)
            depths_moving_avg.append(refraction_correction(avg_mode))
            depths_moving_avg_no_refraction.append(avg_mode)
            x_atc_moving_avg.append(middle_dist)
        except ValueError:
            continue 
        
    return [x_atc_moving_avg, lat_moving_avg, lon_moving_avg, depths_moving_avg, depths_moving_avg_no_refraction]

path = "C:/Users/holge/OneDrive/Documents/GitHub/Fagprojekt_MeltpondsNY/Detected_meltponds/20210706162901_ChristianT"
index = 0 # CHANGE ME meltpond_id in folder!

# Get files in folder
FILES = os.listdir(path)
icesat_files = [f for f in FILES if ("ATL03" in f) and ("csv" in f) and ("depths" not in f)]
icesat_file = icesat_files[index]
icesat_path = os.path.join(path,icesat_file)
icesat_time = extract_datetime(icesat_file)
fileidx = icesat_file.split("/")[-1].split("_")[0]
print(icesat_file)
print(fileidx)
tiff_file = [f for f in FILES if ("tiff" in f) and f.split("_")[0] == fileidx][0]
tiff_path = os.path.join(path,tiff_file)
tiff_time = extract_datetime(tiff_file)

drift_df = pd.read_csv(os.path.join(path,"drift_values.csv"))
driftX = drift_df["xs_drift"].values[int(fileidx)-1]
driftY = drift_df["ys_drift"].values[int(fileidx)-1]
print(driftX,driftY)
time_delta = datetime_diff(icesat_time,tiff_time)
time_delta_min = round(time_delta/60,2)

# Load data
df = pd.read_csv(icesat_path)
df = trim_data_by_height(df,lower_height=-10,upper_height=50)
avg_height,_,_ = calculate_mode(df["h_ph"])
df = trim_data_by_height(df,lower_height=avg_height-5,upper_height=avg_height+5)
avg_height,_,_ = calculate_mode(df["h_ph"])

df_surface_approx = df[(df["h_ph"] >= avg_height-0.3) & (df["h_ph"] <= avg_height+0.3)]
sd = np.std(df_surface_approx["h_ph"])
surface_mode,x,pdf = calculate_mode(df_surface_approx["h_ph"])

# Move data to ice surfaces
df["h_ph"] = df["h_ph"] - surface_mode
x = x - surface_mode
df_surface_approx["h_ph"] = df_surface_approx["h_ph"] - surface_mode

depths = calculate_depths(df,bin_width=10,max_height=-2*sd)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.scatter(df["x_atc"],df["h_ph"],c="black",alpha=0.5)
ax1.scatter(df_surface_approx["x_atc"],df_surface_approx["h_ph"],c="red",alpha=0.5)
ax1.hlines(0, min(df_surface_approx["x_atc"]), max(df_surface_approx["x_atc"]), color="blue", label="Mode surface approx")
ax1.hlines(2*sd, min(df["x_atc"]), max(df["x_atc"]), color="black", label="Confidence interval")
ax1.hlines(-2*sd, min(df["x_atc"]), max(df["x_atc"]), color="black")

#ax1.scatter(depths[0],depths[3],c="red",label="Refraction")
ax1.scatter(depths[0],depths[4],c="orange",label="No refraction")
ax1.legend()

# Plot histogram for whole segment
ax3.hist(df["h_ph"],bins=30,alpha=0.4,color="blue")
ax3.set(xlabel="Height",ylabel="Frequency")
ax31 = ax3.twinx()
ax31.plot(x,pdf,c="red",label="PDF")
ax31.legend()

# Plot histogram for single segment
bin_width = 5
max_height = -2*sd
for i in np.arange(min(df["h_ph"]), max(df["h_ph"]), bin_width):
    print(i)
    if i == 5:
        data = trim_data_by_xatc(df, i - 2 * bin_width, i + 3 * bin_width)
        data = data[(data["h_ph"] < max_height)]
        print(data)
        surface_mode,x,pdf = calculate_mode(data["h_ph"])
        ax4.hist(df["h_ph"],bins=30,alpha=0.4,color="blue")
        ax4.plot(x,pdf,c="red")
    else:
        pass


# Plot Sentinel image for reference
img_1_RGB,transform_1,src = tiff_to_np_RGB(tiff_path)
ice_x,ice_y = transform_coords_to_utm(df,src)
ice_x_drift = ice_x + driftX*time_delta
ice_y_drift = ice_y + driftY*time_delta
ax2.scatter(ice_x,ice_y)
ax2.scatter(ice_x_drift,ice_y_drift,color="red")
ax2.set_title(f"Time delta: {time_delta_min} minutes \nDrift: {round(driftX,5)}m/s,{round(driftY,5)}m/s")
show(img_1_RGB,transform=transform_1,ax=ax2)
plt.show()
