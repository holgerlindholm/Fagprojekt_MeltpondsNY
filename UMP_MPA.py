"""Script opening csv photon heights data, crop to size and get depths
Input: Beam data in csv file
Output: 
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import scipy.optimize
from matplotlib.widgets import Button
import rasterio
from rasterio.plot import show,adjust_band
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import polyfit
import shutil

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

def transform_coords_to_utm(df,crs):
    """Transform the latitude and longitude to the image coordinates"""
    lat = df["lat_ph"]
    lon = df["lon_ph"]
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    Ice_x,Ice_y = transformer.transform(lat, lon)
    return Ice_x,Ice_y

def calc_shgo_mode(data, distribution):
    """Calculate mode of the distribution using SHGO algorithm."""
    def objective(x):
        return 1 / distribution.pdf(x)[0]
    bnds = [[min(data), max(data)]]
    solution = scipy.optimize.shgo(objective, bounds=bnds)
    return solution.x[0]

def calculate_mode(data):
    """Calculate mode of the distribution 
        - input list of heights
        - output mode of gaussian distribution"""
    distribution = scipy.stats.gaussian_kde(data)
    shgo_mode = calc_shgo_mode(data, distribution)
    return shgo_mode

def refraction_correction(data):
    """Perform refraction correction"""
    n_air = 1.00029
    n_water = 1.34116
    return data * (n_air / n_water)

def trim_data_by_xatc(df, lower_xatc=None, upper_xatc=None):
    """Trim dataset based on along track distance (x_atc)"""
    mask = (df['x_atc'] >= lower_xatc) & (df['x_atc'] <= upper_xatc)
    return df[mask]

def trim_data_by_height(df,lower_height=None,upper_height=None):
    """Trim dataset based on height distance (h_ph)"""
    mask = (df['h_ph'] >= lower_height) & (df['h_ph'] <= upper_height)
    return df[mask]

def calculate_depths(df,bin_width=5,upper_height=-0.15,lower_height=-5):
    """Calculate depths"""
    min_x_atc = min(df["x_atc"])
    max_x_atc = max(df["x_atc"])

    # Arrays to store melt pond depths data
    depths_moving_avg = []
    lat_moving_avg = []
    lon_moving_avg = []
    x_atc_moving_avg = []
    depths_moving_avg_no_refraction = []

    for i in np.arange(min_x_atc, max_x_atc, bin_width):
        data = trim_data_by_xatc(df, i - 2 * bin_width, i + 3 * bin_width)
        data = trim_data_by_height(data, lower_height, upper_height)
    
        middle_dist = i+1/2*bin_width
        try:
            middle_index = (data['x_atc'] - middle_dist).abs().idxmin()
            lat_moving_avg.append(data['lat_ph'][middle_index])
            lon_moving_avg.append(data['lon_ph'][middle_index])
            avg_mode = calculate_mode(data['h_ph'])
            depths_moving_avg.append(refraction_correction(avg_mode))
            depths_moving_avg_no_refraction.append(avg_mode)
            x_atc_moving_avg.append(middle_dist)
        except ValueError:
            continue
    return [x_atc_moving_avg, lat_moving_avg, lon_moving_avg, depths_moving_avg, depths_moving_avg_no_refraction]

def cut_and_save(event,ax):
    print("Cutting data")
    xlim = ax.get_xlim()
    xmin = min(xlim)
    xmax = max(xlim)
    print(xlim)

    # Save depths to csv
    df_depths = pd.DataFrame({"x_atc": depths[0], "lat_ph": depths[1], "lon_ph": depths[2], "depth": depths[3]})
    df_depths = trim_data_by_xatc(df_depths,xmin,xmax)
    print(icesat_file)
    icesat_out_path = os.path.join(out_path,icesat_file)
    df_depths.to_csv(f"{icesat_out_path}_depths.csv", index=False)
    tiff_out_path = os.path.join(out_path,tiff_file)
    shutil.copy(tiff_path, tiff_out_path)
    
    # Verify the copy operation
    if os.path.exists(tiff_out_path):
        print(f"File copied successfully to {tiff_out_path}")
    else:
        print("File copy failed")

def finish():
    print("All files have been processed")
    print("Copying drift values to drift.csv")
    drift_path = os.path.join(path,"drift_values.csv")
    drift = pd.read_csv(drift_path)
    indices = [int(f.split("_")[0]) for f in os.listdir(out_path) if "depths" in f]
    print(indices)
    print(drift)
    filtered_drift = drift.loc[drift["Meltpond_ID"].isin(indices)]
    print(filtered_drift)
    filtered_drift.to_csv(os.path.join(out_path,"drift.csv"),index=False)
    print("Drift values copied to drift.csv")

def get_surface_height(df):
    """Get the surface height of the segment"""
    segment_ice_height = calculate_mode(df["h_ph"]) # Get mode surface height of whole segment
    df["h_ph"] -= segment_ice_height
    df = trim_data_by_height(df, -5, 5)
    segment_ice_height = calculate_mode(df["h_ph"]) # Avoid scattering from atmosphere
    df["h_ph"] -= segment_ice_height
    df = trim_data_by_height(df, -5, 5)
    return segment_ice_height,df

def get_zoomed_depths(event,ax,df):
    # Trim data by x_atc
    xlim = ax.get_xlim()
    xmin = min(xlim)
    xmax = max(xlim)
    df = trim_data_by_xatc(df, xmin, xmax)
    print(f"Getting zoomed depths for {xmin} to {xmax}")

    """Get the surface height of the segment"""
    segment_ice_height = calculate_mode(df["h_ph"]) # Get mode surface height of zoomed in segment
    print("Segment ice height:",segment_ice_height)

    depths = calculate_depths(df, bin_width,upper_height=-0.1)

    ax.hlines(segment_ice_height, min(df["x_atc"]), max(df["x_atc"]), color="orange")
    ax.scatter(depths[0], depths[4], c="red")
    ax.scatter(depths[0], depths[4], c="red",label="No refraction correction")
    ax.plot(depths[0], depths[3], c="blue")
    ax.scatter(depths[0], depths[3], c="blue",label="Refraction corrected")
    ax.legend()

############################################
"""MAIN PROGRAM STARTS HERE"""
############################################

data_folder = "20210706221959_floki/" # CHANGE ME!
index = 3 # CHANGE ME meltpond_id in folder!
path = os.path.join(os.getcwd(),"Detected_meltponds",data_folder)
out_path = os.path.join(os.getcwd(),"Detected_meltponds",data_folder,"depths")

# Get files in folder
Files = os.listdir(path)
icesat_files = [f for f in Files if ("ATL03" in f) and ("csv" in f) and ("depths" not in f)]
icesat_file = icesat_files[index]
icesat_path = os.path.join(path,icesat_file)
print(icesat_file)
fileidx = icesat_file.split("/")[-1].split("_")[0]
tiff_file = [f for f in Files if ("tiff" in f) and f.split("_")[0] == fileidx][0]
tiff_path = os.path.join(path,tiff_file)
df = pd.DataFrame(pd.read_csv(icesat_path)) # load icesat data

# Trim data by height
df = trim_data_by_height(df, 0, 30)
bin_width = 5 # meters

segment_ice_height,df = get_surface_height(df)

fig, (ax1,ax2) = plt.subplots(1,2) # Create figure

# Plot ICEsat data - along track distance vs photon height
ax1.scatter(df["x_atc"], df["h_ph"], c="black", alpha=0.5, s=1)
ax1.set(xlabel="Distance along track (m)", ylabel="Photon height (m)")
ax1.hlines(0, min(df["x_atc"]), max(df["x_atc"]), color="green")

# Create a button to print bbox
button_ax_cut = fig.add_axes([0.80, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_cut = Button(button_ax_cut, 'cut')
button_cut.on_clicked(lambda event: cut_and_save(event, ax1))

# Create a finish button
button_ax_finish = fig.add_axes([0.90, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_finish = Button(button_ax_finish, 'finish all')
button_finish.on_clicked(lambda event: finish())

# # Create a button to get surface height
button_ax_surface = fig.add_axes([0.05, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_surface = Button(button_ax_surface, 'get surface')
button_surface.on_clicked(lambda event: get_zoomed_depths(event,ax1,df))

# Plot Sentinel image for reference
img_1_RGB,transform_1,src = tiff_to_np_RGB(tiff_path)
ice_x,ice_y = transform_coords_to_utm(df,src)
ax2.scatter(ice_x,ice_y)
show(img_1_RGB,transform=transform_1,ax=ax2)
plt.show()




