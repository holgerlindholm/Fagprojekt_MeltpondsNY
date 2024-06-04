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

def calculate_depths(df):
    """Calculate depths"""
    bin_width = 5  # meters
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
        data = trim_data_by_height(data, -5, -0.15)
    
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
    df_depths.to_csv(f"{filename}_depths.csv", index=False)

# Test data: 
path = "C:/Users/holge/OneDrive - Danmarks Tekniske Universitet/GitHub/Fagprojekt-Meltponds/meltponds/"
Files = os.listdir(path)
print(Files)
files = [os.path.join(path,f) for f in Files if ("ATL03" in f) and ("depths" not in f)]
filename = files[1] #CHANGE ME!
fileidx = filename.split("/")[-1].split("_")[0]
print(fileidx)
tiff_path = [os.path.join(path,f) for f in Files if ("tiff" in f) and f.split("_")[0] == fileidx][0]
print(tiff_path)
df = pd.DataFrame(pd.read_csv(filename))

# Trim data by height
df = trim_data_by_height(df, 10, 30)

# Get approximate mode of the distribution ie. surface height
segment_ice_height = calculate_mode(df["h_ph"]) # Get mode surface height of whole segment
df = trim_data_by_height(df, segment_ice_height-5, segment_ice_height+5)
df["h_ph"] -= segment_ice_height
print(segment_ice_height)
# Calculate depths
depths = calculate_depths(df)

# Plot depths
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.scatter(df["x_atc"], df["h_ph"], c="black", alpha=0.5, s=1)
ax1.plot(depths[0], depths[4], c="red")
ax1.plot(depths[0],depths[3],c="blue")
ax1.legend(["photon","No refraction correction","Refraction correction"])
ax1.set(xlabel="Distance along track (m)", ylabel="Photon height (m)")
ax1.hlines(0, min(df["x_atc"]), max(df["x_atc"]), color="green")

# Create a button to print bbox
button_ax_cut = fig.add_axes([0.85, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_cut = Button(button_ax_cut, 'cut')
button_cut.on_clicked(lambda event: cut_and_save(event, ax1))

def tiff_to_np_RGB(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        print(src.read().shape)
        #big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        big_img = np.array(adjust_band(src.read(4)))
        gain = 1.5 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

img_1_RGB,transform_1,src = tiff_to_np_RGB(tiff_path)

ice_x,ice_y = transform_coords_to_utm(df,src)
ax2.scatter(ice_x,ice_y)
show(img_1_RGB,transform=transform_1,ax=ax2)

plt.show()



