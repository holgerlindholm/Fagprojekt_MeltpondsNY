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
def getExtremePoints(data, typeOfInflexion = None, maxPoints = None):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfInflexion = None returns all inflexion points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange ==1)[0]

    if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]
        
    elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif typeOfInflexion is not None:
        idx = idx[::2]
    
    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data)-1) in idx:
        idx = np.delete(idx, len(data)-1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx= idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))
    
    return idx

def tiff_to_np_RGB(Sentinel_name):
    """Opens tiff file and converts to np array 
    - NB: Only reads RGB bands (3,2,1) to show image"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        print(f" Shape of img: {src.read().shape}")
        # print(src.read(4))
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        #big_img = np.array(adjust_band(src.read(4))) # Check for NIR
        gain = 1 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

def auto_zoom_tiff(event):
    xlim = ax1.get_xlim()
    df_trimmed = trim_data_by_xatc(df, min(xlim), max(xlim))
    ice_x_trimmed,ice_y_trimmed = transform_coords_to_utm(df_trimmed,src)
    ax2.clear()
    ax2.scatter(ice_x_trimmed,ice_y_trimmed,label="No drift")
    show(img_1_RGB,transform=transform_1,ax=ax2)

    ice_x_drift = ice_x_trimmed + driftX*time_delta
    ice_y_drift = ice_y_trimmed + driftY*time_delta

    ax2.scatter(ice_x_drift,ice_y_drift,color="red",label="Drift")
    ax2.set_title(f"Time delta: {time_delta_min} minutes \nDrift: {round(driftX,5)}m/s,{round(driftY,5)}m/s")
    #ax2.legend()
    plt.draw()

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
    sd = []
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
            trimmed_depth_segment = trim_data_by_height(data,avg_mode-0.3,avg_mode+0.3)
            sd_segment = np.std(trimmed_depth_segment["h_ph"])
            sd.append(sd_segment) # Add standard deviation for each segment
            depths_moving_avg.append(refraction_correction(avg_mode))
            depths_moving_avg_no_refraction.append(avg_mode)
            x_atc_moving_avg.append(middle_dist)
        except ValueError:
            continue 
    
    df_depths = pd.DataFrame({"x_atc": x_atc_moving_avg, "lat_ph": lat_moving_avg, "lon_ph": lon_moving_avg, "depth": depths_moving_avg, "depth_no_refraction": depths_moving_avg_no_refraction,"sd":sd})
    return df_depths

def remove_outliers(df,threshold=2,neighborhood=5):
    """Remove outliers"""
    if len(df) < neighborhood:
        raise ValueError("List must contain at least 3 elements.")
    
    if neighborhood % 2 == 0:
        raise ValueError("Neighborhood must be an odd number.")

    # Calculate mean and standard deviation of neighbors
    data = np.array(df["depth"])
    sd_neighbors = []
    
    cleaned_data = data.copy().tolist()
    N = int((neighborhood-1)/2)
    for i in range(N,len(data)-N):
        # Calculate the mean and standard deviation of the neighboring points
        neighbors = data[i-N:i+N]
        mean_neighbors = np.mean(neighbors)
        std_neighbors = np.std(neighbors)
        sd_neighbors.append(std_neighbors)

        # Check if the point is an outlier
        if abs(data[i]-mean_neighbors) > threshold*std_neighbors: # eg. 2 standard deviations
            cleaned_data[i] = None # Remove the outlier
            print(f"Outlier removed at index {i} with value {data[i]}.")
        
    # Remove the None values
    cleaned_df = df[(df["depth"] == cleaned_data)]
    return cleaned_df, sd

def make_2d_hist(df,bin_width_x = 10,bin_width_y = 0.1):
    """Make 2D histogram"""
    x_bins = np.arange(min(df["x_atc"]), max(df["x_atc"]), bin_width_x)
    y_bins = np.arange(min(df["h_ph"]), max(df["h_ph"]), bin_width_y)
    depth = []
    x = []
    data = trim_data_by_height(df,min(df["x_atc"]),-2*sd)
    for i in x_bins:
        count_vertical = []
        for j in y_bins:
            data = trim_data_by_xatc(df, i - 2 * bin_width_x, i + 3 * bin_width_x)
            data = trim_data_by_height(data, j - 2 * bin_width_y, j + 3 * bin_width_y)
            count_vertical.append(len(data))
        idx = np.argmax(count_vertical)
        depth.append(y_bins[idx])
        x.append(i)
    return x,depth

def cut_and_save(event,ax,out_path,df):
    """Cut the data and save the depths to a csv file"""
    print("Cutting data")
    xlim = ax.get_xlim()
    xmin = min(xlim)
    xmax = max(xlim)
    print(xlim)

    if not os.path.exists(out_path):
        # Create the directory
        os.makedirs(out_path)
        print(f"Directory '{out_path}' created successfully.")
    else:
        print(f"Directory '{out_path}' already exists.")

    # Save depths to csv
    df_depths = trim_data_by_xatc(df,xmin,xmax)
    icesat_out_path = os.path.join(out_path,icesat_file)
    df_depths.to_csv(f"{icesat_out_path}_depths.csv", index=False)
    tiff_out_path = os.path.join(out_path,tiff_file)
    shutil.copy(tiff_path, tiff_out_path)
    
    # Verify the copy operation
    if os.path.exists(tiff_out_path):
        print(f"File copied successfully to {tiff_out_path}")
    else:
        print("File copy failed")
    
    print("Data cut and saved successfully")
    plt.close(fig)

def finish(path,out_path):
    """Finish the program and copy drift values to drift.csv"""
    if not os.path.exists(out_path):
        # Create the directory
        os.makedirs(out_path)
        print(f"Directory '{out_path}' created successfully.")
    else:
        print(f"Directory '{out_path}' already exists.")
        
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
    plt.close(fig)

############################################
"""MAIN PROGRAM STARTS HERE"""
############################################

# Path to folder
# path = "20210707T215049_floki" 25
path = "20190622202251_Holger_Tilling" # 3

# path = "20190805215948_Holger_Tilling" 5
# path = "20210624195859_Christian" 3
# path = "20210628193330__XNR_Holger" 5

# path = "20210628193330_floki" 10 stk ((1,5,6) og (2,7) og (3,8,9,10) er ens)
# path = "20210704232100_lower_Christian" 12 stk INGEN DRIFT?

# path = "20210704232129_upper_Christian_ikkesikker4bånd" # Har ikke 4 bånd LAV IGEN
# path = "20210706162901_ChristianT" 22 (LAV IGEN med 1sd) mega godt tile

# path = "20210706221959_floki" 9 stk # et par stykket med saturation - ikke medtget
# path = "20210706222524_Signe" 7 stk
# path = "20210707141741_Christian" 9 stk # meget af drift virker fucked
path = "20210707215049_Christian" # 17 stk: id 3 og 12 kan bruges som model - mega godt tile
# path = "20210710160901_Christian" # 21 stk En del gode i starten
# path = "20210711153911_Christian" # 4 stk: et par gode

index = 3 # CHANGE ME meltpond_id in folder!
num_sd = 1 # Default = 1
threshold = 2 # For removing outliers default = 2

root_path = "C:/Users/holge/OneDrive/Documents/GitHub/Fagprojekt_MeltpondsNY/Detected_meltponds/"
path = os.path.join(root_path,path)
# Get files in folder
FILES = os.listdir(path)
icesat_files = [f for f in FILES if ("ATL03" in f) and ("csv" in f) and ("depths" not in f)]
icesat_file = icesat_files[index]
icesat_path = os.path.join(path,icesat_file)
out_path = os.path.join(path,"depths")
icesat_time = extract_datetime(icesat_file)
fileidx = icesat_file.split("/")[-1].split("_")[0]
tiff_file = [f for f in FILES if ("tiff" in f) and f.split("_")[0] == fileidx][0]
tiff_path = os.path.join(path,tiff_file)
tiff_time = extract_datetime(tiff_file)

drift_df = pd.read_csv(os.path.join(path,"drift_values.csv"))
driftX = drift_df["xs_drift"].values[int(fileidx)-1]
driftY = drift_df["ys_drift"].values[int(fileidx)-1]
time_delta = datetime_diff(icesat_time,tiff_time)
time_delta_min = round(time_delta/60,2)

# Load data
df = pd.read_csv(icesat_path)
df = trim_data_by_height(df,lower_height=-10,upper_height=50)
# plt.scatter(df["x_atc"],df["h_ph"],color="black",alpha=0.5)
# plt.show()
avg_height,_,_ = calculate_mode(df["h_ph"])
#avg_height = 28.7
df = trim_data_by_height(df,lower_height=avg_height-5,upper_height=avg_height+2)
avg_height,x,pdf = calculate_mode(df["h_ph"])
#avg_height = 28.7
df_surface_approx = df[(df["h_ph"] >= avg_height-0.3) & (df["h_ph"] <= avg_height+0.3)].copy()

sd = np.std(df_surface_approx["h_ph"])
print(f"Surface sd: {sd}")
surface_mode,x,pdf = calculate_mode(df_surface_approx["h_ph"])

# Move data to ice surfaces
df.loc[:, "h_ph"] = df["h_ph"] - surface_mode
x = x - surface_mode
df_surface_approx.loc[:, "h_ph"] = df_surface_approx["h_ph"] - surface_mode

df_depths = calculate_depths(df,bin_width=5,max_height=-sd*num_sd)
df_cleaned, sd_depths = remove_outliers(df_depths,threshold=threshold,neighborhood=5)
new_index = np.linspace(0, len(df_cleaned) - 1, 2 * len(df_cleaned) - 1)
df_interpolated = df_cleaned.reindex(new_index)
df_interpolated = df_interpolated.interpolate(method="linear")

# Add uncertainties from surface and depth and correct for refraction
df_interpolated.loc[:,"sd"] = df_interpolated["sd"] + sd_depths
df_interpolated.loc[:,"sd"] = refraction_correction(df_interpolated["sd"])

X,D = make_2d_hist(df,bin_width_x = 10,bin_width_y = 0.1)
# plt.plot(df_interpolated["x_atc"],df_interpolated["sd"],c="red")
# plt.show()

###########################################
# PLOTS AND VISUALIZATION
###########################################
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

ax1.scatter(df["x_atc"],df["h_ph"],c="black",alpha=0.5)
ax1.scatter(df_surface_approx["x_atc"],df_surface_approx["h_ph"],c="red",alpha=0.5)
ax1.hlines(0, min(df_surface_approx["x_atc"]), max(df_surface_approx["x_atc"]), color="blue", label="Mode surface approx")
ax1.hlines(2*sd, min(df["x_atc"]), max(df["x_atc"]), color="black", label="Confidence interval")
ax1.hlines(-2*sd, min(df["x_atc"]), max(df["x_atc"]), color="black")

# No refraction
ax1.plot(df_cleaned["x_atc"],df_cleaned["depth"]+df_cleaned["sd"],c="orange")
ax1.plot(df_cleaned["x_atc"],df_cleaned["depth"]-df_cleaned["sd"],c="orange")
ax1.scatter(df_interpolated["x_atc"],df_interpolated["depth"],c="blue",label="Interpolate")
ax1.scatter(df_cleaned["x_atc"],df_cleaned["depth"],c="orange",label="Refraction")

# With refraction
ax1.scatter(df_interpolated["x_atc"],df_interpolated["depth_no_refraction"],c="red",label="No refraction")
ax1.set_title("Depth profile")
ax1.set(xlabel="Along track distance",ylabel="Height")
ax1.legend()

# Plot histogram for whole segment
ax3.hist(df["h_ph"],bins=30,alpha=0.4,color="blue",orientation="horizontal")
ax3.set(xlabel="Height",ylabel="Frequency")
ax31 = ax3.twiny()
_,x,pdf = calculate_mode(df["h_ph"],bw_method=0.2)
ax31.plot(pdf,x,c="red",label="PDF")
ax31.set(xlabel="PDF",ylabel="Height")
ax31.legend()

# Plot histogram for single segment
df_trim = trim_data_by_height(df,lower_height=-2,upper_height=-2*sd)
vertical_bins = np.arange(min(df_trim["h_ph"]),max(df_trim["h_ph"]),step=0.1)
horizontal_bins = np.arange(min(df_trim["x_atc"]),max(df_trim["x_atc"]),step = 10)
hist, _,_, image = ax4.hist2d(df_trim["x_atc"],df_trim["h_ph"],bins=[horizontal_bins,vertical_bins],cmap='viridis')
plt.colorbar(image)
ax4.set(xlabel="Along track distance",ylabel="Height")
ax4.set_title("Segment histogram")

# Plot Sentinel image for reference
img_1_RGB,transform_1,src = tiff_to_np_RGB(tiff_path)
ice_x,ice_y = transform_coords_to_utm(df,src)
ice_x_drift = ice_x + driftX*time_delta
ice_y_drift = ice_y + driftY*time_delta
ax2.scatter(ice_x,ice_y)
ax2.scatter(ice_x_drift,ice_y_drift,color="red")
ax2.set_title(f"Time delta: {time_delta_min} minutes \nDrift: {round(driftX,5)}m/s,{round(driftY,5)}m/s")
show(img_1_RGB,transform=transform_1,ax=ax2)

###########################################
# Connect events (buttons and callback)
###########################################

# Auto zoom tiff image based on change in ICEsat plot
ax1.callbacks.connect('xlim_changed', auto_zoom_tiff)
ax1.callbacks.connect('ylim_changed', auto_zoom_tiff)

# Create a button to print bbox
button_ax_cut = fig.add_axes([0.80, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_cut = Button(button_ax_cut, 'cut')
button_cut.on_clicked(lambda event: cut_and_save(event, ax1,out_path,df_interpolated))

# Create a finish button
button_ax_finish = fig.add_axes([0.90, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
button_finish = Button(button_ax_finish, 'finish all')
button_finish.on_clicked(lambda event: finish(path,out_path))

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
