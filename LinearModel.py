import os
import numpy as np
import rasterio
from rasterio.plot import show,adjust_band
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_otsu
from pyproj import Transformer

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

def tiff_to_np(Sentinel_name):
    """Opens tiff file and converts to np array 
    - NB: Only reads RGB bands (3,2,1) to show image"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        print(f" Shape of img: {src.read().shape}")
        # print(src.read(4))
        big_img = np.array([src.read(i) for i in (4,3,2,1)])
        gain = 1 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

def transform_coords_to_utm(df,crs):
    """Transform the latitude and longitude to the image coordinates"""
    lat = df["Latitude"]
    lon = df["Longtitude"]
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    Ice_x,Ice_y = transformer.transform(lat, lon)
    return Ice_x,Ice_y

# Specific depth model
def get_depth(b02,b03,b04,b08):
    depth = -3.1780290 +-0.0004022*b02+ 0.0008491*b03+  0.0003796*b04-0.0006735*b08
    return depth

# General linear model (N=0)
def get_depth_general(b02,b03,b04,b08):
    depth = -1.299e+00  -1.907e-04 *b02+ -1.810e-05*b03+    7.769e-04*b04 -5.583e-04*b08
    sd = [7.031e-02,3.113e-05,5.187e-05,4.066e-05,2.711e-05]
    errors = sd[0] + sd[1]*b02 + sd[2]*b03 + sd[3]*b04 + sd[4]*b08
    return depth,errors

# Linear model with 8 neighbors
def depth_nabo(B02_blue, B03_green, B04_red, B08_NIR, Blue_sum, Green_sum, Red_sum, NIR_sum):
    # Coefficients extracted from R model
    coefficients = {
        "Intercept": -1.299324e+00,
        "B02_blue": 1.870378e-05,
        "B03_green": -1.681697e-04,
        "B04_red": 1.840830e-04,
        "B08_NIR": -4.293757e-05,
        "Blue_sum": -2.082624e-04,
        "Green_sum": 9.737181e-05,
        "Red_sum": 7.239482e-04,
        "NIR_sum": -6.051438e-04
    }
    
    # Compute the predicted value based on the coefficients
    predicted_value = (
        coefficients["Intercept"] +
        coefficients["B02_blue"] * B02_blue +
        coefficients["B03_green"] * B03_green +
        coefficients["B04_red"] * B04_red +
        coefficients["B08_NIR"] * B08_NIR +
        coefficients["Blue_sum"] * Blue_sum +
        coefficients["Green_sum"] * Green_sum +
        coefficients["Red_sum"] * Red_sum +
        coefficients["NIR_sum"] * NIR_sum
    )
    
    return predicted_value

def water_identifier(image, input_image):
    """ Identifies water pixels from threshold"""
    #creating mask
    thres = threshold_otsu(input_image)
    water_mask = input_image > thres
    
    #water pixelvalues
    water = image[:,water_mask]
    
    return water, water_mask

def sum_surrounding_pixels(image):
    # Convert image to numpy array if it's not already
    image = np.array(image)

    # Get dimensions of the image
    height, width,bands = image.shape

    # Initialize an array to store results
    summed_image = np.zeros((height, width, bands), dtype=np.float32)

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Initialize sum for each band
            sum_neighbors = np.zeros(bands, dtype=np.float32)

            # Iterate over 8-neighborhood pixels
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue  # Skip the center pixel itself

                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        sum_neighbors += image[ny, nx]

            # Store the sum in the result array
            summed_image[y, x] = sum_neighbors/8
    return summed_image

file = "Detected_meltponds/20210710160901_Christian/17_T27XWM_20210710T160901_gt3rw.tiff"
path = "C:/Users/holge/OneDrive/Documents/GitHub/Fagprojekt_MeltpondsNY"
mp_path = os.path.join(path,file)

data = "Pixel_med_2naboer.csv"
df = pd.read_csv(os.path.join(path,data))
df = df[(df["Meltpond_index"]==128)]

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))
img,transform,crs = tiff_to_np_RGB(mp_path)
show(img,transform=transform,ax=ax1)
x,y = transform_coords_to_utm(df,crs)
ax1.scatter(x,y,c=df["Depth[m]"],cmap="viridis",label="True depth")

# Get the depth of the meltponds
image,_,_ = tiff_to_np(mp_path)
nabo = sum_surrounding_pixels(image)


b04 = image[1]
b03 = image[2]
b02 = image[3]
b08 = image[0]

depth = get_depth(b02,b03,b04,b08)
depth = np.array(depth)
water, water_mask = water_identifier(image,b08)
depth_image = np.zeros(depth.shape)
depth_image[~water_mask] = depth[~water_mask]
depth_image[water_mask] = 0

depth_general,errors = get_depth_general(b02,b03,b04,b08)
depth_general = np.array(depth_general)
depth_image_general = np.zeros(depth.shape)
depth_image_general[~water_mask] = depth_general[~water_mask]
depth_image_general[water_mask] = 0

depth_neighbour = depth_nabo(b02,b03,b04,b08,nabo[3],nabo[2],nabo[1],nabo[0])
depth_neighbour = np.array(depth_neighbour)
depth_image_neighbour = np.zeros(depth.shape)
depth_image_neighbour[~water_mask] = depth_neighbour[~water_mask]
depth_image_neighbour[water_mask] = 0

vmax = 0
vmin = min(depth_image.min(),depth_general.min())

volumne_specific = np.sum((abs(depth_image[~water_mask]))*10*10)
print(f"Volume of water: {volumne_specific}")
volumne_general = np.sum(abs(depth_image_general[~water_mask]))*10*10
vol_general_error = (np.sum(errors[~water_mask])*10*10)
vol_general_error1 = np.sqrt(np.sum(((errors[~water_mask])*10*10)**2))
print(f"Volume of water: {volumne_general,vol_general_error1}")

# cax2 = ax2.imshow(depth_image,vmin=vmin, vmax=vmax,cmap="viridis")
# ax2.title.set_text("Depth of meltponds specfic model")
# ax2.text(0.5, -0.10, f"Vol of water: {round(volumne_specific,1)} m3", ha='center', va='center', transform=ax2.transAxes)
# ax2.text(0.5, -0.15, f"R2: 0.78, STD: 0.11", ha='center', va='center', transform=ax2.transAxes)

volumne_N8 = np.sum(abs(depth_image_neighbour[~water_mask]))*10*10
print(f"Volume of water: {volumne_N8}")

cax2 = ax2.imshow(depth_image_neighbour,vmin=vmin,vmax=vmax,cmap="viridis")
ax2.title.set_text("Depth of meltponds N=8 model")
ax2.text(0.5, -0.10, f"Vol of water: {round(volumne_N8,1)} m3", ha='center', va='center', transform=ax2.transAxes)
ax2.text(0.5, -0.15, f"R2: 0.55, STD: 0.15", ha='center', va='center', transform=ax2.transAxes)


cax3 = ax3.imshow(depth_image_general,vmin=vmin,vmax=vmax,cmap="viridis")
ax3.title.set_text("Depth of meltponds N=0")
ax3.text(0.5, -0.10, f"Vol of water: {round(volumne_general,1)} m3", ha='center', va='center', transform=ax3.transAxes)
ax2.text(0.5, -0.15, f"R2: 0.51, STD: 0.15", ha='center', va='center', transform=ax3.transAxes)



# Add a colorbar that applies to both subplots
cbar = fig.colorbar(cax3, ax=[ax2, ax3], orientation='vertical')
cbar.set_label('Depth (m)')

plt.show()

#################

Pre_depth = get_depth(df["b02_blue"].values,df["b03_green"].values,df["b04_red"].values,df["b08_NIR"].values)
Pre_depth_general = get_depth_general(df["b02_blue"].values,df["b03_green"].values,df["b04_red"].values,df["b08_NIR"].values)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,5))
ax1.set_title("True vs predicted depth")
ax1.plot(df["idx"],df["Depth[m]"],label="True depth")
ax1.plot(df["idx"],Pre_depth,label="Predicted depth")
ax1.plot(df["idx"],Pre_depth_general,label="Predicted depth general")
ax1.legend()

X = df["Depth[m]"].to_numpy()
Y = Pre_depth
reg = LinearRegression()
reg.fit(X.reshape(-1,1),Y.reshape(-1,1))
depth_pred_linear = reg.predict(X.reshape(-1,1))
print(f"R2 score: {reg.score(X.reshape(-1,1),Y.reshape(-1,1))}")

ax2.scatter(df["Depth[m]"],Pre_depth)
ax2.plot(X,depth_pred_linear)
plt.plot(X,X)
ax2.set_xlabel("True depth")    
ax2.set_ylabel("Predicted depth")
plt.title("True vs predicted depth")

plt.show()
