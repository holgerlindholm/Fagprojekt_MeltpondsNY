import rasterio
from rasterio.plot import show, adjust_band
import numpy as np
import matplotlib.pyplot as plt


def tiff_to_np_RGB(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # Convert tiff to np array and transform coordinates
        big_img = np.array([adjust_band(src.read(i)) for i in (3, 2, 1)])
        gain = 1  # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img, src.transform, src.crs

# Function to open the TIFF file and convert it to a numpy array
def tiff_to_np(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # Read the bands in order: red (band 3), green (band 2), blue (band 1), and NIR (band 4)
        big_img = np.array([src.read(i) for i in (3, 2, 1, 4)])
    return big_img, src.transform, src.crs

# Function that calculates the depth of a pixel using the statistical model
def calculate_new_depth(img):
    band1 = img[0]  # Red
    band2 = img[1]  # Green
    band3 = img[2]  # Blue
    band4 = img[3]  # NIR
    
    # Calculate the depth (NOT WORKING, VALUES ARE WAY TO HIGH)
    depth=(-2.481)+(1.264e-04*band1)+(9.387e-04*band2)+(-5.464e-04*band3)+(-3.618e-04*band4)
    
    return depth

# Function to create a new image shaded depending on depth
def create_depth_image(img, depth_value):
    band1 = img[0]  # Red
    band2 = img[1]  # Green
    band3 = img[2]  # Blue
    band4 = img[3]  # NIR
    
    # Normalize the depth_value to the range 0-1 for visualization
    min_val = np.min(depth_value)
    max_val = np.max(depth_value)
    normalized_value = (depth_value - min_val) / (max_val - min_val)
    
    # Create an RGB image
    depth_img = np.zeros((depth_value.shape[0], depth_value.shape[1], 3), dtype=np.float32)
    
    # Create a mask that is only applied to the pixels that are below a NIR value - better way surely
    mask = band4 < 4800
    depth_img[mask, 0] = 0  # Red channel
    depth_img[mask, 1] = 0  # Green channel
    depth_img[mask, 2] = normalized_value[mask]  # Blue channel
    
    # Set white color where the condition is not met
    depth_img[~mask] = 1  # White color
    
    #Area is the sum of all masked pixels times the area
    area = np.sum(mask) * (10*10)

    #Array containing depth of each pixel
    depth_pixel_values = depth_value[mask]

    print("Pixel depths:")
    print(depth_pixel_values)

    #Volume is calcualted with depth*area
    volume = 0

    for depth in depth_pixel_values:
        volume = volume + depth * (10*10) 

    return depth_img, area, abs(volume)

# Directory of the TIFF file containing meltpond
sentinel_file_dir = "C:\\Users\\35466\\Desktop\\Python Projects\\BPNN_test\\23_mp.tiff"

# Load the image and metadata
img, transform, crs = tiff_to_np(sentinel_file_dir)

# Calculate the new color value for each pixel
depth_value = calculate_new_depth(img)

# Create the conditional red shades image
depth_img, estimated_area, estimated_volume = create_depth_image(img, depth_value)

print("Total")
print("------------------")
print(f"AREA: {estimated_area}m^2")
print(f"VOLUME: {estimated_volume}m^3")
print("------------------")

# Plot the original and new images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Original image (using RGB bands)
img_1_RGB, transform_1, src = tiff_to_np_RGB(sentinel_file_dir)
show(img_1_RGB, transform=transform_1, ax=ax1)
ax1.set_title('Original Image')

# Show the depth image
ax2.imshow(depth_img)
ax2.set_title('Depth Image')

plt.show()
