
import numpy as np
import rasterio
from skimage import io
from skimage.util import img_as_float
from rasterio.plot import show,adjust_band
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.morphology import erosion, opening
from skimage.morphology import disk

path = "C:/Users/signe/OneDrive/Dokumenter/4. semester/Fagprojekt/tile_07_06_2021/"
img_name = "T13XDK_20210706T213041_ATL03_20210706222524_02031204_006_01_gt1rw_16.tiff" 

def tiff_to_np(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([src.read(i) for i in (4,3,2,1)])
        #big_img = np.array(src.read(4))
    return big_img,src.transform,src.crs

def tiff_to_RGB(Sentinel_name):
    """Opens tiff file and converts to np array with only adjusted RGB bands"""
    with rasterio.open(Sentinel_name) as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        #np.array([sr.read(1),src-read(2],srac.read(3])
        gain = 1.2 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img

def NDWI_calc(image):
    """Calculates normalized difference water index"""
    # the two necessary bands
    g = img[2,:,:]
    NIR = img[0,:,:]
    
    NDWI = (g-NIR)/(g+NIR)
    
    return NDWI

def water_identifier(image, NDWI_image):
    """ Identifies water pixels from threshold"""
    #creating mask
    thres = threshold_otsu(NDWI_image)
    water_mask = NDWI > thres
    
    #water pixelvalues
    water = image[:,water_mask]
    
    return water, water_mask

def area_sorting(mask, min_area):
    """ Performing BLOB analysis and sorting out meltponds with 
    too small area """
    
    label_img = measure.label(mask)
    region_props = measure.regionprops(label_img)
    
    label_img_sorted = label_img
    for region in region_props:
    	# Find the areas that do not fit our criteria and set to 0
    	if region.area < min_area:
    		for cords in region.coords:
    			label_img_sorted[cords[0], cords[1]] = 0
    
    sorted_bin_img = label_img_sorted > 0
    
    return label_img, sorted_bin_img

img, transformation, crs = tiff_to_np(path + img_name)
img_to_show = tiff_to_RGB(path + img_name)
NDWI = NDWI_calc(img)
water, water_mask = water_identifier(img, NDWI)

fig1,(axes) = plt.subplots(2,2)
ax1, ax2, ax3, ax4 = axes.flatten()
show(img_to_show,transform=transformation, ax=ax1)
ax2.imshow(NDWI)
ax3.hist(NDWI.ravel(), bins=256)

ax4.imshow(water_mask, cmap=plt.cm.gray)
plt.show()

pond_img = img_to_show
pond_img[1,water_mask] = 0

fig2,(ax1,ax2) = plt.subplots(1,2)
ax1.imshow(water_mask, cmap=plt.cm.gray)
show(pond_img, transform=transformation, ax=ax2)
plt.show()

fig3, ax1 = plt.subplots(1,1)
ax1.hist(water[3,:], bins=256)
plt.show()

pond_mask = water_mask
min_area = 10

label_img, sorted_pond_mask = area_sorting(pond_mask, min_area)
region_props = measure.regionprops(label_img)
meltpond_areas = np.array([prop.area for prop in region_props])
print(meltpond_areas)

img_to_show = tiff_to_RGB(path + img_name)
fig4, (ax1, ax2, ax3) = plt.subplots(1,3)
show(img_to_show, transform=transformation, ax=ax1, title='Meltpond image')
ax2.imshow(pond_mask, cmap='gray')
ax2.set_title('Unsorted meltpond mask')
ax3.imshow(sorted_pond_mask, cmap='gray')
ax3.set_title('Sorted meltpond mask')
plt.show()

#Finding the values of the meltpond pixels
meltpond_pixels = img[:,sorted_pond_mask]


