#create a function
import os
import tifffile as tiff
from skimage import measure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from LinearModel import get_depth_general
from LinearModel import depth_nabo
from LinearModel import sum_surrounding_pixels
from LinearModel import tiff_to_np_RGB
from rasterio.plot import show
from skimage.filters import threshold_otsu
from predict_depth import predict_depth as predict_depth_ml

#path="C:\\Users\\chrel\\Documents\\Fagprojekt_Lokal\\Fagprojekt-Meltponds-master\\Tile_06_07_1629_Rigtig\\T27XWM_20210706T162901_ATL03_20210706160815_01991204_006_01_gt1rw_7.tiff"
#path="C:\\Users\\chrel\Documents\\Fagprojekt_Lokal\\Fagprojekt-Meltponds-master\\Tile_04_07_nedre\\T11XMK_20210704T232100_ATL03_20210704231644_01731204_006_01_gt3ls_13.tiff"
#path="C:\\Users\\chrel\\Documents\\Fagprojekt_Lokal\\Fagprojekt-Meltponds-master\\Tile_07_07_2150\\T14XMQ_20210707T215049_ATL03_20210707215945_02181204_006_01_gt1rw_7.tiff"
#path= "C:\\Users\\chrel\\Documents\\Fagprojekt_Lokal\\Fagprojekt-Meltponds-master\\Tile_10_07_1609\\T27XWM_20210710T160901_ATL03_20210710155956_02601204_006_01_gt3rw_0.tiff"

#img = tiff.imread(path)

def tiff_to_meltpond_mask(tiff_path,ocean_meltpond_threshold,min_area):
    img = tiff.imread(tiff_path)
    #Infrared mask water/Ice mask.
    NDWI = (img[:,:,1]-img[:,:,3])/(img[:,:,1]+img[:,:,3])
    water_ice_threshold = threshold_otsu(NDWI)
    #plt.imshow(NDWI)
    
    #water_mask= img[:,:,3]<water_ice_threshold
    water_mask = NDWI>water_ice_threshold
    label_img = measure.label(water_mask)
    region_props = measure.regionprops(label_img)

    label_img_sorted = label_img

    for region in region_props:
        # Find the areas that do not fit our criteria and set to 0
        
        if region.area < min_area:
            for cords in region.coords:
                label_img_sorted[cords[0], cords[1]] = 0
    
    #now sort ocean_meltpond_threshold
    ponds = np.unique(label_img_sorted)
    Meltpond_filter=[]
    Mean_pixel_ponds=[]
    Histogram=[]
    for pond in ponds:
        #create a mask for each pond
        if pond == 0:
            continue
        pond_mask= label_img_sorted == pond
        #reshape mask
        pond_mask =np.reshape(pond_mask,(img.shape[0],img.shape[1]))
        #apply mask to image
        
        pond_img = img[pond_mask]
        #Alle pixels i hver meltpond er nu i pond_img
        #Make a mean of the pixels in the pond
        Pixel_val = pond_img.flatten()
        Pixel_mean = np.mean(Pixel_val)
        Mean_pixel_ponds.append(Pixel_mean)
         # meltpond ocean discrimination
        if Pixel_mean > ocean_meltpond_threshold:
            Meltpond_filter.append(pond_mask)
    
        Flad=len(pond_img.flatten())/10
    
    
        for i in range(math.floor(Flad)):
            Histogram.append(Pixel_mean)
            
    #combine each pond_mask in Meltpond_filter to one single mask
    combined_mask = np.zeros_like(Meltpond_filter[0])
    for pond_mask in Meltpond_filter:
        # Combine the current pond_mask with the combined_mask using a logical OR operation
        combined_mask = np.logical_or(combined_mask, pond_mask)

    # If needed, convert the combined_mask back to the same dtype as the original masks
    # For example, if the original masks are of dtype 'uint8', convert the combined_mask to 'uint8'
    combined_mask = combined_mask.astype(np.uint8) 
       
    return combined_mask,Mean_pixel_ponds,Histogram,NDWI
 
    
def mask_to_df(path,combined_mask):    
    DATA=[]
    img=tiff.imread(path)
    print(img.shape)
    combined_mask = combined_mask.astype(bool)
    for i in range(1, combined_mask.shape[0]-1):
        for j in range(1, combined_mask.shape[1]-1):
            if combined_mask[i][j] == True:
                DATA.append(img[i][j])
                DATA.append(img[i][j+1])
                DATA.append(img[i][j-1])
                DATA.append(img[i+1][j])
                DATA.append(img[i-1][j])
                DATA.append(img[i+1][j+1])
                DATA.append(img[i-1][j-1])
                DATA.append(img[i+1][j-1])
                DATA.append(img[i-1][j+1])
                #The index is [0,1,2,3,4,5,6,7,8]=["mid","right","left","down","up","downright","upleft","downleft","upright"]
                #Band [0,1,2,3]=[B02,B03,B04,B08]
            else:
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
                DATA.append([0,0,0,0])
        
    DATA=np.array(DATA)   
    DATA=np.reshape(DATA, [combined_mask.shape[0]-2, combined_mask.shape[1]-2,9,4])
    return DATA


def machine_learning_9(DATA):
    #create a dataframe from DATA array
    
    df_9P=pd.DataFrame({
    "B08_mid":DATA[:,:,0,3].flatten(),
    "B04_mid":DATA[:,:,0,2].flatten(),
    "B03_mid":DATA[:,:,0,1].flatten(),
    "B02_mid":DATA[:,:,0,0].flatten(),
    "B08_left":DATA[:,:,2,3].flatten(),
    "B04_left":DATA[:,:,2,2].flatten(),
    "B03_left":DATA[:,:,2,1].flatten(),
    "B02_left":DATA[:,:,2,0].flatten(),
    "B08_right":DATA[:,:,1,3].flatten(),
    "B04_right":DATA[:,:,1,2].flatten(),
    "B03_right":DATA[:,:,1,1].flatten(),
    "B02_right":DATA[:,:,1,0].flatten(),
    "B08_up":DATA[:,:,4,3].flatten(),
    "B04_up":DATA[:,:,4,2].flatten(),
    "B03_up":DATA[:,:,4,1].flatten(),
    "B02_up":DATA[:,:,4,0].flatten(),
    "B08_down":DATA[:,:,3,3].flatten(),
    "B04_down":DATA[:,:,3,2].flatten(),
    "B03_down":DATA[:,:,3,1].flatten(),
    "B02_down":DATA[:,:,3,0].flatten(),
    "B08_upleft":DATA[:,:,6,3].flatten(),
    "B04_upleft":DATA[:,:,6,2].flatten(),
    "B03_upleft":DATA[:,:,6,1].flatten(),
    "B02_upleft":DATA[:,:,6,0].flatten(),
    "B08_upright":DATA[:,:,5,3].flatten(),
    "B04_upright":DATA[:,:,5,2].flatten(),
    "B03_upright":DATA[:,:,5,1].flatten(),
    "B02_upright":DATA[:,:,5,0].flatten(),
    "B08_downleft":DATA[:,:,8,3].flatten(),
    "B04_downleft":DATA[:,:,8,2].flatten(),
    "B03_downleft":DATA[:,:,8,1].flatten(),
    "B02_downleft":DATA[:,:,8,0].flatten(),
    "B08_downright":DATA[:,:,7,3].flatten(),
    "B04_downright":DATA[:,:,7,2].flatten(),
    "B03_downright":DATA[:,:,7,1].flatten(),
    "B02_downright":DATA[:,:,7,0].flatten()
    })
    return df_9P

def machine_learning_1(DATA):
    #create a dataframe from DATA array
    
    df_1P=pd.DataFrame({
    "B08_mid":DATA[:,:,0,3].flatten(),
    "B04_mid":DATA[:,:,0,2].flatten(),
    "B03_mid":DATA[:,:,0,1].flatten(),
    "B02_mid":DATA[:,:,0,0].flatten()
    })
    return df_1P

def calculate_volume(blob_size):
    mask,mea,his= tiff_to_meltpond_mask(path,3300,blob_size)
    mask = mask.astype(bool)
    DATA=mask_to_df(path,mask)
    
    # Linear 0 neighbors
    depth_linear_N0 ,errorsN0 = np.array(get_depth_general(img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3]))
    depth_image_N0 = np.zeros(depth_linear_N0.shape)
    print(depth_linear_N0.shape)
    depth_image_N0[mask] = depth_linear_N0[mask]
    depth_image_N0[~mask] = 0
    print(depth_image_N0.shape)

    # Linear 8 neighbors
    depth_N8 = np.array(sum_surrounding_pixels(img))
    print(f"N8{depth_N8.shape}")
    depth_linear_N8 = np.array(depth_nabo(img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3],depth_N8[:,:,0],depth_N8[:,:,1],depth_N8[:,:,2],depth_N8[:,:,3]))
    depth_image_N8 = np.zeros(depth_linear_N8.shape)
    depth_image_N8[mask] = depth_linear_N8[mask]
    depth_image_N8[~mask] = 0
    
    # AI models
    # ml_uden=predict_depth_ml(machine_learning_1(DATA),1)
    ml_model=predict_depth_ml(machine_learning_9(DATA),9)
    ml_img_9=np.reshape(ml_model,(mask.shape[0]-2,mask.shape[1]-2))
    # ml_img_1=np.reshape(ml_uden,(mask.shape[0]-2,mask.shape[1]-2))
    
    # UDREGNING AF VOLUMNE
    # 0 neighbors
    volume_N0 = np.sum(abs(depth_image_N0)*10*10)
    print(f"Volume N0: {volume_N0}")
    Verror=np.sqrt(np.sum((errorsN0[mask]*10*10)**2))
    print(f"Volume N0 errors: {Verror}")
    # 8 neighbors
    volume_N8 = np.sum(abs(depth_image_N8)*10*10)
    print(f"Volume N8: {volume_N8}")
    voulme_ml_9 = np.sum(abs(ml_img_9)*10*10)
    print(f"Volume ML 9: {voulme_ml_9}")
    # voulme_ml_1 = np.sum(abs(ml_img_1)*10*10)
    # print(f"Volume ML 1: {voulme_ml_1}")
    
    return volume_N0,volume_N8,voulme_ml_9

"""VOLUMNE calculation
L0 = []
L8 = []
AI8 = []
for i in range(0,30,5):
    print(f"Blob size: {i}")
    l0,l8,ai9 = calculate_volume(i)
    L0.append(l0)
    L8.append(l8)
    AI8.append(ai9)

fig,ax = plt.subplots()
ax.plot(L0,label="Linear 0")
ax.plot(L8,label="Linear 8")
ax.plot(AI8,label="AI 9")
ax.legend()
plt.show()
"""


"""# PLOT ALL Modelheigts
# RGB
RGBimg,transform,crs = tiff_to_np_RGB(path)

mask,mea,his= tiff_to_meltpond_mask(path,3300,10)
mask = mask.astype(bool)
DATA=mask_to_df(path,mask)

# Linear 0 neighbors
depth_linear_N0 ,errorsN0 = np.array(get_depth_general(img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3]))
depth_image_N0 = np.zeros(depth_linear_N0.shape)
print(depth_linear_N0.shape)
depth_image_N0[mask] = depth_linear_N0[mask]
depth_image_N0[~mask] = 0
print(depth_image_N0.shape)

# Linear 8 neighbors
depth_N8 = np.array(sum_surrounding_pixels(img))
print(f"N8{depth_N8.shape}")
depth_linear_N8 = np.array(depth_nabo(img[:,:,0],img[:,:,1],img[:,:,2],img[:,:,3],depth_N8[:,:,0],depth_N8[:,:,1],depth_N8[:,:,2],depth_N8[:,:,3]))
depth_image_N8 = np.zeros(depth_linear_N8.shape)
depth_image_N8[mask] = depth_linear_N8[mask]
depth_image_N8[~mask] = 0

# AI models
ml_uden=predict_depth_ml(machine_learning_1(DATA),1)
ml_model=predict_depth_ml(machine_learning_9(DATA),9)
ml_img_9=np.reshape(ml_model,(mask.shape[0]-2,mask.shape[1]-2))
ml_img_1=np.reshape(ml_uden,(mask.shape[0]-2,mask.shape[1]-2))

# calculate volume, remeber to remove [:120,:] for full image!!!
# 0 neighbors
volume_N0 = np.sum(abs(depth_image_N0[:120,:])*10*10)
print(f"Volume N0: {volume_N0}")
Verror=np.sqrt(np.sum((errorsN0[mask]*10*10)**2))
print(f"Volume N0 errors: {Verror}")
# 8 neighbors
volume_N8 = np.sum(abs(depth_image_N8[:120,:])*10*10)
print(f"Volume N8: {volume_N8}")
voulme_ml_9 = np.sum(abs(ml_img_9[:120,:])*10*10)
print(f"Volume ML 9: {voulme_ml_9}")
voulme_ml_1 = np.sum(abs(ml_img_1[:120,:])*10*10)
print(f"Volume ML 1: {voulme_ml_1}")



# # PLOT ALL

fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(2, 2)
vmax = 0
vmin = min(depth_image_N0.min(),ml_img_9.min(),ml_img_1.min())

show(RGBimg[:120,:120], ax=ax1)
ax1.set_title('RGB Image')

cax2 = ax2.imshow(depth_image_N0[:120,:],vmin=vmin,vmax=vmax,cmap='viridis',label="Linear 0")
ax2.set_title('Linear 0')

ax4.imshow(ml_img_9[:120,:],vmin=vmin,vmax=vmax,cmap='viridis',label="AI 9")
ax4.set_title('AI 9')

ax5.imshow(ml_img_1[:120,:],vmin=vmin,vmax=vmax,cmap='viridis',label="AI 1")
ax5.set_title('AI 1')

cbar = fig.colorbar(cax2,ax=[ax2,ax4,ax5])
cbar.set_label('Depth (m)')

plt.show()

"""

#Plot rgb,ndwi, histogram and meltpond mask

path="C:\\Users\\chrel\Documents\\Fagprojekt_Lokal\\Fagprojekt-Meltponds-master\\Tile_04_07_nedre\\T11XMK_20210704T232100_ATL03_20210704231644_01731204_006_01_gt3ls_15.tiff"

RGBimg,transform,crs = tiff_to_np_RGB(path)
Mask,Mean_p,Histogram,ndwi = tiff_to_meltpond_mask(path,3500,10)
Mask=Mask.astype(bool)

fig, (fax1, fax2,fax4) = plt.subplots(1, 3)
show(RGBimg,ax=fax1)
fax1.set_title('RGB Image')

#remove extreme values from NDWI
ndwi = np.where(ndwi > 2, 0, ndwi)


# Display NDWI with correct color mapping
ndwi_display = fax2.imshow(ndwi, cmap='viridis')
fax2.set_title('NDWI')
# Set colorbar for NDWI correctly
#cbar = fig.colorbar(ndwi_display, ax=fax2)

#fax3.hist(Histogram,bins=50)
#fax3.set_title('Histogram')

print(RGBimg.shape)

#apply mask to RGB image
RGBimg_transposed = np.transpose(RGBimg, (1, 2, 0))
Meltponds=RGBimg_transposed.copy()
Meltponds[~Mask]=0
Meltpond_pic=np.transpose(Meltponds, (2, 0, 1))

show(Meltpond_pic,ax=fax4)
fax4.set_title('Meltpond mask')
plt.show()


    
    