# Import dependancies 
from rasterio.plot import show,adjust_band
import rasterio.windows as windows
import pandas as pd
import os
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np

# Load the ICESat data
# The ICEat data is stored in csv files with 5km segment for each beam
def crop_tiff_by_bbox(big_tiff_path,bbox):
    """Crop the Sentinel-2 image by the bounding box"""
    with rasterio.open(big_tiff_path) as src:
        # (left, bottom, right, top)
        window = windows.from_bounds(bbox[0],bbox[1],bbox[2],bbox[3],src.transform)
        # data = np.array([np.clip(src.read(i,window=window)/10000,0,1) for i in (3,2,1)])

        # Define the new transform
        new_transform = rasterio.windows.transform(window, src.transform)

        # Define the new metadata
        new_meta = src.meta.copy()
        new_meta.update({
            'height': window.height,
            'width': window.width,
            'transform': new_transform
        })

        new_data = src.read(window=window)
        
        # Check if the image is empty (all zeros)
        if sum([sum(row) for row in new_data[0]]) == 0:
            print("Empty image")
            return False
        
        return(new_data,new_meta)
    
def save_tiff(new_data,new_meta,out_tiff_path):
    """Write the cropped data to a new TIFF file"""
    with rasterio.open(f"{out_tiff_path}.tiff", 'w', **new_meta) as dst:
        dst.write(new_data)
        print(f"Saved cropped image to {out_tiff_path}_cropped.tiff")

def get_bbox_from_df(df,crs,extend_by=200):
    """Get the bounding box from the dataframe"""
    latmin = min(df["lat_ph"])
    latmax = max(df["lat_ph"])
    lonmin = min(df["lon_ph"])
    lonmax = max(df["lon_ph"])
    transformer = Transformer.from_crs('EPSG:4326',crs) 
    transformed_bbox = transformer.transform([latmin,latmax],[lonmin,lonmax])
    #print(transformed_bbox)

    # Extend the bounding box by a fixed amount (standard is 200m)
    # (left, bottom, right, top)
    transformed_bbox = [transformed_bbox[0][0]-extend_by,transformed_bbox[1][0]-extend_by,
                        transformed_bbox[0][1]+extend_by,transformed_bbox[1][1]+extend_by]
    return transformed_bbox

def tiff_to_np(Sentinel_name):
    """Opens tiff file and converts to np array"""
    with rasterio.open(f"{Sentinel_name}.tiff") as src:
        # # Convert tiff to np array and transform coordinates
        big_img = np.array([adjust_band(src.read(i)) for i in (3,2,1)])
        gain = 2 # Adjust the gain (brightness) of the image
        big_img = big_img * gain
    return big_img,src.transform,src.crs

def get_crs_from_tiff(tiff_path):
    """Get the crs from the tiff file"""
    with rasterio.open(tiff_path) as src:
        return src.crs
    
def main():
    # Define the path to the data
    S2_name_1 = "T20XNS_20190622T185921"
    S2_name_2 = "T20XNS_20190622T194909"
    ICESat_name = "ATL03_20190622202251_13070304_006_02"
    files = os.listdir(os.getcwd()+f"\segments_{ICESat_name}")
    print(f'Number of ICESat-2 segments: {len(files)}')

    # Get the crs (projection) from the big tiff file
    crs = get_crs_from_tiff(f"{S2_name_2}.tiff")
    print(crs)

    # # Plot the big image - for testing
    # fig, ax = plt.subplots()
    # img,transform_big,_ = tiff_to_np(S2_name_2)
    # show(img,transform=transform_big,ax=ax)

    for segment in files:
        print(segment)
        df = pd.read_csv(os.getcwd()+f"\segments_{ICESat_name}/{segment}")
        bbox = get_bbox_from_df(df,crs)
        # print(bbox)

        if crop_tiff_by_bbox(f"{S2_name_1}.tiff",bbox) == False or crop_tiff_by_bbox(f"{S2_name_2}.tiff",bbox) == False:
            pass
        else:
            # Save the cropped image to a new tiff file if it is not empty
            out_tiff_path_1 = f"{S2_name_1}_{segment.replace('.csv','')}"
            new_data_1,new_meta_1 = crop_tiff_by_bbox(f"{S2_name_1}.tiff",bbox)
            save_tiff(new_data_1,new_meta_1,out_tiff_path_1)
            out_tiff_path_2 = f"{S2_name_2}_{segment.replace('.csv','')}"
            new_data_2,new_meta_2 = crop_tiff_by_bbox(f"{S2_name_2}.tiff",bbox)
            save_tiff(new_data_2,new_meta_2,out_tiff_path_2)

            # Plot the cropped image - for testing
            # # Plot the cropped image - for testing
            # img,transform,_ = tiff_to_np(out_tiff_path_1)
            # fig, ax = plt.subplots()
            # show(img,transform=transform_big,ax=ax)
            # plt.show()

        print("Done")

if __name__ == "__main__":
    print("Running main")
    #main()
