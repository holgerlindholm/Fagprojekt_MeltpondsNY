import os
import pandas as pd
import matplotlib.pyplot as plt

# Import the cropping scripts from the other files
import Crop_ICESat2
import Crop_Sentinel2
import Convert_bands_to_tiff

"""Combined program for splitting up ICESat-2 and Sentinel images
- Place ICESat-2 h5 data and 2x sentinel-2 jp2 bands in a folder"""

"""DEFINE INPUT FOLDER AND MAIN SENTINEL FILE"""
input_folder = "C:/Users/signe/OneDrive/Dokumenter/4. semester/Fagprojekt/tile_07_06_2021"
Sentinel_path = "T13XDK_20210706T222059.tiff" # Main Sentinel file

input_files = os.listdir(input_folder)
sentinel_files = set([x[:-12] for x in input_files if x.endswith(".jp2")])
icesat_file = [x for x in input_files if x.endswith(".h5")]

# Convert bands to tiff and save them in the input folder
for file in sentinel_files:
    output_path = os.path.join(input_folder,file)
    print(output_path)
    Convert_bands_to_tiff.convert_to_tiff(input_folder,file,output_path=output_path)
    print(f"Converted {file} to tiff format")

# Crop the ICESat-2 data and save it to csv in the input folder
Sentinel_path = os.path.join(input_folder,Sentinel_path)
ICESat_path = os.path.join(input_folder,icesat_file[0])
print(ICESat_path.split("\\")[-1])
Crop_ICESat2.h5_to_csv(ICESat_path,Sentinel_path,input_folder)
print(f"Converted ICESat h5 file to csv format")

# Crop Sentinel tiffs by ICESat-2 segments
ICESat_segments =  [x for x in os.listdir(input_folder) if x.endswith(".csv")] #get all the segments
Tiff_files = [x for x in os.listdir(input_folder) if x.endswith("tiff")] #get all the tiff files
S2_name_1 = os.path.join(input_folder,Tiff_files[0])
S2_name_2 = os.path.join(input_folder,Tiff_files[1])
print(S2_name_1,S2_name_2)
crs = Crop_Sentinel2.get_crs_from_tiff(S2_name_1) # Get the crs (projection) from the big tiff file

for segment in ICESat_segments:
    segment_path = os.path.join(input_folder,segment)
    df = pd.read_csv(segment_path) # Read the segment
    bbox = Crop_Sentinel2.get_bbox_from_df(df,crs) # Get the bbox from the dataframe
    # print(bbox)

    if Crop_Sentinel2.crop_tiff_by_bbox(S2_name_1,bbox) == False or Crop_Sentinel2.crop_tiff_by_bbox(S2_name_2,bbox) == False:
        # If the cropped image is empty, pass
        pass
    else:
        # Save the cropped image to a new tiff file if it is not empty
        out_tiff_path_1 = f"{S2_name_1.replace('.tiff','')}_{segment.replace('.csv','')}"
        new_data_1,new_meta_1 = Crop_Sentinel2.crop_tiff_by_bbox(S2_name_1,bbox)
        Crop_Sentinel2.save_tiff(new_data_1,new_meta_1,out_tiff_path_1)
        out_tiff_path_2 = f"{S2_name_2.replace('.tiff','')}_{segment.replace('.csv','')}"
        new_data_2,new_meta_2 = Crop_Sentinel2.crop_tiff_by_bbox(S2_name_2,bbox)
        Crop_Sentinel2.save_tiff(new_data_2,new_meta_2,out_tiff_path_2)

        # # Plot the cropped image - for testing
        # img,transform,_ = tiff_to_np(out_tiff_path_1)
        # fig, ax = plt.subplots()
        # show(img,transform=transform_big,ax=ax)
        # plt.show()

print(f"Saved cropped tiff files to inputfolder")

## Delete csv files not corresponding to tiff files
tiff_segments = [x.replace(".tiff","").split("_")[-2:] for x in os.listdir(input_folder) if (x.endswith(".tiff") and x not in [S2_name_1,S2_name_2])]
csv_segments = [x for x in os.listdir(input_folder) if x.endswith(".csv")]
print(tiff_segments[:5])
print(csv_segments[:5])
for csv in csv_segments:
    csv_beam = csv.replace(".csv","").split("_")[-2:]
    if csv_beam not in tiff_segments:
        os.remove(os.path.join(input_folder,csv))
        print(f"Deleted {csv}")

print("Done")



