"""This script loads h5 files, crops them by tiff coordinates and saves each beam to csv files
It further splits the beams into segments of 5000m each and saves them to csv files
Input: ICESat_path, Sentinel_path
Output: Saves cropped beam data to csv files as
    - ICESatname_beam_index.csv
"""

import os
import pandas as pd
import rasterio
from pyproj import Transformer

# Reader files from https://github.com/ICESAT-2HackWeek/Clouds_and_data_filtering/
from readers.read_HDF5_ATL03 import read_HDF5_ATL03
from readers.get_ATL03_x_atc import get_ATL03_x_atc

def load_h5_file(file_path, beam='gt3l'):
    """Load .h5 file using the ATL03 reader and convert to df"""
    try:
        ATL03_file = file_path
        # Get beam orientation 
        IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = read_HDF5_ATL03(ATL03_file, track=beam)
        # Based on orientation get the strong beams
        if IS2_atl03_mds["orbit_info"]["sc_orient"][0] == 0:
            beam_name = ['gt1rs', "gt2rs", "gt3rs",'gt1lw', "gt2lw", "gt3lw"] #R = Strong Beams
        elif IS2_atl03_mds["orbit_info"]["sc_orient"][0] == 1:
            beam_name = ['gt1rw', "gt2rw", "gt3rw",'gt1ls', "gt2ls", "gt3ls"] #L = Strong Beams
        
        beams = ['gt1r', "gt2r", "gt3r",'gt1l', "gt2l", "gt3l"] #all beams
        #beams = ["gt1r"]
        dfs = []
        for i,beam in enumerate(beams):
            IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams = read_HDF5_ATL03(ATL03_file, track=beam)
            get_ATL03_x_atc(IS2_atl03_mds, IS2_atl03_attrs, IS2_atl03_beams)
            df = pd.DataFrame(IS2_atl03_mds[beam])
            dfs.append([df,beam_name[i]])
        return dfs
    except FileNotFoundError:
        print("File not found:", file_path)
        return None
    
def convert_df_to_heights(df):
    """Convert big df to df_heights (only relevant photon data)"""
    df = df["heights"][:15].T.reset_index()
    DF = pd.DataFrame(df.iloc[4].values[1],columns=['lat_ph'])
    DF["delta_time"] = df.iloc[0].values[1]
    DF["dist_ph_across"] = df.iloc[1].values[1]
    DF["dist_ph_along"] = df.iloc[2].values[1]
    DF['h_ph'] = df.iloc[3].values[1]
    DF['lon_ph'] = df.iloc[5].values[1]
    DF["ph_id_channel"] = df.iloc[7].values[1]
    DF["ph_id_count"] = df.iloc[8].values[1]
    DF["ph_id_pulse"] = df.iloc[9].values[1]
    DF["quality"] = df.iloc[10].values[1]
    # DF["signal_conf_ph"] = df.iloc[11].values[1]
    DF["weight_ph"] = df.iloc[12].values[1]
    DF['x_atc'] = df.iloc[13].values[1]
    DF['y_atc'] = df.iloc[14].values[1]
    # print(df.iloc[11].values[1])
    return(DF)

def crop_df_by_latlon(DF,Bbox):
    """crop the heights_dataframe to BBox"""
    lat = Bbox[0]
    lon = Bbox[1]
    latmin = min(lat)
    latmax = max(lat)
    lonmin = min(lon)
    lonmax = max(lon)

    mask = ((DF["lat_ph"] > latmin) & (DF["lat_ph"] < latmax)) & ((DF["lon_ph"] > lonmin) & (DF["lon_ph"] < lonmax))
    DF = DF[mask]
    return(DF)

def split_into_segments(df, segment_length=5000):
    """Split the DataFrame into segments of a fixed length and return a list of DataFrames"""
    min_x = df["x_atc"].min()
    max_x = df["x_atc"].max()
    segments = []

    for segment in range(int(min_x), int(max_x), segment_length):
        segment = df[(df["x_atc"] >= segment) & (df["x_atc"] < segment + segment_length)]
        if not segment.empty:
            segments.append(segment)
    return segments

def get_bbox_by_img(img_path):
    """Get_latlon from tiff file"""
    with rasterio.open(img_path) as data:
        bbox = data.bounds # In UTM
        transformer = Transformer.from_crs(data.crs,'EPSG:4326') 
        transformed_bbox = transformer.transform([bbox[0],bbox[2]],[bbox[1],bbox[3]])
        print(f"Coordinates: {transformed_bbox}")
        return transformed_bbox

def save_df_to_csv(out_file_name,df):
    """Save cropped heights dataframe to csv"""
    df.to_csv(out_file_name,index=False)
    print(f"Saved dataframe to {out_file_name}")

def h5_to_csv(ICESat_path,Sentinel_path,out_path):
    """Combined script cropping and saving h5 to csv"""
    Bbox = get_bbox_by_img(Sentinel_path) # Get Bbox from sentinel in [[lat1,lat2],[lon1,lon2]]
    all_beam_data = load_h5_file(ICESat_path) # Returns list of beams [[data,beam],[data,beam]...]
    ICE_name = ICESat_path.split("\\")[-1].replace(".h5","")
    print(ICE_name)
    for beam_data in all_beam_data:
        df_heights = convert_df_to_heights(beam_data[0]) # Extracts beam from .h5 to heights dataframe
        beam_name = beam_data[1] # Get beam name
        cropped_df = crop_df_by_latlon(df_heights,Bbox=Bbox) # Crop the dataframe by Bbox from image
        # out_file_name = f"{ICE_name}_{beam_name}.csv" 
        # save_df_to_csv(out_file_name,cropped_df)
        beam_segments = split_into_segments(cropped_df, segment_length=5000) # Split the dataframe into segments
        for i,segment in enumerate(beam_segments):
            out_file_name = f"{ICE_name}_{beam_name}_{i}.csv"
            out_file_path = os.path.join(out_path,out_file_name)
            save_df_to_csv(out_file_path,segment)
        
if __name__ == "__main__":
    # Define the paths
    ICE_path = "ATL03_20190622202251_13070304_006_02.h5"
    Sentinel_file = "T20XNS_20190622T194909.tiff"
    Sentinel_path = os.path.join(os.getcwd(),Sentinel_file)
    h5_to_csv(ICE_path,Sentinel_path,out_path=None)


