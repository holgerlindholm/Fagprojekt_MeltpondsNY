# Fagprojekt_MeltpondsNY
New repository for Student Project - Meltpond depth detection using ICESat-2 and Sentinel 2

## Find overlapping data: 
Step 1: Download ICESat-2 and Sentinel-2 flight data:
- ICESat: Reference Ground Track (RGT) KML files from: https://icesat-2.gsfc.nasa.gov/science/specs
- Sentinel 2 opendata catalogue: https://csv.dataspace.copernicus.eu/Sentinel-2/

Step 2: Download shapefiles of Greenland and Canada - used to sort out areas over land. 
Step 3: Run NEW_FindOverlap to get overlap between Sentinel-2 and ICESat-2
Result is stored in: Overlap_20min_sommer19-23.xlsx

## Download data
Step 1: Find overlapping tile from the excel sheet: 
Step 2: Find the Sentinel-2 tile on https://browser.dataspace.copernicus.eu/. Discard tiles with a high Cloud Coverage or w
Step 3: Download 2 x Sentinel 2 110x110 km tiles from COPERNICUS
- Extract the B02, B03, B04 and B08 10m resolution JPEG2000 files and place in a local folder
Step 4: Download ICESat-2 ATL03 V006 Granula (.h5 file): Find granula on https://search.earthdata.nasa.gov/search - filename must match filename from excel sheet.
- Place .h5 file in same folder as JPEG2000 files

## Cut data
Step 1: Use crop_all.py
- Define input folder with 8 x JPEG2000 files and 1 x .h5 file
- Define which Sentinel tile is closest to the ICESat time.

The extracts all ICESat-2 beams from h5 to csv and cuts them in to 5km segments. Weak/Strong beams are determined based on spacecraft orientation. 
Subsequently, the JPEG2000 files are converted to a (4 x 110.000 x 110.000) tiff file, which are cropped to size according to the boundary box of the ICESat-2 subsegments. If a cropped tiff file from either tiles is empty, the corresponding ICESat segment and both tiff files are deleted. 

## Look for Meltponds
Step 1: Open Plot_Sentinel2_and_ICESat2
Step 2: Determine input folder and output folder
Step 3: Run script on a specific beam: eg. selected_data = select_data("10","gt1ls")
Step 4: Find Meltpond: 
- Zoom in on meltpond on both Sentinel 2 images
- Drift correction: Use pointers to mark identical spots on the two images
- Click crop --> This crops the original tiff file and the icesat file and adds a drift vector to a csv file.

## UMP-MPA
Step 1: Determine Meltpond depths using the UMP-MPA algotithm: UMP_MPA.py
Step 2: Tweak the vertical crop to find the correct icesurface: 
- df = trim_data_by_height(df, 10, 30)
Step 3: Tweak the bin width to be able to track the subsurface of the meltpond:
- bin_width = 5  (eg 2m, 5m or 10m)
Step 4: Crop ICESat-2 data to fit Meltpond and click on the crop button.

## Drift correction and color correlation: 
HOLGERSIGNENY.py




