"""Converts the bands of a Sentinel-2 image to a tiff file format."""
import rasterio
import os

def convert_to_tiff(file_path,filename,output_path=None):
    # Read the bands
    band2 = rasterio.open(f"{file_path}/{filename}_B02_10m.jp2", driver="JP2OpenJPEG")  # blue
    band3 = rasterio.open(f"{file_path}/{filename}_B03_10m.jp2", driver="JP2OpenJPEG")  # green
    band4 = rasterio.open(f"{file_path}/{filename}_B04_10m.jp2", driver="JP2OpenJPEG")  # red
    band8 = rasterio.open(f"{file_path}/{filename}_B08_10m.jp2", driver="JP2OpenJPEG")  # NIR
    NIR = band8.read(1)
    red = band4.read(1)
    green = band3.read(1)
    blue = band2.read(1)

    # Print some information about the bands
    print(band4.count)
    print(band4.width)
    print(band4.height)
    print(band4.dtypes[0])
    print(band4.crs)
    print(band4.transform)
    print(band4.bounds)

    # Convert the 4 bands to tiff format
    with rasterio.open(f'{output_path}.tiff', 'w', driver='GTiff', width=band4.width, height=band4.height, count=4, crs=band4.crs, transform=band4.transform, dtype=band4.dtypes[0]) as rgb:
        rgb.write(NIR, 4)
        rgb.write(red, 3)
        rgb.write(green, 2)
        rgb.write(blue, 1)

def main():
    filename = "T20XNS_20210625T201851"
    filepath = "C:/Users/holge\OneDrive - Danmarks Tekniske Universitet/30110 Fagprojekt/Getdata/Images"
    convert_to_tiff(filepath,filename)
    # filename = "T20XNS_20190622T185921" # Image from june22
    # file_path = os.path.join(os.getcwd(),"Images")
    # convert_to_tiff(file_path,filename)


if __name__ == "__main__":
    main()