# Import Libraries
import os
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt

PATH = 'Datasets\\Excel\\'
OUT = 'Datasets\\Full\\'
FILE_PATH = os.listdir(PATH)
MAP_PATH = r'GEE-DEMO\LANDSAT8_4_7_band_new_Projec21.tif'

ds = gdal.Open(MAP_PATH, gdal.GA_ReadOnly)

# ulx, uly: up left (x,y)
# xres, yres: x,y scale

ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
# lrx = ulx + (ds.RasterXSize * xres)
# lry = uly + (ds.RasterYSize * yres)

# Read Excel File
for file in FILE_PATH:
    df = pd.read_excel(PATH + file, engine='openpyxl')

    x = df['X']
    y = df['Y']

    # Get Bands Value
    for k in range(ds.RasterCount):
        band_k = ds.GetRasterBand(k + 1)
        band_k = band_k.ReadAsArray()
        b_k = []
        for i in range(len(x)):
            pixel_x = (x[i] - ulx) / xres
            pixel_y = (y[i] - uly) / yres
            pixel_x = int(pixel_x)
            pixel_y = int(pixel_y)

            val = band_k[pixel_y][pixel_x]

            b_k.append(val)

        df_b = pd.DataFrame(b_k)
        df['Band_{}'.format(k + 1)] = df_b

    # Export file
    df.to_excel(OUT + file.split('.')[0] + '_7_Bands.xlsx', index=False)
