from osgeo import gdal
import numpy as np
import pandas as pd

img_path = r'data\LANDSAT.tif'

ds = gdal.Open(img_path, gdal.GA_ReadOnly)

data = ds.ReadAsArray()

bands, rows, cols = data.shape

rsl = []
for i in range(rows):
    for j in range(cols):
        if data[0, i, j] > -9999:
            tmp = np.append([i, j], data[:, i, j])
            rsl.append(tmp)

df = pd.DataFrame(rsl, columns=['rows', 'cols', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
df.to_csv('map.csv', index=False)