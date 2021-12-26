from osgeo import gdal
import numpy as np
import pandas as pd

data_path = r'data\data.xlsx'

img_path = r'data\LANDSAT.tif'

map_path = r'map.csv'

df = pd.read_excel(data_path)
map_df = pd.read_csv(map_path)
X_labels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
y_label = 'label'

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(max_features='auto', n_estimators=100, random_state=9, criterion='entropy')
clf.fit(df[X_labels], df[y_label])

map_df['pred'] = clf.predict(map_df[X_labels])
map_df = map_df[['rows', 'cols', 'pred']]
print(map_df)

ds = gdal.Open(img_path, gdal.GA_ReadOnly)
data = ds.ReadAsArray()
bands, rows, cols = data.shape

output_map = np.zeros(shape=(rows, cols), dtype=np.float32) + 255
for lines in map_df.values:
    i, j, pred = lines
    output_map[int(i)][int(j)] = pred


outfname ='LCmap.output.tif'
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(outfname, cols, rows, 1, gdal.GDT_Float32)
dst_ds.SetGeoTransform(ds.GetGeoTransform())
dst_ds.SetProjection(ds.GetProjection())
band = dst_ds.GetRasterBand(1)
band.SetNoDataValue(255)
band.WriteArray(output_map)
dst_ds = None
