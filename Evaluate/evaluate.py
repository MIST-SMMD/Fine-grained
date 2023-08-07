import pandas as pd
import numpy as np
from geopy.distance import geodesic

def split_coordinates(coord_str):
    x = float(coord_str.split(',')[0])
    y = float(coord_str.split(',')[1])
    return x, y
# 读取Excel文件
df = pd.read_excel(r'C:\Users\LENOVO\Desktop\result.xlsx').head(1)

# 分离经纬度坐标
df[['cor_loc_x', 'cor_loc_y']] = df['cor_loc'].apply(split_coordinates).apply(pd.Series)
df[['[1]fin_loc_x', '[1]fin_loc_y']] = df['[1]fin_loc'].apply(split_coordinates).apply(pd.Series)
df[['[12]fin_loc_x', '[12]fin_loc_y']] = df['[12]fin_loc'].apply(split_coordinates).apply(pd.Series)
df[['[13]fin_loc_x', '[13]fin_loc_y']] = df['[13]fin_loc'].apply(split_coordinates).apply(pd.Series)
df[['[123]fin_loc_x', '[123]fin_loc_y']] = df['[123]fin_loc'].apply(split_coordinates).apply(pd.Series)
df[['true_loc_x', 'true_loc_y']] = df['true_loc'].apply(split_coordinates).apply(pd.Series)

print(df['cor_loc_y'])
# 计算Cor_Space_Error和Fin_Space_Error
df['Cor_Space_Error'] = df.apply(lambda row: geodesic((row['cor_loc_y'], row['cor_loc_x']), (row['true_loc_y'], row['true_loc_x'])).m, axis=1)
df['[1]Fin_Space_Error'] = df.apply(lambda row: geodesic((row['[1]fin_loc_y'], row['[1]fin_loc_x']), (row['true_loc_y'], row['true_loc_x'])).m, axis=1)
df['[12]Fin_Space_Error'] = df.apply(lambda row: geodesic((row['[12]fin_loc_y'], row['[12]fin_loc_x']), (row['true_loc_y'], row['true_loc_x'])).m, axis=1)
df['[13]Fin_Space_Error'] = df.apply(lambda row: geodesic((row['[13]fin_loc_y'], row['[13]fin_loc_x']), (row['true_loc_y'], row['true_loc_x'])).m, axis=1)
df['[123]Fin_Space_Error'] = df.apply(lambda row: geodesic((row['[123]fin_loc_y'], row['[123]fin_loc_x']), (row['true_loc_y'], row['true_loc_x'])).m, axis=1)



# 计算MAE和RMSE
cor_mae = df['Cor_Space_Error'].mean()
cor_rmse = np.sqrt((df['Cor_Space_Error']**2).mean())

fin_mae_1 = df['[1]Fin_Space_Error'].mean()
fin_rmse_1 = np.sqrt((df['[1]Fin_Space_Error']**2).mean())
fin_mae_12 = df['[12]Fin_Space_Error'].mean()
fin_rmse_12 = np.sqrt((df['[12]Fin_Space_Error']**2).mean())
fin_mae_13 = df['[13]Fin_Space_Error'].mean()
fin_rmse_13 = np.sqrt((df['[13]Fin_Space_Error']**2).mean())
fin_mae_123 = df['[123]Fin_Space_Error'].mean()
fin_rmse_123 = np.sqrt((df['[123]Fin_Space_Error']**2).mean())

# 输出结果
print(f"Cor_Space_Error MAE: {cor_mae}")
print(f"Cor_Space_Error RMSE: {cor_rmse}")

print(f"[1]Fin_Space_Error MAE: {fin_mae_1}")
print(f"[1]Fin_Space_Error RMSE: {fin_rmse_1}")

print(f"[12]Fin_Space_Error MAE: {fin_mae_12}")
print(f"[12]Fin_Space_Error RMSE: {fin_rmse_12}")

print(f"[13]Fin_Space_Error MAE: {fin_mae_13}")
print(f"[13]Fin_Space_Error RMSE: {fin_rmse_13}")

print(f"[123]Fin_Space_Error MAE: {fin_mae_123}")
print(f"[123]Fin_Space_Error RMSE: {fin_rmse_123}")
