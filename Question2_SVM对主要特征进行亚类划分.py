import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
file_path = '附件.xlsx'
sheet_name = '表单2_version2'  # 根据你的数据表单名称进行调整

# 使用pandas读取指定表单
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 删除第一列和表面风化特征
data = data.drop(columns=[data.columns[0], '表面风化'])

# 对高钾和铅钡进行分别处理
data_high_potassium = data[data['类型'] == '高钾'][['二氧化硅(SiO2)', '氧化钾(K2O)']]
data_lead_barium = data[data['类型'] == '铅钡'][['二氧化硅(SiO2)', '氧化钡(BaO)']]

# 填充缺失值为0
data_high_potassium = data_high_potassium.fillna(0)
data_lead_barium = data_lead_barium.fillna(0)

# 标准化数据
scaler_high_potassium = StandardScaler()
data_high_potassium_scaled = scaler_high_potassium.fit_transform(data_high_potassium)

scaler_lead_barium = StandardScaler()
data_lead_barium_scaled = scaler_lead_barium.fit_transform(data_lead_barium)

# 进行层次聚类
linkage_type = 'ward'

# 高钾类型矿石的层次聚类
hclust_high_potassium = linkage(data_high_potassium_scaled, method=linkage_type)
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram (High Potassium)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(hclust_high_potassium)
plt.show()

# 使用fcluster划分为两类
labels_high_potassium = fcluster(hclust_high_potassium, 2, criterion='maxclust')

# 显示划分结果
plt.scatter(data_high_potassium.iloc[:, 0], data_high_potassium.iloc[:, 1], c=labels_high_potassium)
plt.title('Clustered High Potassium Data')
plt.xlabel('SiO2')
plt.ylabel('K2O')
plt.show()

# 铅钡类型矿石的层次聚类
hclust_lead_barium = linkage(data_lead_barium_scaled, method=linkage_type)
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram (Lead Barium)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(hclust_lead_barium)
plt.show()

# 使用fcluster划分为三类
labels_lead_barium = fcluster(hclust_lead_barium, 3, criterion='maxclust')

# 显示划分结果
plt.scatter(data_lead_barium.iloc[:, 0], data_lead_barium.iloc[:, 1], c=labels_lead_barium)
plt.title('Clustered Lead Barium Data')
plt.xlabel('SiO2')
plt.ylabel('BaO')
plt.show()
