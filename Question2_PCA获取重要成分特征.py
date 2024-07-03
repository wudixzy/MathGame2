import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 读取数据
file_path = '附件.xlsx'
sheet_name = '表单2_version2'  # 根据你的数据表单名称进行调整

# 使用pandas读取指定表单
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 删除第一列
data = data.drop(columns=data.columns[0])

# 删除表面风化特征
data = data.drop(columns=['表面风化'])

# 定义一个SimpleImputer实例来填充NaN值
imputer = SimpleImputer(strategy='constant', fill_value=0)

# 填充缺失值
data_imputed = imputer.fit_transform(data)

# 创建一个新DataFrame以包含填充后的数据
data_filled = pd.DataFrame(data_imputed, columns=data.columns)

# 对目标特征类型进行标签编码
label_encoder = LabelEncoder()
data_filled['类型'] = label_encoder.fit_transform(data_filled['类型'])

# 获取高钾和铅钡类型的数据
high_potassium_data = data_filled[data_filled['类型'] == label_encoder.transform(['高钾'])[0]]
lead_barium_data = data_filled[data_filled['类型'] == label_encoder.transform(['铅钡'])[0]]


# 定义一个函数来计算PCA并输出结果
def calculate_pca(data, title):
    X = data.drop(columns=['类型'])

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    pca_components = pca.components_

    # 获取前两个成分的特征
    pca_features = [X.columns[np.argmax(np.abs(component))] for component in pca_components]
    print(f"{title} - PCA selected features: {pca_features}")


# 计算高钾矿石的PCA结果
calculate_pca(high_potassium_data, 'High Potassium')

# 计算铅钡矿石的PCA结果
calculate_pca(lead_barium_data, 'Lead Barium')
