import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '附件.xlsx'
sheet_name = '表单1'  # 指定表单名称

# 使用pandas读取指定表单
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 移除非特征列 '文物编号'
data = data.drop(columns=data.columns[0])

# 将类别变量转换为数值
data_numeric = data.apply(lambda x: pd.factorize(x)[0])

# 特征名替换字典
feature_translation = {
    '纹饰': 'Pattern',
    '类型': 'Type',
    '颜色': 'Color',
    '表面风化': 'Surface Weathering'
}

# 使用替换后的英文特征名
data_numeric.columns = [feature_translation[col] for col in data_numeric.columns]

# 数据预处理函数
def data_preprocessing(df):
    return (df - df.min()) / (df.max() - df.min())

# 计算关联系数函数
def relation(mother, compare, p=0.5):
    distance = abs(mother - compare)
    max_distance = max(distance)
    min_distance = min(distance)
    relation = (min_distance + p * max_distance) / (distance + p * max_distance)
    relation_rate = sum(relation) / len(relation)
    return relation_rate

# 初始化关联矩阵
features = data_numeric.columns
correlation_matrix = pd.DataFrame(index=features, columns=features)

# 计算关联度
for feature1 in features:
    for feature2 in features:
        if feature1 != feature2:
            mother = data_preprocessing(data_numeric[feature1])
            compare = data_preprocessing(data_numeric[feature2])
            relation_rate = relation(mother, compare)
            correlation_matrix.loc[feature1, feature2] = relation_rate
        else:
            correlation_matrix.loc[feature1, feature2] = 1

# 确保关联矩阵中的数据类型为浮点型
correlation_matrix = correlation_matrix.fillna(0).astype(float)

# 显示结果
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title('Grey Relational Analysis Correlation Matrix')
plt.show()
