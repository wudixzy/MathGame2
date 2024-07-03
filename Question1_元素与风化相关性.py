import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
merged_data_path = 'Merged_SoftMax_CLR_data1.xlsx'
merged_data = pd.read_excel(merged_data_path, sheet_name='表面风化')

# 查看数据结构
print(merged_data)

# 去除不需要分析的列（文物采样点和文物编号）
data_for_analysis = merged_data.drop(columns=['文物采样点'])

# 将分类变量进行编码
data_for_analysis['表面风化'] = data_for_analysis['表面风化'].map({'无风化': 1, '风化': -1})

# 重命名列名以避免显示中文
column_renaming = {
    '二氧化硅(SiO2)': 'SiO2',
    '氧化钠(Na2O)': 'Na2O',
    '氧化钾(K2O)': 'K2O',
    '氧化钙(CaO)': 'CaO',
    '氧化镁(MgO)': 'MgO',
    '氧化铝(Al2O3)': 'Al2O3',
    '氧化铁(Fe2O3)': 'Fe2O3',
    '氧化铜(CuO)': 'CuO',
    '氧化铅(PbO)': 'PbO',
    '氧化钡(BaO)': 'BaO',
    '五氧化二磷(P2O5)': 'P2O5',
    '氧化锶(SrO)': 'SrO',
    '氧化锡(SnO2)': 'SnO2',
    '二氧化硫(SO2)': 'SO2',
    '表面风化': 'Surface Weathering'
}
data_for_analysis.rename(columns=column_renaming, inplace=True)

# 归一化处理
data_normalized = (data_for_analysis - data_for_analysis.min()) / (data_for_analysis.max() - data_for_analysis.min())

# 计算相关性矩阵
correlation_matrix = data_normalized.corr()

# 绘制相关性热力图
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
