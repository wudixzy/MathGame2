import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
file_path = '附件.xlsx'
sheet_name = '表单2_version2'  # 数据表单名称

# 使用pandas读取指定表单
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 删除第一列
data = data.drop(columns=data.columns[0])

# 统一风化程度
data['表面风化'] = data['表面风化'].replace({'轻度风化': '风化', '严重风化': '风化', '风化': '风化'})

# 将元素名称映射到英文
element_map = {
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
    '二氧化硫(SO2)': 'SO2'
}

data.rename(columns=element_map, inplace=True)

# 删除 'Unnamed: 15' 列
if 'Unnamed: 15' in data.columns:
    data = data.drop(columns=['Unnamed: 15'])

# 定义绘图函数
def plot_correlation_heatmap(data_subset, title):
    correlation_matrix = data_subset.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.show()
    return correlation_matrix

# 分别处理风化、未风化、高钾、铅钡组合的四种情况
conditions = {
    'Weathering_High_Potassium': (data['表面风化'] == '风化') & (data['类型'] == '高钾'),
    'No_Weathering_High_Potassium': (data['表面风化'] == '无风化') & (data['类型'] == '高钾'),
    'Weathering_Lead_Barium': (data['表面风化'] == '风化') & (data['类型'] == '铅钡'),
    'No_Weathering_Lead_Barium': (data['表面风化'] == '无风化') & (data['类型'] == '铅钡')
}

correlation_matrices = {}

for condition_name, condition in conditions.items():
    data_subset = data[condition].drop(columns=['类型', '表面风化'])
    correlation_matrices[condition_name] = plot_correlation_heatmap(data_subset, f'Correlation Heatmap: {condition_name}')

# 比较不同类别之间的化学成分关联关系的差异性
def plot_difference_heatmap(corr1, corr2, title):
    difference_matrix = corr1 - corr2
    plt.figure(figsize=(12, 10))
    sns.heatmap(difference_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.show()
    return difference_matrix

def extract_significant_differences(difference_matrix, threshold=0.4):
    significant_differences = []
    for i in range(difference_matrix.shape[0]):
        for j in range(i+1, difference_matrix.shape[1]):
            if abs(difference_matrix.iloc[i, j]) > threshold:
                significant_differences.append({
                    'Component Pair': f"{difference_matrix.index[i]} - {difference_matrix.columns[j]}",
                    'Difference': difference_matrix.iloc[i, j]
                })
    return significant_differences

# 高钾玻璃文物样品风化前后化学成分关联关系差异
diff_high_potassium = plot_difference_heatmap(correlation_matrices['Weathering_High_Potassium'], correlation_matrices['No_Weathering_High_Potassium'], 'Difference in Correlation (High Potassium: Weathering vs No Weathering)')
# 铅钡玻璃文物样品风化前后化学成分关联关系差异
diff_lead_barium = plot_difference_heatmap(correlation_matrices['Weathering_Lead_Barium'], correlation_matrices['No_Weathering_Lead_Barium'], 'Difference in Correlation (Lead Barium: Weathering vs No Weathering)')
# 高钾与铅钡玻璃文物样品在风化状态下的化学成分关联关系差异
diff_weathering = plot_difference_heatmap(correlation_matrices['Weathering_High_Potassium'], correlation_matrices['Weathering_Lead_Barium'], 'Difference in Correlation (Weathering: High Potassium vs Lead Barium)')
# 高钾与铅钡玻璃文物样品在未风化状态下的化学成分关联关系差异
diff_no_weathering = plot_difference_heatmap(correlation_matrices['No_Weathering_High_Potassium'], correlation_matrices['No_Weathering_Lead_Barium'], 'Difference in Correlation (No Weathering: High Potassium vs Lead Barium)')

# 提取差异显著的成分对
significant_diff_high_potassium = extract_significant_differences(diff_high_potassium)
significant_diff_lead_barium = extract_significant_differences(diff_lead_barium)
significant_diff_weathering = extract_significant_differences(diff_weathering)
significant_diff_no_weathering = extract_significant_differences(diff_no_weathering)

# 打印显著差异
print("Significant differences (High Potassium: Weathering vs No Weathering):")
print(significant_diff_high_potassium)
print("\nSignificant differences (Lead Barium: Weathering vs No Weathering):")
print(significant_diff_lead_barium)
print("\nSignificant differences (Weathering: High Potassium vs Lead Barium):")
print(significant_diff_weathering)
print("\nSignificant differences (No Weathering: High Potassium vs Lead Barium):")
print(significant_diff_no_weathering)

# 保存输出为Excel文件
with pd.ExcelWriter('significant_differences.xlsx') as writer:
    pd.DataFrame(significant_diff_high_potassium).to_excel(writer, sheet_name='High_Pot_W_vs_No', index=False)
    pd.DataFrame(significant_diff_lead_barium).to_excel(writer, sheet_name='Lead_Ba_W_vs_No', index=False)
    pd.DataFrame(significant_diff_weathering).to_excel(writer, sheet_name='Weathering_HK_vs_LB', index=False)
    pd.DataFrame(significant_diff_no_weathering).to_excel(writer, sheet_name='NoWeather_HK_vs_LB', index=False)
