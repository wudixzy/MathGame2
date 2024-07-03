import pandas as pd
import numpy as np

# 加载数据
merged_data_path = 'Merged_SoftMax_CLR_data.xlsx'
merged_data = pd.read_excel(merged_data_path, sheet_name='严重风化')

# 去掉不需要的列
merged_data = merged_data.drop(columns=['文物采样点', '文物编号'])

# 找出所有数值型列
numeric_columns = merged_data.select_dtypes(include=[np.number]).columns

weathering_map = {'无风化': 'No Weathering', '风化': 'Weathering', '严重风化': 'Weathering', '轻度风化': 'Weathering'}

# 将风化程度列重命名和合并以避免显示中文
merged_data['Weathering Type'] = merged_data['风化程度'].map(weathering_map)
merged_data['Weathering Type'] = merged_data['Weathering Type'].replace({'严重风化': 'Weathering', '轻度风化': 'Weathering'})

# 将类型列重命名以避免显示中文
type_map = {'高钾': 'High Potassium', '铅钡': 'Lead Barium'}
merged_data['Type'] = merged_data['类型'].map(type_map)

# 确保所有数值型列都是数值型
for column in numeric_columns:
    merged_data[column] = pd.to_numeric(merged_data[column], errors='coerce')

# 定义函数计算均值和标准差
def calculate_statistics(data):
    mean = data.mean().values
    std = data.std(ddof=0).values  # 使用无偏标准差（ddof=0）
    return mean, std

# 分别计算四种情况下的均值和标准差
statistics = {}
for typ in ['High Potassium', 'Lead Barium']:
    for weathering in ['Weathering', 'No Weathering']:
        subset = merged_data[(merged_data['Type'] == typ) & (merged_data['Weathering Type'] == weathering)][numeric_columns].dropna()
        mean, std = calculate_statistics(subset)
        statistics[f'{typ}_{weathering}'] = {'mean': mean, 'std': std}

# 按照公式计算风化点在风化前的数值
results = {}
for typ in ['High Potassium', 'Lead Barium']:
    for index, row in merged_data[merged_data['Type'] == typ].iterrows():
        if row['Weathering Type'] == 'Weathering':
            CLR_A = row[numeric_columns].values
            mu_A = statistics[f'{typ}_Weathering']['mean']
            sigma_A = statistics[f'{typ}_Weathering']['std']
            mu_B = statistics[f'{typ}_No Weathering']['mean']
            sigma_B = statistics[f'{typ}_No Weathering']['std']
            CLR_B = mu_B + sigma_B / sigma_A * (CLR_A - mu_A)
            results[index] = pd.Series(CLR_B, index=numeric_columns)

# 将结果添加到原始数据中
for index, series in results.items():
    for column in numeric_columns:
        merged_data.loc[index, f'{column} (Estimated Pre-Weathering)'] = series[column]

# 保存结果到新的Excel文件
output_path = 'Estimated_Pre_Weathering_Data.xlsx'
merged_data.to_excel(output_path, index=False)

print(f'Results saved to {output_path}')
