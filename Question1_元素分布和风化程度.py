import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
merged_data_path = 'Merged_SoftMax_CLR_data.xlsx'
merged_data = pd.read_excel(merged_data_path, sheet_name='严重风化')

# 选择需要绘制的元素和风化程度
elements = ['二氧化硅(SiO2)', '氧化钾(K2O)', '氧化钙(CaO)', '氧化铁(Fe2O3)', '氧化铅(PbO)', '五氧化二磷(P2O5)', '二氧化硫(SO2)']
weathering_map = {'无风化': 'No Weathering', '风化': 'Weathering', '严重风化': 'Weathering', '轻度风化': 'Weathering'}

# 将风化程度列重命名和合并以避免显示中文
merged_data['Weathering Type'] = merged_data['风化程度'].map(weathering_map)

# 将元素名称映射到英文，以避免显示中文
element_map = {
    '二氧化硅(SiO2)': 'SiO2',
    '氧化钾(K2O)': 'K2O',
    '氧化钙(CaO)': 'CaO',
    '氧化铁(Fe2O3)': 'Fe2O3',
    '氧化铅(PbO)': 'PbO',
    '五氧化二磷(P2O5)': 'P2O5',
    '二氧化硫(SO2)': 'SO2'
}

# 将类型列重命名以避免显示中文
type_map = {'高钾': 'High Potassium', '铅钡': 'Lead Barium'}
merged_data['Type'] = merged_data['类型'].map(type_map)

# 确保所有元素列都是数值型
for element in elements:
    merged_data[element] = pd.to_numeric(merged_data[element], errors='coerce')

# 获取所有类型
types = merged_data['Type'].unique()

# 绘制元素分布图（密度图），在同一幅图内绘制相同类型的风化和未风化的元素分布
for typ in types:
    for element in elements:
        plt.figure(figsize=(12, 8))
        for weathering in ['No Weathering', 'Weathering']:
            subset = merged_data[(merged_data['Type'] == typ) & (merged_data['Weathering Type'] == weathering)]
            if not subset.empty:
                sns.kdeplot(subset[element].dropna(), label=f'{weathering}', fill=True)
        plt.title(f'{element_map[element]} Distribution for {typ} by Weathering Type')
        plt.xlabel(f'{element_map[element]} Concentration')
        plt.ylabel('Density')
        plt.legend()
        # 保存图像
        plt.savefig(f'{element_map[element]}_Distribution_for_{typ}_by_Weathering_Type.png')
        plt.show()
