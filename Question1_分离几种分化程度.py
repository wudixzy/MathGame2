import pandas as pd
import re

# Load the data
softmax_data_path = 'SoftMax_CLR_data.xlsx'
attachment_data_path = '附件.xlsx'

# Read the sheets
softmax_data = pd.read_excel(attachment_data_path, sheet_name='表单2_version2')
attachment_data = pd.read_excel(attachment_data_path, sheet_name='表单1')

# 确保文物编号可以匹配
# 先将所有值转换为字符串
softmax_data['。。文物采样点'] = softmax_data['文物采样点'].astype(str)

# 提取文物编号函数
def extract_number(x):
    match = re.search(r'\d+', x)
    return int(match.group()) if match else None

softmax_data['文物编号'] = softmax_data['文物采样点'].apply(extract_number)
attachment_data['文物编号'] = attachment_data['文物编号'].astype(int)

# 合并数据
merged_data = pd.merge(softmax_data, attachment_data[['文物编号', '表面风化']], on='文物编号', how='left')

# 检查合并后的数据
print(merged_data.head())

# 保存合并后的数据
merged_data.to_excel('Merged_SoftMax_CLR_data1.xlsx', sheet_name='表面风化', index=False)
