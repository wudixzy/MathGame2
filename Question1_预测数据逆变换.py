import numpy as np
import pandas as pd

def softmax_transform(data):
    """
    对数据进行SoftMax变换。
    参数：
    data (numpy.ndarray): 输入数据，形状为 (n_samples, n_features)
    返回：
    numpy.ndarray: 经过SoftMax变换后的数据，形状与输入数据相同。
    """
    exp_data = np.exp(data - np.max(data, axis=1, keepdims=True))
    return exp_data / np.sum(exp_data, axis=1, keepdims=True)

def inverse_softmax_transform(softmax_data, sum_exp_original):
    """
    对SoftMax变换后的数据进行逆变换。
    参数：
    softmax_data (numpy.ndarray): 经过SoftMax变换的数据，形状为 (n_samples, n_features)
    sum_exp_original (numpy.ndarray): 原始数据的指数和，形状为 (n_samples, 1)
    返回：
    numpy.ndarray: 原始数据，形状与输入数据相同。
    """
    original_data = softmax_data * sum_exp_original
    return np.log(original_data)

def inverse_clr_transform(clr_data, geometric_mean):
    """
    对CLR变换后的数据进行逆变换。
    参数：
    clr_data (numpy.ndarray): 经过CLR变换的数据，形状为 (n_samples, n_features)
    geometric_mean (numpy.ndarray): 几何平均数，形状为 (n_samples, 1)
    返回：
    numpy.ndarray: 原始成分数据，形状与输入数据相同。
    """
    compositional_data = np.exp(clr_data) * geometric_mean
    return compositional_data

# 读取原始数据
original_file_path = '附件.xlsx'
original_sheet_name = '表单2_version3'
original_data = pd.read_excel(original_file_path, sheet_name=original_sheet_name)
original_data_values = original_data.drop(columns=original_data.columns[0]).values

# 对原始数据进行SoftMax变换
softmax_data_original = softmax_transform(original_data_values)

# 计算SoftMax变换后的几何平均数
geometric_mean = np.exp(np.mean(np.log(softmax_data_original), axis=1, keepdims=True))

# 计算SoftMax变换后的数据的指数和
sum_exp_original = np.sum(np.exp(original_data_values), axis=1, keepdims=True)

# 读取变换后的数据
transformed_file_path = 'SoftMax_CLR_data.xlsx'
transformed_data = pd.read_excel(transformed_file_path, sheet_name='Sheet2')
transformed_data_values = transformed_data.drop(columns=transformed_data.columns[0]).values

# 逆CLR变换
inverse_clr_data = inverse_clr_transform(transformed_data_values, geometric_mean)

# 逆SoftMax变换
inverse_softmax_data = inverse_softmax_transform(inverse_clr_data, sum_exp_original)

# 将处理后的数据转换回DataFrame
inverse_softmax_data_df = pd.DataFrame(inverse_softmax_data, columns=original_data.columns[1:])

# 将第一列重新加入到处理后的数据中
result_data = pd.concat([transformed_data.iloc[:, 0], inverse_softmax_data_df], axis=1)

# 保存到新的Excel文件
result_file_path = 'Inverse_Transformed_Pre_Data1.xlsx'
result_data.to_excel(result_file_path, index=False)

print("数据已处理并保存到", result_file_path)
