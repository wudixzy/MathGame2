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
    data = data.astype(float)
    exp_data = np.exp(data - np.max(data, axis=1, keepdims=True))
    softmax_data = exp_data / np.sum(exp_data, axis=1, keepdims=True)
    return softmax_data


def clr_transform(compositional_data):
    """
    对成分数据进行中心对数变换（CLR）。

    参数：
    compositional_data (numpy.ndarray): 成分数据，形状为 (n_samples, n_features)，其中每行是一个样本，每列是一个特征。

    返回：
    numpy.ndarray: 经过CLR变换后的数据，形状与输入数据相同。
    """
    # 将数据转换为浮点数类型
    compositional_data = compositional_data.astype(float)

    # 计算几何平均数
    geometric_mean = np.exp(np.mean(np.log(compositional_data), axis=1, keepdims=True))
    # 计算CLR变换
    clr_data = np.log(compositional_data / geometric_mean)

    return clr_data


# 读取Excel文件中的指定表单
file_path = '附件.xlsx'
sheet_name = '表单2_version3'  # 指定表单名称

# 使用pandas读取指定表单
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 提取第一列
first_column = data.iloc[:, 0]

# 删除第一列
data = data.drop(columns=data.columns[0])

# 将数据转换为NumPy数组
compositional_data = data.values

# 先应用SoftMax变换
softmax_data = softmax_transform(compositional_data)

# 然后应用CLR变换
clr_data = clr_transform(softmax_data)

# 检查每行之和是否为零
print("每行之和是否为零：")
print(np.sum(clr_data, axis=1))

# 将处理后的数据转换回DataFrame
clr_data_df = pd.DataFrame(clr_data, columns=data.columns)

# 将第一列重新加入到处理后的数据中
result_data = pd.concat([first_column, clr_data_df], axis=1)

# 保存到新的Excel文件
result_data.to_excel("SoftMax_CLR_data.xlsx", index=False)

print("数据已处理并保存到SoftMax_CLR_data.xlsx")
