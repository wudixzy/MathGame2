import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 读取数据
file_path = '附件.xlsx'
sheet_name_train = '表单2_version2'  # 训练数据表单
sheet_name_predict = '表单3'  # 预测数据表单

# 使用pandas读取指定表单
data_train = pd.read_excel(file_path, sheet_name=sheet_name_train)
data_predict = pd.read_excel(file_path, sheet_name=sheet_name_predict)

# 删除第一列
data_train = data_train.drop(columns=data_train.columns[0])

# 统一风化程度
data_train['表面风化'] = data_train['表面风化'].replace({'轻度风化': '风化', '严重风化': '风化', '风化': '风化'})

# 对风化程度进行编码
data_train['表面风化'] = data_train['表面风化'].map({'无风化': -1, '风化': 1})

# 对目标特征类型进行标签编码
label_encoder = LabelEncoder()
data_train['类型'] = label_encoder.fit_transform(data_train['类型'])

# 定义软间隔参数
C = 1.0  # 你可以调整这个值来进行敏感性分析

# 定义一个SimpleImputer实例来填充NaN值
imputer = SimpleImputer(strategy='constant', fill_value=0)

# 先划分一个训练集和测试集
X = data_train.drop(columns=['类型', '表面风化'])
y = data_train['类型']

# 填充缺失值
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在训练集上训练模型，并在测试集上评估
print("\n在训练集和测试集上进行训练和评估:")

# 创建软间隔SVM分类器
svm_classifier = SVC(kernel='linear', C=C)

# 在训练集上训练模型
svm_classifier.fit(X_train, y_train)

# 获取支持向量的数量和间隔大小
support_vectors = svm_classifier.support_vectors_
dual_coef = svm_classifier.dual_coef_
margin = 2 / np.sqrt(np.sum(svm_classifier.coef_ ** 2))

print(f"支持向量数量: {len(support_vectors)}")
print(f"间隔大小: {margin}")

# 在测试集上进行预测
y_pred = svm_classifier.predict(X_test)

# 评估模型性能
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

# 使用全部数据重新训练模型并测试
print("\n在全部数据上进行训练和测试:")

X_all = data_train.drop(columns=['类型', '表面风化'])
y_all = data_train['类型']

# 填充缺失值
X_all = imputer.fit_transform(X_all)

# 创建软间隔SVM分类器
svm_classifier = SVC(kernel='linear', C=C)

# 在全部数据上训练模型
svm_classifier.fit(X_all, y_all)

# 获取支持向量的数量和间隔大小
support_vectors = svm_classifier.support_vectors_
dual_coef = svm_classifier.dual_coef_
margin = 2 / np.sqrt(np.sum(svm_classifier.coef_ ** 2))

print(f"支持向量数量: {len(support_vectors)}")
print(f"间隔大小: {margin}")

# 在全部数据上进行预测
y_pred_all = svm_classifier.predict(X_all)

# 评估模型性能
print("Classification Report:")
print(classification_report(y_all, y_pred_all))

print("Accuracy Score:")
print(accuracy_score(y_all, y_pred_all))

# 填充预测数据集中的缺失值
# 检查预测数据中是否包含文物编号列，如果有则删除
if '文物编号' in data_predict.columns:
    X_predict = data_predict.drop(columns=['文物编号'])
else:
    X_predict = data_predict.copy()

# 检查预测数据中是否包含表面风化列，如果有则删除
if '表面风化' in X_predict.columns:
    X_predict = X_predict.drop(columns=['表面风化'])

X_predict = imputer.fit_transform(X_predict)

# 进行预测
y_pred_predict = svm_classifier.predict(X_predict)

# 将预测结果添加到原数据中
data_predict['预测类型'] = label_encoder.inverse_transform(y_pred_predict)

# 保存预测结果
result_file_path = 'Prediction_Result.xlsx'
data_predict.to_excel(result_file_path, index=False)

print(f"预测结果已保存到 {result_file_path}")