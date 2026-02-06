import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
# 忽略警告
warnings.filterwarnings('ignore')

def train_iris_model(file_path):
    # 1. 检查文件是否存在 (工程最佳实践：防御性编程)
    if not os.path.exists(file_path):
        print(f"❌ 错误：在当前目录下找不到文件 '{file_path}'。")
        print("请确保将数据保存为 'iris.data' 并与脚本放在同一目录，或者提供绝对路径。")
        return
    try:
        print(f"正在从 {file_path} 加载数据...")

        # 定义列名
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

        # 读取本地文件
        # header=None 表示文件中没有列名行
        df = pd.read_csv(file_path, header=None, names=column_names)

        print(f"数据加载成功。共 {len(df)} 条记录。")

        # 2. 数据集划分
        X = df.drop('class', axis=1)
        y = df['class']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. 模型训练
        print("正在训练随机森林模型...")
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # 4. 预测与评估
        y_pred = rf_classifier.predict(X_test)

        print("-" * 30)
        print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
        print("\n分类报告 (Classification Report):")
        print(classification_report(y_test, y_pred))

        # 5. 特征重要性
        print("-" * 30)
        print("特征重要性分析:")
        importances = rf_classifier.feature_importances_
        feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        print(feature_imp_df.sort_values(by='Importance', ascending=False))
        print("-" * 30)

        # 6. 限定范围内随机生成一条新数据进行预测
        # ---------------------------------------------------------
        print("正在生成随机样本进行测试...")

        # 获取每个特征的最小值和最大值，作为随机生成的边界
        min_values = X.min()
        max_values = X.max()
        print(f"\n特征取值范围参考 (Min - Max):")
        for col in X.columns:
            print(f"  {col}: {min_values[col]:.1f} - {max_values[col]:.1f}")

        # 使用 numpy 生成随机数据
        # np.random.uniform 会在 [min, max] 之间生成连续均匀分布的随机数
        random_features = np.random.uniform(low=min_values, high=max_values)

        # 将 Series 转换为 numpy 数组并调整形状为 (1, 4) 以符合 sklearn 的输入要求
        new_sample_reshaped = random_features.reshape(1, -1)

        # 创建一个 DataFrame 用于展示，包含列名，方便查看
        new_sample_df = pd.DataFrame(new_sample_reshaped, columns=X.columns)
        print("\n生成的随机样本数据:")
        print(new_sample_df.round(2).to_string(index=False))  # 保留两位小数打印

        # 进行预测
        prediction = rf_classifier.predict(new_sample_reshaped)
        print("-" * 30)
        print(f"🌲 模型预测分类结果: {prediction[0]}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    # 指定本地文件路径
    local_filename = 'iris.csv'

    train_iris_model(local_filename)