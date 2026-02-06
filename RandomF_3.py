import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class IrisModelAnalyzer:
    def __init__(self):
        # 初始化一些成员变量
        self.model = None
        self.column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    def run_analysis(self, file_path):
        """
        执行完整的训练和预测流程，并返回分析结果文本。
        """
        # 用于收集输出日志，最后统一返回给界面显示
        logs = []

        def log(text):
            logs.append(str(text))
            # 同时也打印到控制台，方便调试
            print(text)

        # 1. 检查文件
        if not os.path.exists(file_path):
            return f"❌ 错误：找不到文件 '{file_path}'"

        try:
            log(f"正在从 {file_path} 加载数据...")

            # 读取文件
            df = pd.read_csv(file_path, header=None, names=self.column_names)
            log(f"数据加载成功。共 {len(df)} 条记录。")

            # 2. 数据集划分
            X = df.drop('class', axis=1)
            y = df['class']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 3. 模型训练
            log("正在训练随机森林模型...")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # 4. 预测与评估
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            log("-" * 30)
            log(f"✅ 模型准确率 (Accuracy): {acc:.4f}")
            log("\n分类报告:")
            log(classification_report(y_test, y_pred))

            # 5. 特征重要性
            log("-" * 30)
            log("特征重要性分析:")
            importances = self.model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
            log(feature_imp_df.sort_values(by='Importance', ascending=False).to_string(index=False))

            # 6. 随机样本预测演示
            log("-" * 30)
            log("正在生成随机样本进行测试...")

            min_values = X.min()
            max_values = X.max()

            # 生成随机数据
            random_features = np.random.uniform(low=min_values, high=max_values)
            new_sample_reshaped = random_features.reshape(1, -1)

            # 创建 DataFrame (修复之前的警告问题)
            new_sample_df = pd.DataFrame(new_sample_reshaped, columns=X.columns)

            log("\n生成的随机样本:")
            log(new_sample_df.round(2).to_string(index=False))

            # 预测
            prediction = self.model.predict(new_sample_df)
            log("-" * 30)
            log(f"🌲 随机样本预测结果: {prediction[0]}")

            # 将列表合并成一个长字符串返回
            return "\n".join(logs)

        except Exception as e:
            error_msg = f"❌ 发生未知错误: {str(e)}"
            print(error_msg)
            return error_msg


# 单元测试代码
if __name__ == "__main__":
    analyzer = IrisModelAnalyzer()
    # 测试一下当前目录是否有 iris.csv
    if os.path.exists('iris.csv'):
        print(analyzer.run_analysis('iris.csv'))
    else:
        print("当前目录无 iris.csv，请先准备数据。")