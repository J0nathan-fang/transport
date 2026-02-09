import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

class RFAnalyzer:
    """
    默认最后一列为目标y，前面为特征x
    """
    def run_analysis(self, file_path):
        """
        执行完整的训练和预测流程，并返回分析结果文本
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
            df = pd.read_csv(file_path)
            log(f"数据加载成功。共 {len(df)} 条记录。")
            if df.shape[1] < 2:
                return "❌ 错误：数据文件列数不足，至少需要2列（特征列 + 目标列）。"
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            # 数据预处理
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                mapping_info = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"标签映射关系: {mapping_info}")

            x = x.select_dtypes(include=['number'])

            if x.shape[1] == 0:
                return "❌ 错误：在去除最后一列后，未发现有效的数值型特征列。"

            # 数据集划分
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, random_state=None
            )

            # 模型训练
            log("正在训练随机森林模型...")
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=None)
            self.model.fit(x_train, y_train)

            # 预测与评估
            y_pred = self.model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)

            log("-" * 30)
            log(f"✅ 模型准确率 (Accuracy): {acc:.4f}")
            log("\n分类报告:")
            log(classification_report(y_test, y_pred))

            # 特征重要性
            log("-" * 30)
            log("特征重要性分析:")
            importances = self.model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': x.columns, 'Importance': importances})
            log(feature_imp_df.sort_values(by='Importance', ascending=False).to_string(index=False))

            # KFC验证
            log("-" * 30)
            log("K折交叉验证分析:")
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
            scores = cross_val_score(self.model, x, y, cv=splitter)
            log(f"\n每次得分: {scores.round(2)}")
            log(f"平均准确率: {scores.mean():.3f}")

            # 随机样本预测演示
            log("-" * 30)
            log("正在生成随机样本进行测试...")

            min_values = x.min()
            max_values = x.max()

            # 生成随机数据
            random_features = np.random.uniform(low=min_values, high=max_values)
            new_sample_reshaped = random_features.reshape(1, -1)

            # 创建 DataFrame (修复之前的警告问题)
            new_sample_df = pd.DataFrame(new_sample_reshaped, columns=x.columns)

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
    analyzer = RFAnalyzer()
    # 测试一下当前目录是否有 iris.csv
    if os.path.exists('iris.csv'):
        print(analyzer.run_analysis('iris.csv'))
    else:
        print("当前目录无 iris.csv，请先准备数据。")