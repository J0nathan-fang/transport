import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from data_manager import Loader
from data_manager import Cleaner

class RFAnalyzer:
    """执行RF模型构造"""

    def run_analysis(self, file_path, target_column=None):
        """执行完整的训练和预测流程，并返回分析结果文本"""

        # 用于收集输出日志，最后统一返回给界面显示
        logs = []

        def log(text):
            logs.append(str(text))
            # 同时也打印到控制台，方便调试
            print(text)

        # 检查文件
        if not os.path.exists(file_path):
            return f"❌ 错误：找不到文件 '{file_path}'"

        try:
            log(f"正在从 {file_path} 加载数据...")

            # 数据加载与清洗
            cleaner = Cleaner()
            loader = Loader()
            raw_df = loader.load_csv(file_path)
            if raw_df is None:
                return "错误：数据加载失败，请检查文件格式或内容。"
            df = cleaner.clean_data(raw_df)

            y = df[target_column]
            x = df.drop(columns=[target_column])

            # 数据集划分
            log("正在划分训练集和测试集...")
            try:
                # 尝试进行分层抽样 (Stratified)
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError as e:
                # 如果某个类别样本太少导致分层失败，则回退到随机抽样
                log(f"⚠️ 警告：检测到稀有类别样本（数量少于2），无法进行分层抽样。")
                log(">>> 已自动切换为随机抽样模式。")
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.3, random_state=42, stratify=None
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
            try:
                report = classification_report(y_test, y_pred, zero_division=0)
                log("\n分类报告:")
                log(report)
            except Exception as e:
                log(f"无法生成详细分类报告: {e}")

            # 特征重要性
            log("-" * 30)
            log("特征重要性分析:")
            importances = self.model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': x.columns, 'Importance': importances})
            log(feature_imp_df.sort_values(by='Importance', ascending=False).to_string(index=False))

            # KFC验证
            log("-" * 30)
            log("K折交叉验证分析:")

            # 检查最小类别样本数，决定使用 StratifiedKFold 还是 KFold
            min_class_count = y.value_counts().min()
            n_splits = 5

            if min_class_count < n_splits:
                log(f"⚠️ 注意：部分类别样本数({min_class_count})少于折数({n_splits})，切换为普通KFold。")
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            try:
                scores = cross_val_score(self.model, x, y, cv=splitter)
                log(f"\n每次得分: {scores.round(2)}")
                log(f"平均准确率: {scores.mean():.3f}")
            except Exception as cv_e:
                log(f"交叉验证执行失败: {str(cv_e)}")

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


if __name__ == "__main__":
    # 单元测试

    analyzer = RFAnalyzer()
    # 测试一下当前目录是否有 iris.csv
    if os.path.exists('iris.csv'):
        print(analyzer.run_analysis('iris.csv'))
    else:
        print("当前目录无 iris.csv，请先准备数据。")