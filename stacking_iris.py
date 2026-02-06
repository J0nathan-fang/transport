import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# -------------- 全局配置：解决中文显示和负号问题 --------------
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 黑体显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
plt.rcParams['figure.figsize'] = (10, 6)  # 全局画布大小
plt.rcParams['font.size'] = 12  # 全局字体大小

# -------------- 导入机器学习相关库 --------------
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector

# -------------- 1、读取鸢尾花数据集（在线读取，无需本地文件） --------------
# 鸢尾花数据集列名：花萼长度、花萼宽度、花瓣长度、花瓣宽度、类别
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
datas = pd.read_csv(url, sep=',', header=None,
                    names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度', '类别'])

# -------------- 2、数据预处理 --------------
# 提取特征X（前4列）和目标Y（最后1列）
X = datas.iloc[:, :-1]
Y = datas.iloc[:, -1]

# 标签编码：将鸢尾花3个类别（字符串）转为0/1/2数字编码
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)  # 编码后：setosa=0, versicolor=1, virginica=2

# -------------- 3、分割训练集/测试集 --------------
# 测试集占比20%，随机种子28保证结果可复现
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=28, stratify=Y  # stratify=Y：分层抽样，保证类别分布一致
)

# -------------- 4、定义基学习器和元学习器 --------------
# 基学习器：4个经典分类器（调优后参数）
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')  # K近邻：7个邻居，距离加权
softmax = LogisticRegression(C=0.1, solver='lbfgs',
                             fit_intercept=False, max_iter=500)  # 软最大化：多分类
gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,
                                  max_depth=3, random_state=28)  # 梯度提升
rf = RandomForestClassifier(max_depth=5, n_estimators=150,
                            random_state=28, n_jobs=-1)  # 随机森林：多线程训练

# 元学习器：逻辑回归（融合基学习器结果）
meta_lr = LogisticRegression(C=0.1, solver='lbfgs',
                             random_state=28, max_iter=500)

# -------------- 5、定义3种Stacking集成方式 --------------
# 方式1：基础Stacking - 基学习器的类别预测结果作为元学习器输入
stacking01 = StackingClassifier(
    classifiers=[knn, softmax, gbdt, rf],
    meta_classifier=meta_lr,
)

# 方式2：概率型Stacking - 基学习器的类别概率作为元学习器输入（不平均，直接拼接）
# 例：2个基学习器对3分类的概率 → 拼接为[0.2,0.5,0.3, 0.3,0.3,0.4]
stacking02 = StackingClassifier(
    classifiers=[knn, softmax, gbdt, rf],
    meta_classifier=meta_lr,
    use_probas=True,  # 使用类别概率代替硬预测
    average_probas=False,  # 不平均概率，直接拼接所有基学习器的概率
)

# 方式3：特征分治Stacking - 不同基学习器使用不同特征子集训练（通过Pipeline+列选择）
# 为每个基学习器定制特征子集，充分利用不同特征的区分能力
pipe_knn = Pipeline([('col_sel', ColumnSelector([0, 1])), ('knn', knn)])  # KNN用：花萼长度、花萼宽度
pipe_softmax = Pipeline([('col_sel', ColumnSelector([2, 3])), ('softmax', softmax)])  # Softmax用：花瓣长度、花瓣宽度
pipe_rf = Pipeline([('col_sel', ColumnSelector([0, 3])), ('rf', rf)])  # RF用：花萼长度、花瓣宽度
pipe_gbdt = Pipeline([('col_sel', ColumnSelector([1, 2])), ('gbdt', gbdt)])  # GBDT用：花萼宽度、花瓣长度

stacking03 = StackingClassifier(
    classifiers=[pipe_knn, pipe_softmax, pipe_rf, pipe_gbdt],
    meta_classifier=meta_lr,
)

# -------------- 6、模型训练与性能评估 --------------
# 初始化存储结果的列表
train_scores = []  # 训练集准确率
test_scores = []  # 测试集准确率
model_names = []  # 模型名称
train_times = []  # 训练耗时（秒）

# 待训练模型列表：单模型 + 3种Stacking模型
models = [knn, softmax, gbdt, rf, stacking01, stacking02, stacking03]
names = ['KNN', 'Softmax', 'GBDT', 'RF', 'Stacking(基础)', 'Stacking(概率)', 'Stacking(特征分治)']

# 批量训练并评估
for clf, name in zip(models, names):
    print(f"\n========== 开始训练：{name} ==========")
    start_time = time.time()
    clf.fit(x_train, y_train)  # 模型训练
    train_time = round(time.time() - start_time, 4)  # 计算训练耗时（保留4位小数）

    # 计算准确率
    train_acc = round(clf.score(x_train, y_train), 4)  # 训练集准确率
    test_acc = round(clf.score(x_test, y_test), 4)  # 测试集准确率

    # 结果存储
    train_scores.append(train_acc)
    test_scores.append(test_acc)
    model_names.append(name)
    train_times.append(train_time)

    # 打印单模型结果
    print(f"训练耗时：{train_time} 秒")
    print(f"训练集准确率：{train_acc}")
    print(f"测试集准确率：{test_acc}")

# 打印所有模型汇总结果
print(f"\n========== 所有模型性能汇总 ==========")
print(f"模型名称：{model_names}")
print(f"训练集准确率：{train_scores}")
print(f"测试集准确率：{test_scores}")
print(f"训练耗时（秒）：{train_times}")

# -------------- 7、绘制结果对比图 --------------
# 创建画布：2个子图（上下排列），共享x轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
x_ticks = np.arange(len(model_names))  # x轴刻度位置

# 子图1：训练集/测试集准确率对比
ax1.plot(x_ticks, train_scores, 'r-o', linewidth=2, markersize=6, label='训练集')
ax1.plot(x_ticks, test_scores, 'b-s', linewidth=2, markersize=6, label='测试集')
ax1.set_title('鸢尾花数据集-不同分类器准确率对比', fontsize=18, pad=20)
ax1.set_ylabel('准确率', fontsize=14)
ax1.set_ylim(0.85, 1.02)  # 限定y轴范围，突出差异
ax1.legend(loc='lower right', fontsize=12)
ax1.grid(True, alpha=0.3)  # 添加网格线

# 为每个点标注准确率数值
for i, (t, te) in enumerate(zip(train_scores, test_scores)):
    ax1.text(i, t + 0.005, f'{t}', ha='center', fontsize=10, color='red')
    ax1.text(i, te - 0.01, f'{te}', ha='center', fontsize=10, color='blue')

# 子图2：模型训练耗时对比
ax2.bar(x_ticks, train_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                     '#9467bd', '#8c564b', '#e377c2'], alpha=0.8)
ax2.set_title('鸢尾花数据集-不同分类器训练耗时对比', fontsize=18, pad=20)
ax2.set_xlabel('分类器模型', fontsize=14)
ax2.set_ylabel('训练耗时（秒）', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')  # 仅y轴添加网格线

# 为每个柱子标注耗时数值
for i, t in enumerate(train_times):
    ax2.text(i, t + 0.01, f'{t}', ha='center', va='bottom', fontsize=10)

# 设置x轴刻度标签
plt.xticks(x_ticks, model_names, rotation=15, ha='right', fontsize=12)
plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.show()
