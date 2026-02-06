import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------- 全局配置：解决中文显示和负号问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常
plt.rcParams['figure.figsize'] = (12, 7)      # 全局画布大小
plt.rcParams['font.size'] = 12                # 全局字体大小

# -------------------------- 1、读取红酒数据集并设置列名 --------------------------
# 读取本地wine.txt文件（无表头，header=None）
df = pd.read_csv(r'wine.txt', header=None)
# 设置列名：第0列为类别标签，1-13列为特征
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# -------------------------- 2、划分特征矩阵X和目标向量Y --------------------------
# 特征X：第1列到最后一列（所有特征列），目标Y：第0列（类别标签）
X = df.iloc[:, 1:].values  # 特征矩阵，(178,13)
Y = df.iloc[:, 0].values   # 目标向量，(178,)
# 提取特征标签（用于后续可视化和输出）
feat_labels = df.columns[1:]  # 13个特征的名称列表

# -------------------------- 3、分割训练集和测试集 --------------------------
# 测试集占比30%，随机种子0保证结果可复现，分层抽样保证类别分布一致
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0, stratify=Y
)

# -------------------------- 4、训练随机森林模型并计算特征重要性 --------------------------
# 初始化随机森林分类器：10000棵树，多线程训练，固定随机种子
# 修复原代码：random_st`在这里插入代码片`ate → random_state（语法错误）
forest = RandomForestClassifier(
    n_estimators=10000,  # 森林中树的数量，数量越多结果越稳定
    random_state=0,      # 固定随机种子，结果可复现
    n_jobs=-1            # 多线程训练（利用所有CPU核心），提升训练速度
)
# 模型训练（仅用训练集，避免数据泄露）
forest.fit(X_train, Y_train)

# 提取特征重要性：feature_importances_ 是随机森林的核心属性，值越大特征越重要
importances = forest.feature_importances_
print("="*80)
print("所有特征原始重要性值：\n", importances)
print("="*80)

# -------------------------- 5、特征重要性排序并打印详细结果 --------------------------
# np.argsort(importances)[::-1]：先升序排序取索引，再逆序→得到重要性从高到低的索引
indices = np.argsort(importances)[::-1]
# 存储排序后的特征名称（用于后续可视化）
sorted_feat_labels = []

# 按重要性从高到低打印特征排名、名称、重要性值
print("特征重要性排名（从高到低）：")
for f in range(X_train.shape[1]):  # X_train.shape[1] = 13（特征数量）
    rank = f + 1  # 排名（从1开始）
    feat_name = feat_labels[indices[f]]  # 排序后的特征名称
    feat_imp = importances[indices[f]]   # 排序后的特征重要性值
    sorted_feat_labels.append(feat_name) # 保存排序后的特征名称
    # 格式化打印：排名、特征名称（占30字符）、重要性值（保留6位小数）
    print("%2d) %-*s %.6f" % (rank, 30, feat_name, feat_imp))
print("="*80)

# -------------------------- 6、筛选高重要性特征（按阈值过滤） --------------------------
threshold = 0.15  # 重要性阈值，可根据需求调整
# 筛选训练集中重要性 > 阈值的特征，返回筛选后的特征矩阵
X_train_selected = X_train[:, importances > threshold]
# 打印筛选结果
selected_feat_names = feat_labels[importances > threshold]  # 筛选后的特征名称
print(f"重要性阈值 = {threshold}，筛选后的特征数量：{len(selected_feat_names)}")
print(f"筛选后的特征名称：{list(selected_feat_names)}")
print(f"筛选后训练集特征矩阵形状：{X_train_selected.shape}")  # 原(124,13) → 筛选后(124, n)
print("="*80)

# -------------------------- 7、可视化特征重要性（优化版柱状图） --------------------------
plt.figure()
# 绘制柱状图：x为特征索引，y为排序后的重要性值，橙色填充，居中对齐
plt.bar(
    x=np.arange(len(feat_labels)),  # x轴：0-12（13个特征）
    height=importances[indices],    # y轴：按重要性从高到低排序的值
    color='#FF7F0E',                # 橙色（美观且醒目）
    align='center',                 # 柱子居中对齐
    alpha=0.8                       # 透明度，避免遮挡
)

# 设置图表标题和坐标轴标签
plt.title('红酒数据集-随机森林特征重要性排序', fontsize=18, pad=20)
plt.ylabel('特征重要性', fontsize=16)
plt.xlabel('特征名称', fontsize=16, labelpad=10)

# 设置x轴刻度：替换为排序后的特征名称，旋转90度避免重叠，调整字体大小
plt.xticks(
    ticks=np.arange(len(feat_labels)),
    labels=sorted_feat_labels,
    rotation=90,
    fontsize=11
)

# 设置y轴范围：从0开始，上限略高于最大重要性值，提升美观度
plt.ylim(0, max(importances) * 1.1)

# 为每个柱子标注重要性值（保留4位小数），显示在柱子顶部
for i, imp in enumerate(importances[indices]):
    plt.text(
        x=i,
        y=imp + 0.005,  # 数值在柱子顶部上方0.005处
        s=f'{imp:.4f}', # 保留4位小数
        ha='center',    # 水平居中
        va='bottom',    # 垂直靠下
        fontsize=10
    )

# 添加网格线（仅y轴），提升可读性
plt.grid(True, alpha=0.3, axis='y')
# 自动调整布局，防止标签、标题重叠
plt.tight_layout()
# 显示图表
plt.show()