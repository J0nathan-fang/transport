import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Cleaner:
    def __init__(self):
        # 你可以在这里定义一些默认配置，比如默认要删除的列名
        # self.drop_columns_list = ['Unnecessary_ID', 'Temp_Column']
        pass

    def clean_data(self, df):
        """
        主清洗流程：按顺序执行所有清洗步骤
        """
        # 1. 创建副本，避免修改原始数据
        df_clean = df.copy()

        # 2. 调用各个具体的清洗步骤
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._remove_useless_columns(df_clean)
        df_clean = self._convert_types(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        # df_clean = self._filter_invalid_rows(df_clean)

        # 3. 重置索引（因为删除了行，索引可能不连续了）
        df_clean.reset_index(drop=True, inplace=True)

        print(f"Cleaner: 清洗完成，剩余行数: {len(df_clean)}")
        return df_clean

    def _remove_duplicates(self, df):
        """去除完全重复的行"""
        initial_len = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < initial_len:
            print(f"Cleaner: 删除了 {initial_len - len(df)} 行重复数据")
        return df

    def _handle_missing_values(self, df):
        """
        处理缺失值：
        1. 自动识别数值型列，使用该列的【平均值】填充 NaN。
        2. 对于非数值型列（字符串/类别），使用【众数】（出现最多的值）填充。
        """
        df_clean = df.copy()
        # --- 1. 处理数值型列 (int, float) ---
        # 筛选出所有数值列
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        # 批量计算均值并填充
        # fillna 接受一个 Series（即每列对应的均值），会自动对齐列名
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        # --- 2. 处理非数值型列 (可选优化) ---
        # 如果列不是数字，算不了平均值，通常用“众数”填充
        object_cols = df_clean.select_dtypes(exclude=[np.number]).columns

        for col in object_cols:
            if df_clean[col].isnull().any():
                # 获取众数 (mode() 返回的是 Series，取第一个值)
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    # 如果整列都是空的，填充一个占位符
                    df_clean[col] = df_clean[col].fillna("Unknown")
        return df_clean

    def _convert_types(self, df):
        """转换字符串"""

        df_out = df.copy()
        # 遍历所有列
        for col in df_out.columns:
            # 只处理非数值类型的列 (object, category, bool)
            if not pd.api.types.is_numeric_dtype(df_out[col]):
                try:
                    # 尝试 1: 直接转换为数字
                    # errors='raise' 意味着如果有非数字字符，直接报错跳到 except
                    # 我们先尝试转，看是不是纯数字字符串
                    df_out[col] = pd.to_numeric(df_out[col], errors='raise')
                except:
                    # 尝试 2: 如果转换失败，说明包含真正的文本
                    # 使用 LabelEncoder 将文本类别转为数字索引
                    # 注意：LabelEncoder 不支持 NaN，所以先填充 NaN 为 "Missing" 或其他占位符

                    # 为了防止混合类型报错 (比如 float NaN 和 string)，先统一转为 str
                    df_out[col] = df_out[col].astype(str)

                    le = LabelEncoder()
                    df_out[col] = le.fit_transform(df_out[col])

                    # 打印日志（可选，如果你想知道哪些列被编码了）
                    print(f"列 '{col}' 已进行 LabelEncoding 编码。")
        return df_out

    def _remove_useless_columns(self, df):
        """
        功能：
        1. 删除单一值列（所有行数据都一样）。
        2. (谨慎使用) 删除高势列（High Cardinality），如 ID 列。
           这里保守起见，只删除 100% 唯一的 Object/String 列（通常是 ID）。
        """
        # 1. 删除单一值列
        # nunique() 计算唯一值的数量，如果为 1，说明该列无信息量
        cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]

        # 2. 删除 ID 类列 (假设：如果是字符串且每一行都不重复，大概率是 ID)
        # 注意：如果是数值型 ID (1,2,3...)，通常会被当做特征，这里暂不自动删除以免误删
        for col in df.select_dtypes(include=['object', 'string']):
            if df[col].nunique() == len(df):
                cols_to_drop.append(col)
        if cols_to_drop:
            print(f"⚠️ 自动移除了无用列: {list(set(cols_to_drop))}")
            df = df.drop(columns=list(set(cols_to_drop)))

        return df

    """
    def _filter_invalid_rows(self, df):
        基于业务逻辑过滤无效数据
        # 速度大于0
        if 'speed' in df.columns:
            df = df[df['speed'] >= 0]

        # 去除异常大的离群值
        # if 'speed' in df.columns:
        #     df = df[df['speed'] < 200]

        return df
    """



