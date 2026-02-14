import pandas as pd

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
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._convert_types(df_clean)
        df_clean = self._filter_invalid_rows(df_clean)

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
        """处理缺失值 (NaN/None)"""
        # 策略 1：关键列缺失直接删除该行 (例如：时间戳或ID缺失)
        # 假设 'timestamp' 是必须存在的
        if 'timestamp' in df.columns:
            df.dropna(subset=['timestamp'], inplace=True)

        # 策略 2：数值列缺失用 0 或均值填充
        # df.fillna({'speed': 0}, inplace=True)

        # 策略 3：统一向前填充（用前一行数据填补空缺，适合时间序列）
        # df.ffill(inplace=True)

        return df

    def _convert_types(self, df):
        """转换数据类型 (这是最关键的一步)"""

        # 1. 转换时间列 (String -> Datetime)
        # 假设你的 CSV 里有一列叫 'timestamp' 或 'time'
        time_col = 'timestamp'
        if time_col in df.columns:
            try:
                # errors='coerce' 表示如果解析失败，就设为 NaT (空时间)
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                # 转换完后再删掉解析失败的时间行
                df.dropna(subset=[time_col], inplace=True)
            except Exception as e:
                print(f"Cleaner: 时间转换失败 - {e}")

        # 2. 转换数值列 (String -> Float/Int)
        # 防止 CSV 里有像 "123.45km/h" 这种带单位的字符串导致变成 object 类型
        numeric_cols = ['speed', 'volume', 'occupancy']  # 替换成你实际的列名
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _filter_invalid_rows(self, df):
        """基于业务逻辑过滤无效数据"""
        # 速度大于0
        if 'speed' in df.columns:
            df = df[df['speed'] >= 0]

        # 去除异常大的离群值
        # if 'speed' in df.columns:
        #     df = df[df['speed'] < 200]

        return df