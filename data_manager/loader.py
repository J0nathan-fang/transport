import pandas as pd

class Loader:
    def __init__(self):
        pass

    def load_csv(self, file_path):
        """
        加载 CSV 文件并返回 DataFrame
        """
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path)
        except Exception as e:
            print(f"Loader: 错误 {e}")
            return None
