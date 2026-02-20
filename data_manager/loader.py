import pandas as pd
import os

class Loader:
    def __init__(self):
        pass

    def load_csv(self, file_path):
        """
        加载 CSV 文件并返回 DataFrame
        """
        if not os.path.exists(file_path):
            print(f"Loader: 错误 - 找不到文件: {file_path}")
            return None
        # 忽略大小写判断后缀
        lower_path = file_path.lower()
        # 处理Excel
        if lower_path.endswith(('.xls', '.xlsx')):
            try:
                return pd.read_excel(file_path)
            except Exception as e:
                print(f"Loader: Excel 读取错误 - {e}")
                return None
        # 处理CSV
        elif lower_path.endswith('.csv'):
            encodings_to_try = [
                'utf-8',
                'gbk',
                'gb18030',  # GBK的超集，支持更多字符
                'iso-8859-1',  # 西欧编码，通常能读取绝大多数无法识别的文件
                'cp1252'  # Windows 默认西文编码
            ]

            for encoding in encodings_to_try:
                try:
                    # encoding_errors='replace': 遇到无法识别的乱码字符，替换为 ?，而不是报错崩溃
                    # on_bad_lines='skip': 遇到格式严重错误的行，跳过，防止报错
                    print(f"Loader: 尝试使用 {encoding} 编码读取...")

                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        encoding_errors='replace',  # 关键参数：替换坏字符
                        quotechar='"',  # 确保摘要中的逗号不换行
                        on_bad_lines='skip'  # 跳过坏行
                    )

                    print(f"Loader: 成功使用 {encoding} 读取! 行数: {len(df)}")
                    return df

                except UnicodeDecodeError:
                    continue  # 尝试下一种编码
                except Exception as e:
                    # 如果不是编码错误，可能是其他问题，打印出来但继续尝试
                    print(f"Loader: {encoding} 读取遇到非编码错误: {e}")
                    continue
            print("Loader: 所有编码尝试均失败。")
            return None

        else:
            print("Loader: 不支持的文件格式")
            return None


if __name__ == "__main__":
    # 单元测试

    loader = Loader()
    # 检验
    file_path = r"E:\codeP\work-1\data\CRASH.CSV"

    df = loader.load_csv(file_path)

    if df is not None:
        print("✅ 数据加载成功！")
        print(df)
    else:
        print("❌ 数据加载失败")