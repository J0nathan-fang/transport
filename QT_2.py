import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox


# 1. 导入你刚才生成的界面模块
from ui.analysis_1 import UiForm


class MyDataApp(QMainWindow, UiForm):
    def __init__(self):
        super().__init__()
        # 1. 初始化 UI 界面
        self.setupUi(self)

        # 2. 信号与槽连接 (绑定事件)
        # 当 btn_select 被点击时，执行 self.on_select_file 函数
        self.btn_import.clicked.connect(self.on_select_file)

    def on_select_file(self):
        """
        槽函数：处理文件选择逻辑
        """
        # 获取当前脚本所在目录，作为打开文件的默认起点（提升体验）
        current_dir = os.getcwd()
        # 调用 PyQt6 的文件选择对话框
        # 参数含义: (父窗口, 标题, 默认路径, 文件过滤器)
        file_path, file_type = QFileDialog.getOpenFileName(
            self,
            "请选择数据文件",  # 弹窗标题
            current_dir,  # 默认打开路径
            "CSV 文件 (*.csv);;Excel 文件 (*.xlsx *.xls);;所有文件 (*.*)"  # 过滤器
        )
        # 判断用户是否点击了“确定”（如果没有选文件直接关掉，file_path 会是空字符串）
        if file_path:
            # 1. 将路径显示在文本框中
            self.line_file_path.setText(file_path)

            print(f"用户选中了文件: {file_path}")

            # (可选) 你可以在这里加一段逻辑，自动读取文件预览
            # self.load_data(file_path)
        else:
            print("用户取消了选择")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyDataApp()
    window.show()
    sys.exit(app.exec())