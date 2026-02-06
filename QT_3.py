import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from RandomF_3 import IrisModelAnalyzer
from ui.analysis_1 import UiForm
# import warnings
# 忽略警告
# warnings.filterwarnings('ignore')


class MyDataApp(QMainWindow, UiForm):
    def __init__(self):
        super().__init__()
        # 初始化 UI 界面
        self.setupUi(self)

        # 实例化业务逻辑类
        self.analyzer = IrisModelAnalyzer()

        # 信号与槽连接
        # 1. 选择文件
        # 当 btn_select 被点击时，执行 self.on_select_file 函数
        self.btn_import.clicked.connect(self.on_select_file)
        # 2. 开始分析
        if hasattr(self, 'btn_analyze'):
            self.btn_analyze.clicked.connect(self.on_run_analysis)
        else:
            print("⚠️ 警告: UI中未找到 btn_analyze 按钮，请检查 .ui 文件")

    def on_select_file(self):
        """
        选择文件
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
        if file_path:
            self.line_file_path.setText(file_path)
            # 清空之前的日志，准备新的开始
            if hasattr(self, 'textBrowser_log'):
                self.textBrowser_log.setText("文件已加载，请点击“开始模型训练与分析”...")

    def on_run_analysis(self):
        """
        负责读取路径并执行分析
        """
        # 1. 获取文件路径
        file_path = self.line_file_path.text().strip()

        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择一个数据文件！")
            return
        # 2. 界面提示正在运行
        if hasattr(self, 'textBrowser_log'):
            self.textBrowser_log.setText(f"正在分析 {file_path} ...\n请稍候...")
            QApplication.processEvents()  # 强制刷新界面，防止卡顿感
        # 3. 调用后台逻辑 (RandomF_2)
        # run_analysis 现在返回的是一个包含所有结果的长字符串
        result_text = self.analyzer.run_analysis(file_path)
        # 4. 将结果显示在文本框中
        if hasattr(self, 'textBrowser_log'):
            self.textBrowser_log.setText(result_text)
            # 滚动到底部 (可选)
            # self.textBrowser_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        else:
            # 如果没有文本框，还是弹窗显示
            QMessageBox.information(self, "分析结果", result_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyDataApp()
    window.show()
    sys.exit(app.exec())