import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6 import uic
from analysis_engine import RFAnalyzer
import warnings
# 忽略警告
warnings.filterwarnings('ignore')


class MyDataApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # UI初始化

        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(current_dir, 'ui', 'main_window.ui')
        
        if not os.path.exists(ui_file_path):
            print(f"❌ 错误: 找不到 UI 文件: {ui_file_path}")
            sys.exit(1)
        self.ui = uic.loadUi(ui_file_path, self)

        # 实例化业务逻辑类
        self.analyzer = RFAnalyzer()

        # 信号与槽连接
        # 选择文件
        if hasattr(self, 'btn_import'):
            self.btn_import.clicked.connect(self.on_select_file)
        # 开始分析
        if hasattr(self, 'btn_analyze'):
            self.btn_analyze.clicked.connect(self.on_run_analysis)

    def on_select_file(self):
        # 选择文件

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
        # 读取路径并分析

        # 获取文件路径
        file_path = self.line_file_path.text().strip()

        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择一个数据文件！")
            return

        # 界面提示
        if hasattr(self, 'textBrowser_log'):
            self.textBrowser_log.setText(f"正在分析 {file_path} ...\n请稍候...")
            QApplication.processEvents()  # 刷新界面

        # 调用后台逻辑
        result_text = self.analyzer.run_analysis(file_path)

        # 将结果显示在文本框中
        if hasattr(self, 'textBrowser_log'):
            self.textBrowser_log.setText(result_text)
            # 滚动到底部 (可选)
            # self.textBrowser_log.moveCursor(QtGui.QTextCursor.MoveOperation.End)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyDataApp()
    window.show()
    sys.exit(app.exec())