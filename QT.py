import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6 import uic
from analysis_engine import RFAnalyzer
import pandas as pd
from data_manager import Loader

class MyDataApp(QMainWindow):
    def __init__(self):
        super().__init__()
        """UI初始化"""

        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(current_dir, 'ui', 'main_window.ui')
        
        if not os.path.exists(ui_file_path):
            print(f"❌ 错误: 找不到 UI 文件: {ui_file_path}")
            sys.exit(1)
        self.ui = uic.loadUi(ui_file_path, self)

        # 实例化业务逻辑类
        self.analyzer = RFAnalyzer()

        # 选择文件
        if hasattr(self, 'btn_import'):
            self.btn_import.clicked.connect(self.on_select_file)
        # 开始分析
        if hasattr(self, 'btn_analyze'):
            self.btn_analyze.clicked.connect(self.on_run_analysis)
        # 分析目标选择
        if hasattr(self, 'btn_choose'):
            self.btn_choose.clicked.connect(self.on_load_columns)

    def on_select_file(self):
        """选择文件"""

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

            self.log(f"已选择文件: {file_path}，请点击“开始模型训练与分析...")
            self.on_load_columns()

    def on_load_columns(self):
        """读取文件表头并填充ComboBox"""

        file_path = self.line_file_path.text().strip()
        loader = Loader()
        if not file_path or not os.path.exists(file_path):
            return  # 路径为空不做处理
        try:
            self.log("正在读取文件列信息...")
            QApplication.processEvents()  # 刷新界面防止假死
            df = loader.load_csv(file_path)

            columns = df.columns.tolist()

            # 更新 ComboBox
            if hasattr(self, 'comboBox'):
                self.comboBox.clear()
                self.comboBox.addItems(columns)
                self.log(f"列名加载成功，共 {len(columns)} 列。请在下方下拉框选择分析目标。")

        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"无法读取文件列名:\n{str(e)}")
            self.log(f"❌ 读取列名失败: {str(e)}")

    def on_run_analysis(self):
        """读取路径并分析"""

        # 获取文件路径
        file_path = self.line_file_path.text().strip()

        if not file_path:
            QMessageBox.warning(self, "提示", "请先选择一个数据文件！")
            return

        # 获取分析目标
        target_column = ""
        if hasattr(self, 'comboBox'):
            target_column = self.comboBox.currentText()

        if not target_column:
            QMessageBox.warning(self, "提示", "请选择需要分析的目标列 (Target)！")
            return

        # 界面提示
        self.log(f"正在分析 {file_path} ...\n目标列: {target_column}\n请稍候...")

        # 禁用按钮防止重复点击
        self.btn_analyze.setEnabled(False)
        QApplication.processEvents()

        # 调用后台逻辑
        try:
            result_text = self.analyzer.run_analysis(file_path, target_column)
            self.log(result_text)
            self.log("=== 分析完成 ===")
        except Exception as e:
            self.log(f"❌ 分析过程中发生错误:\n{str(e)}")
            QMessageBox.critical(self, "分析错误", str(e))

        finally:
            # 恢复按钮状态
            self.btn_analyze.setEnabled(True)

    def log(self, message):
        """打印日志"""

        if hasattr(self, 'textBrowser_log'):
            self.textBrowser_log.append(message)
            # 滚动到底部
            cursor = self.textBrowser_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.textBrowser_log.setTextCursor(cursor)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyDataApp()
    window.show()
    sys.exit(app.exec())