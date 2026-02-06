import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QPlainTextEdit, QMessageBox, QWidget,
                             QVBoxLayout, QHBoxLayout)


class SalaryStatsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 1. 初始化窗口
        self.setWindowTitle('薪资统计 Pro (PyQt6版)')
        self.resize(500, 400)

        # 2. 设置中心部件 (PyQt6 强制要求主窗口必须有中心部件才能设置 Layout)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主布局：水平布局
        self.main_layout = QHBoxLayout(self.central_widget)

        # 3. 创建 UI 组件
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("请输入格式：姓名 薪资 年龄\n例如：\n张三 25000 30\n李四 15000 25")

        self.btn_calc = QPushButton('开始统计')
        self.btn_calc.clicked.connect(self.handle_calc)

        # 4. 布局装载
        # 左侧：文本框 (stretch=2 占用 2/3 宽度)
        self.main_layout.addWidget(self.text_edit, stretch=2)

        # 右侧：按钮容器 (垂直布局)
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.btn_calc)
        self.right_layout.addStretch()  # 弹簧，将按钮顶到上方

        # 将右侧布局加入主布局 (stretch=1 占用 1/3 宽度)
        self.main_layout.addLayout(self.right_layout, stretch=1)

    def handle_calc(self):
        info = self.text_edit.toPlainText()

        list_above_20k = []
        list_below_20k = []
        errors = []

        for line_num, line in enumerate(info.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            # 自动处理任意数量空格
            parts = line.split()

            # 健壮性：防止因缺少数据导致的崩溃
            if len(parts) < 2:
                errors.append(f"第 {line_num} 行数据不足")
                continue

            name = parts[0]
            salary_str = parts[1]

            # 健壮性：防止因非数字字符导致的崩溃
            try:
                salary = int(salary_str)
            except ValueError:
                errors.append(f"第 {line_num} 行薪资格式错误: {salary_str}")
                continue

            if salary >= 20000:
                list_above_20k.append(name)
            else:
                list_below_20k.append(name)

        # 结果拼接
        result_text = "【薪资 20000 以上】\n" + ("\n".join(list_above_20k) if list_above_20k else "无")
        result_text += "\n\n"
        result_text += "【薪资 20000 以下】\n" + ("\n".join(list_below_20k) if list_below_20k else "无")

        if errors:
            result_text += "\n\n⚠️ 解析错误：\n" + "\n".join(errors)

        QMessageBox.about(self, '统计结果', result_text)


if __name__ == '__main__':
    # 传入 sys.argv 以支持命令行参数
    app = QApplication(sys.argv)

    window = SalaryStatsWindow()
    window.show()

    # PyQt6 标准退出方式
    sys.exit(app.exec())

