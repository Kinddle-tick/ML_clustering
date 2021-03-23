import pyqtgraph.examples
pyqtgraph.examples.run()
#
# # -*- coding: utf-8 -*-
# """
# Simple example of GraphItem use.
# """
#
# # import initExample  ## Add path to library (just for examples; you do not need this)
#
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# import numpy as np
#
# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)
#
# w = pg.GraphicsLayoutWidget(show=True)
# w.setWindowTitle('pyqtgraph example: GraphItem')
# v = w.addViewBox()
# v.setAspectLocked()
#
# g = pg.GraphItem()
# v.addItem(g)
#
# ## Define positions of nodes
# pos = np.array([
#     [0, 0],
#     [10, 0],
#     [0, 10],
#     [10, 10],
#     [5, 5],
#     [15, 5]
# ])
#
# ## Define the set of connections in the graph
# adj = np.array([
#     [0, 1],
#     [1, 3],
#     [3, 2],
#     [2, 0],
#     [1, 5],
#     [3, 5],
# ])
#
# ## Define the symbol to use for each node (this is optional)
# symbols = ['o', 'o', 'o', 'o', 't', '+']
#
# ## Define the line style for each connection (this is optional)
# lines = np.array([
#     (255, 0, 0, 255, 1),
#     (255, 0, 255, 255, 2),
#     (255, 0, 255, 255, 3),
#     (255, 255, 0, 255, 2),
#     (255, 0, 0, 255, 1),
#     (255, 255, 255, 255, 4),
# ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])
#
# ## Update the graph
# g.setData(pos=pos, adj=adj, pen=lines, size=1, symbol=symbols, pxMode=False)
#
# ## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()


# from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
#
#
# class myLabel(QtWidgets.QLabel):  # 自定义的QLabel类
#
#     def __init__(self, parent=None):
#         super(myLabel, self).__init__(parent)
#
#     def mousePressEvent(self, e):  ##重载一下鼠标点击事件
#         # 左键按下
#         if e.buttons() == QtCore.Qt.LeftButton:
#             self.setText("左")
#         # 右键按下
#         elif e.buttons() == QtCore.Qt.RightButton:
#             self.setText("右")
#         # 中键按下
#         elif e.buttons() == QtCore.Qt.MidButton:
#             self.setText("中")
#         # 左右键同时按下
#         elif e.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.RightButton:
#             self.setText("左右")
#         # 左中键同时按下
#         elif e.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton:
#             self.setText("左中")
#         # 右中键同时按下
#         elif e.buttons() == QtCore.Qt.MidButton | QtCore.Qt.RightButton:
#             self.setText("右中")
#         # 左中右键同时按下
#         elif e.buttons() == QtCore.Qt.LeftButton | QtCore.Qt.MidButton | QtCore.Qt.RightButton:
#             self.setText("左中右")
#
#
# class MyWindow(QtWidgets.QWidget):
#     def __init__(self):
#         super(MyWindow, self).__init__()
#         self.label = myLabel("点我")
#         self.gridLayout = QtWidgets.QGridLayout(self)
#         self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
#
#
# if __name__ == "__main__":
#     import sys
#
#     app = QtWidgets.QApplication(sys.argv)
#     myshow = MyWindow()
#     myshow.show()
#     sys.exit(app.exec_())

# from PyQt4 import QtGui  # (the example applies equally well to PySide)
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# ## Always start by initializing Qt (only once per application)
#
# class MyWidget(QtGui.QWidget):
#     def __init__(self):
#         super(MyWidget, self).__init__()
#         ## Define a top-level widget to hold everything
#         # w = QtGui.QWidget()
#         ## Create some widgets to be placed inside
#         self.btn = QtGui.QPushButton('press me',self)
#         self.btn.clicked.connect(self.Action)
#
#         text = QtGui.QLineEdit('enter text')
#         listw = QtGui.QListWidget()
#         plot = pg.GraphicsLayoutWidget(show=True)
#         ## Create a grid layout to manage the widgets size and position
#         layout = QtGui.QGridLayout()
#         self.setLayout(layout)
#         ## Add widgets to the layout in their proper positions
#         layout.addWidget(self.btn, 0, 1)   # button goes in upper-left
#         layout.addWidget(text, 1, 1)   # text edit goes in middle-left
#         layout.addWidget(listw, 2, 1)  # list widget goes in bottom-left
#         layout.addWidget(plot, 0, 0, 3, 1)  # plot goes on right side, spanning 3 rows
#     def Action(self):
#         print("啊～")
#
#
# ## Display the widget as a new window
# app = QtGui.QApplication([])
# w= MyWidget()
# w.show()
# ## Start the Qt event loop
# app.exec_()
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# import sys
#
#
# class Example(QWidget):
#
#     def __init__(self):
#         super().__init__()
#
#         self.t = 0
#
#         window = QWidget()
#         vbox = QVBoxLayout(window)
#         # vbox = QVBoxLayout(window)
#
#         self.lcdNumber = QLCDNumber()
#         button = QPushButton("测试")
#         vbox.addWidget(self.lcdNumber)
#         vbox.addWidget(button)
#
#         self.timer = QTimer()
#
#         button.clicked.connect(self.Work)
#         self.timer.timeout.connect(self.CountTime)
#
#         self.setLayout(vbox)
#         self.show()
#
#     def CountTime(self):
#         self.t += 1
#         self.lcdNumber.display(self.t)
#
#     def Work(self):
#         self.timer.start(1000)
#         self.thread = RunThread()
#         self.thread.start()
#         # self.thread.trigger.connect(self.TimeStop)
#
#     def TimeStop(self):
#         self.timer.stop()
#         print("运行结束用时", self.lcdNumber.value())
#         self.t = 0
#
#
# class RunThread(QThread):
#     # python3,pyqt5与之前的版本有些不一样
#     #  通过类成员对象定义信号对象
#     # _signal = pyqtSignal(str)
#
#     # trigger = pyqtSignal()
#
#     def __init__(self, parent=None):
#         super(RunThread, self).__init__()
#
#     def __del__(self):
#         self.wait()
#
#     def run(self):
#         # 处理你要做的业务逻辑，这里是通过一个回调来处理数据，这里的逻辑处理写自己的方法
#         # wechat.start_auto(self.callback)
#         # self._signal.emit(msg);  可以在这里写信号焕发
#         for i in range(2033000):
#             print(i)
#             pass
#         # self.trigger.emit()
#         # self._signal.emit(msg)
#
#     def callback(self, msg):
#         # 信号焕发，我是通过我封装类的回调来发起的
#         # self._signal.emit(msg)
#         pass
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     th = Example()
#     sys.exit(app.exec_())


