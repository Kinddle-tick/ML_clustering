import pyqtgraph as pg
import pandas as pd
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from Myfunc import DistanceMatrix,Prim,KMeans,AlgorithmList,Coopcheckdiv
from PyQt5 import QtWidgets
import time
import copy
from validation import validation

file = "five_cluster.txt"
divnum = 5
pg.setConfigOptions(antialias=True)
class Runthread(QtCore.QThread):
    #  通过类成员对象定义信号对象
    _signal = QtCore.pyqtSignal(str)
    stop_flag = False
    def __init__(self,func):
        super(Runthread, self).__init__()
        self.func = func
        # self.func=func
        # self.qtlock = QtCore.QMutex()

    def __del__(self):
        self.wait()

    def run(self):
        self.stop_flag=False
        num=0
        # self.qtlock.tryLock()
        # self.qtlock.unlock()
        while not self.stop_flag:
            # self.qtlock.lock()
            self.func()
            # self.qtlock.unlock()
            num+=1
            # time.sleep(0.01)

    # def changefunc(self,func):
    #     self.func = func

    def stop(self):
        self.stop_flag=True
        self.exit(0)

class MyWidget(QtGui.QWidget):
    function = None
    default_para=[]
    point_size=0.1
    file=file
    maxLine=10
    infotext=[]
    auto_try_flag=False
    if_best_flag=False
    div=[]
    sin = QtCore.pyqtSignal()
    rate=0
    color_set =[tuple([40,64,64,255])] + [tuple(list(i) + [255]) for i in np.random.randint(64, 256, size=[30, 3])]
    # 第一个颜色是噪声的颜色，在聚类中标为-1
    def __init__(self):
        super(MyWidget, self).__init__()
        #  用来表示点阵的区域
        w = pg.GraphicsLayoutWidget(show=True)
        self.__SetPlotWidget(w)
        #   标签
        title = QtWidgets.QLabel(self)
        title.setText("聚类算法")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lb1 = QtWidgets.QLabel(self)
        lb1.setText("文件名:")
        lb2 = QtWidgets.QLabel(self)
        lb2.setText("聚类方法:")
        lb3 = QtWidgets.QLabel(self)
        lb3.setText("参数:")
        lb4 = QtWidgets.QLabel(self)
        lb4.setText("反馈:")
        lb5 = QtWidgets.QLabel(self)
        lb5.setText("准确率:")
        lb6 = QtWidgets.QLabel(self)
        lb6.setText("用时:")
        lb7 = QtWidgets.QLabel(self)
        lb7.setText("描点大小：")
        #  参数控制
        self.para = [QtWidgets.QLabel(self) for i in range(3)]
        self.para_value=[QtWidgets.QLineEdit() for i in range(3)]
        for i in range(3):
            self.para[i].setText("--:")
            self.para_value[i].setText("")
        #  函数选择
        self.func_text = QtWidgets.QComboBox()
        self.func_text.addItems(["--"] + list(AlgorithmList.keys()))
        self.func_text.activated.connect(self.changefunc)
        #  特殊选项勾选
        self.ifbest = QtWidgets.QCheckBox("最佳模式",self)
        self.ifbest.stateChanged.connect(self.changeifbest)

        self.auto_retry = QtWidgets.QCheckBox("自动刷新", self)
        self.auto_retry.stateChanged.connect(self.changeAutoRetry)
        self.retry_thread = QtCore.QThread()
        self.retry_thread.started.connect(self.__autoretry)
        self.qtlock = QtCore.QMutex()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timeOut)
        self.threadd = Runthread(self._threadRetry)
        self.sin.connect(self.threadd.stop)
        #  编写文件名
        self.file_name=QtWidgets.QLineEdit()
        self.file_name.setText(self.file)
        self.file_name.editingFinished.connect(self.changefile)
        self.load_data(self.file)
        #  修改点大小
        self.sizeEdit = QtWidgets.QLineEdit()
        self.sizeEdit.setText(f"{self.point_size}")
        self.sizeEdit.editingFinished.connect(self.changesize)
        #  准确率显示
        self.Accuracy = QtWidgets.QLabel(self)
        self.Accuracy.setText("--")
        #  用时显示
        self.Time = QtWidgets.QLabel(self)
        self.Time.setText("--")

        # 按钮组
        self.btn1 = QtWidgets.QPushButton('Retry', self)
        self.btn1.clicked.connect(self.button_Retry)

        self.btn2 = QtWidgets.QPushButton('Check', self)
        self.btn2.clicked.connect(self.button_Check)
        self.info = QtWidgets.QTextEdit()


        ## 创建一个栅格Grid布局作为窗口的布局
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        ## 细节的控制每个区域所属的格子
        layout.addWidget(w, 0, 0, 0, 1)  # plot goes on right side, spanning all rows
        layout.addWidget(title, 0, 1, 1, 2)
        layout.addWidget(lb1, 1, 1)
        layout.addWidget(self.file_name, 1, 2)
        layout.addWidget(lb2, 2, 1)
        layout.addWidget(self.func_text, 2, 2)
        layout.addWidget(lb3, 3, 1)
        for i in range(3):
            layout.addWidget(self.para[i], 4+i, 1)
            layout.addWidget(self.para_value[i], 4+i, 2)

        layout.addWidget(self.btn1, 7, 1)   # button goes in upper-left
        layout.addWidget(self.btn2, 7, 2)   # button goes in upper-left
        layout.addWidget(lb4, 8, 1)
        layout.addWidget(self.info, 9, 1, 1, 2)  # list widget goes in bottom-left
        layout.addWidget(lb7, 10, 1)
        layout.addWidget(self.sizeEdit, 10, 2)
        layout.addWidget(lb5, 11, 1)
        layout.addWidget(self.Accuracy, 11, 2)
        layout.addWidget(lb6, 12, 1)
        layout.addWidget(self.Time, 12, 2)
        layout.addWidget(self.ifbest, 13, 1)
        layout.addWidget(self.auto_retry, 13, 2)

    def load_data(self,filename):
        train = pd.read_csv(filename, sep=' ', header=None)
        self.answer = train.iloc[:, 0].copy()
        self.point = train.iloc[:, 1:3].copy()
        # self.__show_info("数据加载完成")

    def Onetry(self):
        # print("good try!")
        if self.function == None:
            self.show_info("请选择聚类方法！")
            return False
        else:
            try:
                para = [float(i) for i in self.default_para]
                for i in range(self.para_num):
                    text = self.para_value[i].text()
                    if text != '':
                        para[i] = float(text)
            except Exception as z:
                self.show_info("请输入正确的参数值")
                print(z)
                return True
        t = time.time_ns()
        div, m = self.function(np.array(self.point), *para)
        Thetime=(time.time_ns()-t)/1e6
        rate = Coopcheckdiv(np.array(div), np.array(self.answer))
        if self.if_best_flag:
            if rate>self.rate or (rate==self.rate and Thetime<self.Thetime):
                self.rate=rate
                self.Accuracy.setText(f"{self.rate}")
                self.div = div
                self.m = m
                self.Thetime = Thetime
                self.Oneshow()
            else:
                return True
        self.div = div
        self.m = m
        self.Thetime = Thetime
        self.rate=rate
        return True

    def Oneshow(self):
        self.Time.setText(f"{self.Thetime}ms")
        self.plot_update(self.point, self.div, self.m)
        self.Accuracy.setText(f"{self.rate}")

    def plot_update(self, point, div, m):
        pointsize = self.point_size
        divnum = len(set(div))
        if divnum>30:
            self.show_info(f"类型数达到{divnum},请调参后尝试")
            return 8
        pos = np.array(point)
        # color_set=[tuple(list(i)+[255]) for i in np.random.randint(64,256,size=[divnum,3])]
        color_set = np.array(self.color_set[0:divnum+1],
                             dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])

        color = np.array([color_set[i+1] for i in div],
                         dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
        symbol = np.array(["o" if i >=0 else "t" for i in div])


        pos_m = np.array(m).reshape(-1, 2)
        color_m = np.array([color_set[i+1] for i in range(len(m))],
                           dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
        symbol_m = ['+']*len(m)
        symbols = np.hstack([symbol, symbol_m])
        # symbols[np.where()]
        sizes = [pointsize] * len(div) + [pointsize * 5] * len(m)

        self.g.setData(pos=np.vstack([pos, pos_m]), size=sizes, symbol=symbols, symbolBrush=np.hstack([color, color_m]),
                       pxMode=False)

    def button_Retry(self):
        if self.Onetry() :
            self.Oneshow()

    def button_Check(self):
        if len(self.div)==0:
            self.show_info("暂未分类")
            return 0
        # self.rate = Coopcheckdiv(np.array(self.div),np.array(self.answer))
        self.Accuracy.setText(f"{self.rate}")
        print(f"function name:{self.func_text.currentText()}")
        dic = validation(self.answer,self.div,True)
        self.show_info(str(dic))
        print(self.answer)

    def __SetPlotWidget(self, w):
        w.setWindowTitle('A Window')
        v = w.addViewBox()
        v.setAspectLocked()
        self.g = pg.GraphItem()
        v.addItem(self.g)

    def show_info(self, text):
        self.infotext.append(text)
        if len(self.infotext)>self.maxLine:
            self.infotext.pop(0)
        self.info.setPlainText("\n".join(self.infotext))

    def changeifbest(self):
        if self.ifbest.isChecked():
            self.btn2.setEnabled(False)
            self.if_best_flag=True
        else:
            self.btn2.setEnabled(True)
            self.if_best_flag=False
    def changeAutoRetry(self):
        if self.auto_retry.isChecked():
            if self.function == None:
                self.show_info("请选择聚类方法！")
                self.auto_retry.setChecked(False)
                # self.btn1.setEnabled(True)
                return 0
            self.btn1.setEnabled(False)
            self.auto_try_flag = True
            self.__autoretry()
        else:
            self.btn1.setEnabled(True)
            self.timer.stop()
            self.threadd.exit(0)
            self.sin.emit()
            self.auto_try_flag=False
    def __autoretry(self):
        # self.Retry()
        # print("__autoretry")
        self.timer.start(100)  #  fps
        self.qtlock.tryLock()
        self.qtlock.unlock()
        # print("now start")
        self.threadd.start()
    def _threadRetry(self):
        self.qtlock.lock()
        # print("yesyesyes")
        self.Onetry()
        # print("Ok Ok")
        self.qtlock.unlock()
    def timeOut(self):
        self.Oneshow()
    def changefunc(self):
        if self.func_text.itemText(0)== "--":
            self.func_text.removeItem(0)
        currect = self.func_text.currentText()
        temp = AlgorithmList[currect]
        self.function=temp["func"]
        # self.threadd.changefunc(temp["func"])
        self.para_num=0
        for i in temp["para"]:
            self.para[self.para_num].setText(i)
            self.para_value[self.para_num].setText(str(temp["para"][i]))
            self.para_num+=1
        for i in range(self.para_num,3):
            self.para[i].setText("--")
        self.default_para=list(temp["para"].values())
        self.show_info("函数修改完成")
        # print(temp)
    def changesize(self):
        text=self.sizeEdit.text()
        try:
            self.point_size=float(text)
            self.show_info("描点大小已修改")
        except Exception as z:
            self.sizeEdit.setText(str(self.point_size))
            self.show_info("请输入float类型数据！")
    def changefile(self):
        text = self.file_name.text()
        try:
            self.load_data(text)
            self.file = text
            self.show_info("文件已加载")
        except Exception as z:
            self.sizeEdit.setText(str(self.file))
            self.show_info("找不到该文件")

if __name__ == '__main__':
    filename=["spiral_unbalance.txt","three_cluster.txt","two_cluster.txt"]
    app = QtGui.QApplication([])
    w = MyWidget()
    w.show()
    ## Start the Qt event loop
    app.exec_()


