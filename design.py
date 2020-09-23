# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(790, 542)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(30, 30, 256, 192))
        self.listView.setObjectName("listView")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(390, 30, 256, 192))
        self.tableView.setObjectName("tableView")
        self.trackersIndex = QtWidgets.QLineEdit(self.centralwidget)
        self.trackersIndex.setGeometry(QtCore.QRect(40, 290, 113, 20))
        self.trackersIndex.setObjectName("trackersIndex")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 260, 131, 16))
        self.label.setObjectName("label")
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setGeometry(QtCore.QRect(20, 320, 75, 23))
        self.stopButton.setObjectName("stopButton")
        self.pauseButton = QtWidgets.QPushButton(self.centralwidget)
        self.pauseButton.setGeometry(QtCore.QRect(110, 320, 75, 23))
        self.pauseButton.setObjectName("pauseButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 790, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Tracking Optimizer"))
        self.trackersIndex.setInputMask(_translate("MainWindow", "1, 3-5, 2, *"))
        self.trackersIndex.setText(_translate("MainWindow", "1, 3-5, 2, *"))
        self.label.setText(_translate("MainWindow", "Enter indexes of trackers"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.pauseButton.setText(_translate("MainWindow", "Pause"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())