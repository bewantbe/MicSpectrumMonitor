# Form implementation generated from reading ui file 'control_pannel.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dock4(object):
    def setupUi(self, Dock4):
        Dock4.setObjectName("Dock4")
        Dock4.resize(553, 226)
        self.gridLayoutWidget = QtWidgets.QWidget(parent=Dock4)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 20, 479, 131))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(10, 0, 10, 10)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox_sr = QtWidgets.QComboBox(parent=self.gridLayoutWidget)
        self.comboBox_sr.setEditable(False)
        self.comboBox_sr.setObjectName("comboBox_sr")
        self.comboBox_sr.addItem("")
        self.comboBox_sr.addItem("")
        self.gridLayout.addWidget(self.comboBox_sr, 1, 3, 1, 1)
        self.label_ch = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_ch.setObjectName("label_ch")
        self.gridLayout.addWidget(self.label_ch, 2, 2, 1, 1)
        self.lineEdit_ch = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.lineEdit_ch.setObjectName("lineEdit_ch")
        self.gridLayout.addWidget(self.lineEdit_ch, 2, 3, 1, 1)
        self.label_datetime = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_datetime.setObjectName("label_datetime")
        self.gridLayout.addWidget(self.label_datetime, 4, 6, 1, 2)
        self.lineEdit_wavpath = QtWidgets.QLineEdit(parent=self.gridLayoutWidget)
        self.lineEdit_wavpath.setObjectName("lineEdit_wavpath")
        self.gridLayout.addWidget(self.lineEdit_wavpath, 3, 2, 1, 4)
        self.comboBox_nave = QtWidgets.QComboBox(parent=self.gridLayoutWidget)
        self.comboBox_nave.setEditable(False)
        self.comboBox_nave.setObjectName("comboBox_nave")
        self.comboBox_nave.addItem("")
        self.comboBox_nave.addItem("")
        self.comboBox_nave.addItem("")
        self.comboBox_nave.addItem("")
        self.comboBox_nave.addItem("")
        self.comboBox_nave.addItem("")
        self.comboBox_nave.addItem("")
        self.gridLayout.addWidget(self.comboBox_nave, 1, 6, 1, 1)
        self.comboBox_dev = QtWidgets.QComboBox(parent=self.gridLayoutWidget)
        self.comboBox_dev.setEditable(False)
        self.comboBox_dev.setObjectName("comboBox_dev")
        self.comboBox_dev.addItem("")
        self.comboBox_dev.addItem("")
        self.comboBox_dev.addItem("")
        self.gridLayout.addWidget(self.comboBox_dev, 0, 3, 1, 1)
        self.comboBox_ftlen = QtWidgets.QComboBox(parent=self.gridLayoutWidget)
        self.comboBox_ftlen.setEditable(False)
        self.comboBox_ftlen.setObjectName("comboBox_ftlen")
        self.comboBox_ftlen.addItem("")
        self.comboBox_ftlen.addItem("")
        self.comboBox_ftlen.addItem("")
        self.comboBox_ftlen.addItem("")
        self.comboBox_ftlen.addItem("")
        self.comboBox_ftlen.addItem("")
        self.gridLayout.addWidget(self.comboBox_ftlen, 0, 6, 1, 1)
        self.pushButton_screenshot = QtWidgets.QPushButton(parent=self.gridLayoutWidget)
        self.pushButton_screenshot.setObjectName("pushButton_screenshot")
        self.gridLayout.addWidget(self.pushButton_screenshot, 1, 7, 1, 1)
        self.label_nave = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_nave.setObjectName("label_nave")
        self.gridLayout.addWidget(self.label_nave, 1, 5, 1, 1)
        self.label_sr = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_sr.setObjectName("label_sr")
        self.gridLayout.addWidget(self.label_sr, 1, 2, 1, 1)
        self.label_dev = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_dev.setObjectName("label_dev")
        self.gridLayout.addWidget(self.label_dev, 0, 2, 1, 1)
        self.label_ftlen = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_ftlen.setObjectName("label_ftlen")
        self.gridLayout.addWidget(self.label_ftlen, 0, 5, 1, 1)
        self.pushButton_mon = QtWidgets.QPushButton(parent=self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_mon.sizePolicy().hasHeightForWidth())
        self.pushButton_mon.setSizePolicy(sizePolicy)
        self.pushButton_mon.setMinimumSize(QtCore.QSize(110, 0))
        self.pushButton_mon.setObjectName("pushButton_mon")
        self.gridLayout.addWidget(self.pushButton_mon, 0, 7, 1, 1)
        self.pushButton_rec = QtWidgets.QPushButton(parent=self.gridLayoutWidget)
        self.pushButton_rec.setObjectName("pushButton_rec")
        self.gridLayout.addWidget(self.pushButton_rec, 3, 7, 1, 1)
        self.toolButton_path = QtWidgets.QToolButton(parent=self.gridLayoutWidget)
        self.toolButton_path.setObjectName("toolButton_path")
        self.gridLayout.addWidget(self.toolButton_path, 3, 6, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 4, 1, 1)
        self.label_rec_remain = QtWidgets.QLabel(parent=self.gridLayoutWidget)
        self.label_rec_remain.setObjectName("label_rec_remain")
        self.gridLayout.addWidget(self.label_rec_remain, 4, 2, 1, 4)

        self.retranslateUi(Dock4)
        QtCore.QMetaObject.connectSlotsByName(Dock4)
        Dock4.setTabOrder(self.comboBox_dev, self.comboBox_sr)
        Dock4.setTabOrder(self.comboBox_sr, self.lineEdit_ch)
        Dock4.setTabOrder(self.lineEdit_ch, self.comboBox_ftlen)
        Dock4.setTabOrder(self.comboBox_ftlen, self.comboBox_nave)
        Dock4.setTabOrder(self.comboBox_nave, self.pushButton_mon)
        Dock4.setTabOrder(self.pushButton_mon, self.pushButton_screenshot)
        Dock4.setTabOrder(self.pushButton_screenshot, self.lineEdit_wavpath)
        Dock4.setTabOrder(self.lineEdit_wavpath, self.toolButton_path)
        Dock4.setTabOrder(self.toolButton_path, self.pushButton_rec)

    def retranslateUi(self, Dock4):
        _translate = QtCore.QCoreApplication.translate
        Dock4.setWindowTitle(_translate("Dock4", "Form"))
        self.comboBox_sr.setCurrentText(_translate("Dock4", "48kHz"))
        self.comboBox_sr.setItemText(0, _translate("Dock4", "48kHz"))
        self.comboBox_sr.setItemText(1, _translate("Dock4", "250kHz"))
        self.label_ch.setText(_translate("Dock4", "Channels: "))
        self.lineEdit_ch.setText(_translate("Dock4", "1, 2"))
        self.lineEdit_ch.setPlaceholderText(_translate("Dock4", "comma seperated list of channel indexes"))
        self.label_datetime.setText(_translate("Dock4", "YYYY-MM-DD HH:MM:SS"))
        self.lineEdit_wavpath.setPlaceholderText(_translate("Dock4", "WAV path for recording"))
        self.comboBox_nave.setCurrentText(_translate("Dock4", "1"))
        self.comboBox_nave.setItemText(0, _translate("Dock4", "1"))
        self.comboBox_nave.setItemText(1, _translate("Dock4", "2"))
        self.comboBox_nave.setItemText(2, _translate("Dock4", "4"))
        self.comboBox_nave.setItemText(3, _translate("Dock4", "8"))
        self.comboBox_nave.setItemText(4, _translate("Dock4", "16"))
        self.comboBox_nave.setItemText(5, _translate("Dock4", "32"))
        self.comboBox_nave.setItemText(6, _translate("Dock4", "64"))
        self.comboBox_dev.setCurrentText(_translate("Dock4", "System mic"))
        self.comboBox_dev.setItemText(0, _translate("Dock4", "System mic"))
        self.comboBox_dev.setItemText(1, _translate("Dock4", "AD7606C"))
        self.comboBox_dev.setItemText(2, _translate("Dock4", "refresh list"))
        self.comboBox_ftlen.setCurrentText(_translate("Dock4", "1024"))
        self.comboBox_ftlen.setItemText(0, _translate("Dock4", "1024"))
        self.comboBox_ftlen.setItemText(1, _translate("Dock4", "2048"))
        self.comboBox_ftlen.setItemText(2, _translate("Dock4", "4096"))
        self.comboBox_ftlen.setItemText(3, _translate("Dock4", "8192"))
        self.comboBox_ftlen.setItemText(4, _translate("Dock4", "16384"))
        self.comboBox_ftlen.setItemText(5, _translate("Dock4", "32768"))
        self.pushButton_screenshot.setText(_translate("Dock4", "Window-shot"))
        self.label_nave.setText(_translate("Dock4", "n average:"))
        self.label_sr.setText(_translate("Dock4", "Sampling rate: "))
        self.label_dev.setText(_translate("Dock4", "Device: "))
        self.label_ftlen.setText(_translate("Dock4", "FFT length:"))
        self.pushButton_mon.setText(_translate("Dock4", "Start monitoring"))
        self.pushButton_rec.setText(_translate("Dock4", "Start recording"))
        self.toolButton_path.setText(_translate("Dock4", "Browse"))
        self.label_rec_remain.setText(_translate("Dock4", "Rec: XX:XX:XX  (XX;XX:XX left)"))
