from typing import List, Any

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QAction, QApplication, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
                             QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
                             QTextEdit, QWidget)
from PyQt5.QtGui import QPainter, QIcon
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.io as sp
import numpy as np
import sys


# canvas initiation
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=40, height=20, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(211)
        self.axes2 = fig.add_subplot(212, sharex=self.axes, sharey=self.axes)
        super(MplCanvas, self).__init__(fig)


# Initializing GUi window and setting size
class Frame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 1000
        self.top = 400
        self.width = 1600
        self.height = 900
        self.title = 'CS Detection GUI'
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.table_widget = Content(self)
        self.setCentralWidget(self.table_widget)


# creating content for the Frame
class Content(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QGridLayout()
        "self.popup_window = QWidget()"

        # Initialize arrays and helper values
        self.RAW = []
        self.HIGH = []
        self.Labels = []
        self.Interval_inspected = []
        self.sampling_rate = 25000
        self.x_values = [[0] * 2 for i in range(10)]
        self.value_counter = 0

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab_preprocessing = QWidget()
        self.tab_train = QWidget()
        self.tab_detect = QWidget()
        self.tab_postprocessing = QWidget()
        self.tabs.resize(300, 200)

        # Figure to plot on
        self.canvas = MplCanvas(self, width=40, height=20, dpi=100)
        self.canvas.setParent(self)

        # Add tabs
        self.tabs.addTab(self.tab_preprocessing, "Pre-processing")
        self.tabs.addTab(self.tab_train, "Train CNN")
        self.tabs.addTab(self.tab_detect, "Detect CS")
        self.tabs.addTab(self.tab_postprocessing, "Post-processing")

        # Create first tab
        self.tab_preprocessing.layout = QGridLayout(self)

        # Groupboxes
        self.data_input_box = QGroupBox("Data input")
        self.information_box = QGroupBox("Please note:")
        self.training_set_box = QGroupBox("Train")

        # Add groupboxes to first tab
        self.create_data_input_box()
        self.create_information_box()
        self.create_training_set_box()

        # layout for first tab
        self.tab_preprocessing.layout.addWidget(self.data_input_box, 1, 0)
        self.tab_preprocessing.layout.addWidget(self.information_box, 0, 1)
        self.tab_preprocessing.layout.addWidget(self.training_set_box, 1, 1)
        self.tab_preprocessing.layout.setColumnStretch(0, 4)

        # self.tab_preprocessing.layout.setRowStretch(0, 4)
        self.tab_preprocessing.setLayout(self.tab_preprocessing.layout)

        # Create second tab
        self.tab_train.layout = QGridLayout(self)

        # Groupboxes
        self.select_cs_box = QGroupBox("Select complex spikes")

        # Add Groupboxes to second tab
        self.create_select_cs_box()

        # layout for second tab
        self.tab_train.layout.addWidget(self.select_cs_box, 0, 0)

        self.tab_train.setLayout(self.tab_train.layout)

        # Create third tab
        self.tab_detect.layout = QGridLayout(self)

        self.tab_detect.setLayout(self.tab_detect.layout)

        # Create last tab
        self.tab_postprocessing.layout = QGridLayout(self)

        self.tab_postprocessing.setLayout(self.tab_postprocessing.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    # creating a box in the first tab containing sampling rate input and file upload
    def create_data_input_box(self):
        data_input_layout = QGridLayout()
        data_input_layout.setColumnStretch(0, 4)
        data_input_layout.setColumnStretch(1, 4)

        sampling_button = QPushButton("Choose sampling rate")
        sampling_button.setToolTip('Enter your sampling rate')
        sampling_button.clicked.connect(self.getInteger)

        upload_button = QPushButton("Choose PC data for training")
        upload_button.clicked.connect(self.openFileNameDialog)

        sampling_label = QLabel("Sampling rate:")
        upload_label = QLabel("Upload Files:")

        data_input_layout.addWidget(sampling_label, 0, 0)
        data_input_layout.addWidget(sampling_button, 0, 1)
        data_input_layout.addWidget(upload_label, 1, 0)
        data_input_layout.addWidget(upload_button, 1, 1)

        self.data_input_box.setLayout(data_input_layout)

    # creating the box in the first tab containing user information
    def create_information_box(self):
        information_layout = QGridLayout()
        information_layout.setColumnStretch(1, 1)
        information_layout.setRowStretch(1, 1)

        textedit = QTextEdit()
        textedit.resize(300, 200)
        textedit.setPlainText("Separate files individual PCs (unfiltered raw data) \n"
                              "Name the variables as: \n  - High-Pass: action potentials \n If LFP is available use: "
                              "\n - Low-Pass: LFP (if not avaialable then extract)\n"
                              "Ask for cut-off frequencies: upper cut off and lower cut off, "
                              "sampling rate or use default values (use from our paper)")

        information_layout.addWidget(textedit, 0, 0)

        self.information_box.setLayout(information_layout)

    # creating the box in the first tab containing the train button
    def create_training_set_box(self):
        create_set_layout = QGridLayout()
        create_set_layout.setColumnStretch(0, 2)
        create_set_layout.setColumnStretch(1, 2)

        train_button = QPushButton("Create training set")
        train_button.resize(300, 200)
        # train_button.clicked.connect(self.tab_train)

        train_label = QLabel("Train Algorithm by \nmanually detecting \n10 complex spikes")

        create_set_layout.addWidget(train_label, 0, 0)
        create_set_layout.addWidget(train_button, 0, 1)

        self.training_set_box.setLayout(create_set_layout)

    # creating sampling input dialog
    def getInteger(self):
        i, okPressed = QInputDialog.getInt(self, "Enter sampling rate", "Sampling rate in Hz:", 25000, 0,
                                           100000, 1000)
        if okPressed:
            self.sampling_rate = i

    # creating file upload dialog
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            mat = sp.loadmat(fileName)
            self.RAW = np.array(mat['RAW'])
            self.HIGH = np.array(mat['HIGH'])
            self.Labels = np.array(mat['Labels'])
            self.Interval_inspected = np.array(mat['Interval_inspected'])

    # creating canvas and toolbar for second tab
    def create_select_cs_box(self):
        select_cs_layout = QGridLayout()
        select_cs_layout.setColumnStretch(0, 1)
        select_cs_layout.setColumnStretch(1, 0)

        toolbar = NavigationToolbar(self.canvas, self)

        plot_button = QPushButton('Plot')
        plot_button.clicked.connect(self.plot_data)

        labeling_button = QPushButton('Select CS')
        labeling_button.clicked.connect(self.select_cs)

        select_cs_layout.addWidget(toolbar, 0, 0)
        select_cs_layout.addWidget(self.canvas, 1, 0)
        select_cs_layout.addWidget(plot_button, 0, 1)
        select_cs_layout.addWidget(labeling_button, 0, 2)

        self.select_cs_box.setLayout(select_cs_layout)

    # updating plot for raw data
    def plot_data(self):
        raw_data = self.RAW[0]
        high_data = self.HIGH[0]
        time = np.arange(len(self.RAW[0]))

        self.canvas.axes.set_title('select timespan for cs')

        self.canvas.axes.cla()
        self.canvas.axes2.cla()
        self.canvas.axes.plot(time, raw_data, 'r')
        self.canvas.axes2.plot(time, high_data, 'r')
        self.canvas.draw()

    def select_cs(self):
        self.span = SpanSelector(self.canvas.axes, self.onselect, 'horizontal', useblit=True,
                                 span_stays=True, rectprops=dict(alpha=0.5, facecolor='tab:blue'))

    def onselect(self, min_value, max_value):
        if self.value_counter < 10:
            self.x_values[self.value_counter][0] = min_value
            self.x_values[self.value_counter][1] = max_value
            self.value_counter += 1
            print(min_value, max_value)
            print(self.value_counter)
            self.select_cs()
        else:
            replybox = QMessageBox.question(self, "Are all CS chosen correctly?",
                                            "Press yes if you are happy and no if you want to restart!",
                                            QMessageBox.Yes, QMessageBox.No)
            if replybox == QMessageBox.Yes:
                print(self.x_values)
            elif replybox == QMessageBox.No:
                self.x_values = [[0] * 2 for i in range(10)]
                self.value_counter = 0
                self.plot_data()


def create():
    app = QApplication(sys.argv)
    main = Frame()
    main.show()
    sys.exit(app.exec_())


create()
