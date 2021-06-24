from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QAction, QApplication, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
                             QInputDialog, QLabel, QMainWindow, QPushButton, QTabWidget,
                             QTextEdit, QWidget)
from PyQt5.QtGui import QPainter, QIcon
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.io as sp
import numpy as np
import sys
import pandas as pd


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

        # Initialize arrays
        self.RAW = []
        self.HIGH = []
        self.Labels = []
        self.Interval_inspected = []
        self.sampling_rate = 25000

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab_preprocessing = QWidget()
        self.tab_train = QWidget()
        self.tab_detect = QWidget()
        self.tab_postprocessing = QWidget()
        self.tabs.resize(300, 200)

        # Figure to plot on
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

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

        #layout for second tab
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
            return self.sampling_rate

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
            print(self.RAW)
            #print(self.HIGH)
            #print(Labels)
            #print(Interval_inspected)
            #dict_keys(['RAW', 'HIGH', 'Labels', 'Interval_inspected'])
            self.update_plot()

            return self.RAW, self.HIGH, self.Labels, self.Interval_inspected


    # creating canvas and toolbar for second tab
    def create_select_cs_box(self):
        select_cs_layout = QGridLayout()
        select_cs_layout.setColumnStretch(0, 1)
        select_cs_layout.setColumnStretch(1, 1)

        toolbar = NavigationToolbar(self.canvas, self)

        plot_button = QPushButton('Plot')
        plot_button.clicked.connect(self.update_plot)

        select_cs_layout.addWidget(toolbar, 0, 0)
        select_cs_layout.addWidget(self.canvas, 1, 0)
        select_cs_layout.addWidget(plot_button, 0, 1)

        self.select_cs_box.setLayout(select_cs_layout)

    # updating plot for new data?
    def update_plot(self):
        data = self.RAW

        ax = self.figure.add_subplot(111)

        ax.clear()

        ax.plot(data, '*')

        self.canvas.draw()
        #plt.plot(np.arange(sampling_rate) / sampling_rate, RAW[0:sampling_rate])
        #plt.xlabel('time (s)')
        #plt.show()

# canvas initiation
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


def create():
    app = QApplication(sys.argv)
    main = Frame()
    main.show()
    sys.exit(app.exec_())

create()
