
# from typing import List, Any

from PyQt5.QtWidgets import (QAction, QApplication, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
                             QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
                             QTextEdit, QWidget)
from PyQt5.QtGui import QPainter, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
# import mat4py as m4p
import scipy as sp
import numpy as np
import sys
# from CS import load_data, concatenate_segments, norm_LFP, norm_high_pass, butter_bandpass


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.high_axes = fig.add_subplot(311)
        self.high_axes.get_xaxis().set_visible(False)
        self.lfp_axes = fig.add_subplot(312, sharex=self.high_axes, sharey=self.high_axes)
        self.lfp_axes.get_xaxis().set_visible(False)
        self.label_axes = fig.add_subplot(313, sharex=self.high_axes, sharey=self.high_axes)
        self.label_axes.set_xlabel("Time")
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
        self.popup_window = QWidget()

        # Initialize arrays and helper values
        self.RAW = []
        self.HIGH = []
        self.Labels = []
        self.Interval_inspected = []
        self.PC_Number = 10
        self.PC_Array = [[0]*10 for i in range(self.PC_Number)]
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

        # Figures to plot on
        self.canvas = MplCanvas(self, width=40, height=20, dpi=100)
        self.canvas.setParent(self)

        self.canvas2 = MplCanvas(self, width=40, height=20, dpi=100)
        self.canvas2.setParent(self)

        # Add tabs
        self.tabs.addTab(self.tab_preprocessing, "Pre-processing")
        self.tabs.addTab(self.tab_detect, "Detect CS")
        self.tabs.addTab(self.tab_postprocessing, "Post-processing")

        # Create first tab
        self.tab_preprocessing.layout = QGridLayout(self)

        # Groupboxes
        self.data_input_box = QGroupBox("Data input")
        self.information_box = QGroupBox("Please note:")
        self.training_set_box = QGroupBox("Train")
        self.select_cs_box = QGroupBox("Select complex spikes")

        # Add groupboxes to first tab
        self.create_data_input_box()
        self.create_information_box()
        self.create_training_set_box()
        self.create_select_cs_box()

        # layout for first tab
        self.tab_preprocessing.layout.addWidget(self.select_cs_box, 0, 0)
        self.tab_preprocessing.layout.addWidget(self.data_input_box, 1, 0)
        self.tab_preprocessing.layout.addWidget(self.information_box, 0, 1)
        self.tab_preprocessing.layout.addWidget(self.training_set_box, 1, 1)
        self.tab_preprocessing.layout.setColumnStretch(0, 4)

        # self.tab_preprocessing.layout.setRowStretch(0, 4)
        self.tab_preprocessing.setLayout(self.tab_preprocessing.layout)

        # Create second tab
        self.tab_detect.layout = QGridLayout(self)

        # Groupboxes
        self.detect_cs_box = QGroupBox("Detect complex spikes")

        # Add Groupboxes to second tab
        self.create_detect_cs_box()

        # layout for second tab
        self.tab_detect.layout.addWidget(self.detect_cs_box, 0, 0)
        self.tab_detect.layout.setColumnStretch(0, 1)

        self.tab_detect.setLayout(self.tab_detect.layout)

        # Create last tab
        self.tab_postprocessing.layout = QGridLayout(self)

        self.tab_postprocessing.setLayout(self.tab_postprocessing.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    # FUNCTIONS FIRST TAB

    # creating canvas and toolbar for first tab
    def create_select_cs_box(self):
        select_cs_layout = QGridLayout()
        select_cs_layout.setColumnStretch(0, 2)
        select_cs_layout.setColumnStretch(1, 0)

        toolbar = NavigationToolbar(self.canvas, self)

        plot_button = QPushButton('Plot')
        plot_button.clicked.connect(self.plot_data)
        plot_button.resize(300, 200)

        labeling_button = QPushButton('Select CS')
        labeling_button.clicked.connect(self.select_cs)

        select_cs_layout.addWidget(toolbar, 0, 0)
        select_cs_layout.addWidget(self.canvas, 1, 0)
        select_cs_layout.addWidget(plot_button, 0, 1)
        select_cs_layout.addWidget(labeling_button, 0, 2)

        self.select_cs_box.setLayout(select_cs_layout)

    # creating a box in the first tab containing sampling rate input and file upload
    def create_data_input_box(self):
        data_input_layout = QGridLayout()
        data_input_layout.setColumnStretch(0, 4)
        data_input_layout.setColumnStretch(1, 4)

        pc_number_button = QPushButton("Choose number of Purkinje cells")
        pc_number_button.setToolTip('Enter the number of files you want to train the algorithm with')
        pc_number_button.clicked.connect(self.getPCnumber)

        sampling_button = QPushButton("Choose sampling rate")
        sampling_button.setToolTip('Enter your sampling rate')
        sampling_button.clicked.connect(self.getInteger)

        upload_button = QPushButton("Choose PC data for training")
        upload_button.clicked.connect(self.openFileNameDialog)

        sampling_label = QLabel("Sampling rate:")
        upload_label = QLabel("Upload Files:")
        pc_number_label = QLabel("Number of Pc´s")

        data_input_layout.addWidget(sampling_label, 0, 0)
        data_input_layout.addWidget(sampling_button, 0, 1)
        data_input_layout.addWidget(upload_label, 1, 0)
        data_input_layout.addWidget(upload_button, 1, 1)
        data_input_layout.addWidget(pc_number_label, 2, 0)
        data_input_layout.addWidget(pc_number_button, 2, 1)

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

    def getPCnumber(self):
        i, okPressed = QInputDialog.getInt(self, "Enter number of Pc´s", "How many different Pc´s?", 10, 0,
                                           100, 1)
        if okPressed:
            self.PC_Number = i
            self.PC_Array = [[0]*10 for i in range(self.PC_Number)]

    # creating file upload dialog
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            mat = sp.loadmat(fileName)
            print(mat)
            self.RAW = np.array(mat['RAW'])
            self.HIGH = np.array(mat['HIGH'])
            self.Labels = np.array(mat['Labels'])
            self.Interval_inspected = np.array(mat['Interval_inspected'])

    # updating plot for raw data
    def plot_data(self):
        #raw_data = self.LFP
        raw_data = self.RAW[0]
        high_data = self.HIGH[0]

        time = np.arange(len(self.RAW[0]))

        self.canvas.axes.set_title('select timespan for cs')

        self.canvas.axes.cla()
        self.canvas.axes2.cla()
        self.canvas.axes.plot(time, high_data, 'r')
        self.canvas.axes2.plot(time, raw_data, 'r')
        self.canvas.draw()

    def select_cs(self):
        self.span = SpanSelector(self.canvas.axes, self.onselect, 'horizontal', useblit=True,
                                 span_stays=True, rectprops=dict(alpha=0.5, facecolor='tab:blue'))

    def onselect(self, min_value, max_value):
        array_index = 0
        #TODO: Stepback? click to delete last or sth
        if self.value_counter < 10:
            self.x_values[self.value_counter][0] = min_value
            self.x_values[self.value_counter][1] = max_value
            self.value_counter += 1
            print(min_value, max_value)
            print(self.value_counter)
            self.select_cs()
        else:
            replybox = QMessageBox.question(self, "Are all CS chosen correctly?",
                                            "Press yes to upload next file and no if you want to restart!",
                                            QMessageBox.Yes, QMessageBox.No)
            if replybox == QMessageBox.Yes:
                i = 0
                if i < 10:
                    # self.PC_Array[] = self.x_values
                    i += 1

            elif replybox == QMessageBox.No:
                self.x_values = [[0] * 2 for i in range(10)]
                self.value_counter = 0
                self.plot_data()

    # FUNCTIONS SECOND TAB

    # creating upload for files to detect on and plotting detected spikes third tab
    def create_detect_cs_box(self):
        detect_cs_layout = QGridLayout()
        detect_cs_layout.setColumnStretch(0, 0)
        detect_cs_layout.setColumnStretch(1, 0)

        detect_upload_button = QPushButton("Upload files to detect on")
        #detect_upload_button.clicked.connect(self.openFileNameDialog)

        detect_upload_weights_button = QPushButton("Upload your downloaded weights from Colab")
        #detect_upload_weights_button.clicked.connect(self.openFileNameDialog)

        labeling_button = QPushButton('Detect CS')
        #labeling_button.clicked.connect()

        # detect_cs_layout.addWidget(self.canvas2, 3, 0)
        detect_cs_layout.addWidget(detect_upload_button, 0, 1)
        detect_cs_layout.addWidget(detect_upload_weights_button, 1, 1)
        detect_cs_layout.addWidget(labeling_button, 2, 1)

        self.detect_cs_box.setLayout(detect_cs_layout)

    # TODO: Create Functions for detecting cs and uploading files, postprocessing

    # FUNCTIONS THIRD TAB

def create():
    app = QApplication(sys.argv)
    main = Frame()
    main.show()
    sys.exit(app.exec_())


create()
