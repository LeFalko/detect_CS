
# from typing import List, Any

from PyQt5.QtWidgets import (QApplication, QComboBox, QDesktopWidget, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
                             QTextEdit, QWidget)
from PyQt5.QtGui import QPainter, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sp
import numpy as np
import sys
#from CS import detect_CS


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.high_axes = fig.add_subplot(211)
        self.high_axes.get_xaxis().set_visible(False)
        self.high_axes.set_ylabel('High-pass signal')
        self.lfp_axes = fig.add_subplot(212, sharex=self.high_axes, sharey=self.high_axes)
        self.lfp_axes.set_ylabel('Low field potential')
        #self.lfp_label_axes.get_yaxis().set_visible(False)
        self.lfp_axes.get_xaxis().set_visible(True)
        self.lfp_axes.set_xlabel('Time')

        super(MplCanvas, self).__init__(fig)


# Initializing GUi window and setting size
class Frame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 500
        self.top = 200
        self.width = 1600
        self.height = 900
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.title = 'CS Detection GUI'
        self.setWindowTitle(self.title)

        '''qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
'''
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
        self.PC_Counter = 0
        self.PC_Number = 10
        self.CSNumber = 10
        self.PC_Array = [[[0] * 2 for i in range(10)] for j in range(self.PC_Number)]
        self.sampling_rate = 25000
        self.x_values = [[0] * 2 for i in range(self.CSNumber)]
        self.value_counter = 0
        self.backwardscounter = 9
        self.Filename=[]
        self.names = []
        self.compLFP = []
        self.compHIGH = []
        self.compLABELS = []
        self.detect_LFP = []
        self.detect_HIGH = []
        #self.detect_CS = detect_CS


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
        self.select_cs_box = QGroupBox("Select complex spikes")

        # Add groupboxes to first tab
        self.create_data_input_box()
        self.create_information_box()
        self.create_select_cs_box()

        # layout for first tab
        self.tab_preprocessing.layout.addWidget(self.select_cs_box, 1, 0)
        self.tab_preprocessing.layout.addWidget(self.data_input_box, 0, 0)
        self.tab_preprocessing.layout.addWidget(self.information_box, 1, 1)
        self.tab_preprocessing.layout.setColumnStretch(0, 4)
        self.tab_preprocessing.layout.setRowStretch(0, 0)
        self.tab_preprocessing.layout.setRowStretch(1, 4)

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
        select_cs_layout.setColumnStretch(0, 0)
        select_cs_layout.setColumnStretch(1, 0)

        toolbar = NavigationToolbar(self.canvas, self)

        labeling_button = QPushButton('Select CS')
        labeling_button.clicked.connect(self.select_cs)

        delete_button = QPushButton('Delete last Selection')
        delete_button.clicked.connect(self.delete_last_CS)

        select_cs_layout.addWidget(toolbar, 0, 0)
        select_cs_layout.addWidget(self.canvas, 1, 0)
        select_cs_layout.addWidget(labeling_button, 0, 1)
        select_cs_layout.addWidget(delete_button, 0, 2)

        self.select_cs_box.setLayout(select_cs_layout)

    # creating a box in the first tab containing sampling rate input and file upload
    def create_data_input_box(self):
        data_input_layout = QGridLayout()

        sampling_button = QPushButton("Enter sampling rate")
        sampling_button.setToolTip('Choose your preferred sampling rate')
        sampling_button.clicked.connect(self.getInteger)

        upload_button = QPushButton("Choose PC for manual labeling")
        upload_button.setToolTip('Upload and plot the first file for labeling')
        upload_button.clicked.connect(self.openFileNameDialog)

        '''sampling_label = QLabel("Sampling rate:")
        upload_label = QLabel("Upload Files:")
        pc_number_label = QLabel("Number of Pc´s")'''

        data_input_layout.addWidget(sampling_button, 0, 0)
        data_input_layout.addWidget(upload_button, 0, 1)
        # data_input_layout.addWidget(pc_number_button, 0, 2)

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
                              "\n - Low-Pass: LFP (if not available then extract)\n"
                              "Ask for cut-off frequencies: upper cut off and lower cut off, "
                              "sampling rate or use default values (use from our paper)")

        information_layout.addWidget(textedit, 0, 0)

        self.information_box.setLayout(information_layout)

    # creating the box in the first tab containing the train button
    '''def create_training_set_box(self):
        create_set_layout = QGridLayout()
        create_set_layout.setColumnStretch(0, 2)
        create_set_layout.setColumnStretch(1, 2)

        train_button = QPushButton("Create training set")
        train_button.resize(300, 200)

        train_label = QLabel("Train Algorithm by \nmanually detecting \n10 complex spikes")

        create_set_layout.addWidget(train_label, 0, 0)
        create_set_layout.addWidget(train_button, 0, 1)

        self.training_set_box.setLayout(create_set_layout)'''
    # creating sampling input dialog
    def getInteger(self):
        i, okPressed = QInputDialog.getInt(self, "Enter sampling rate", "Sampling rate in Hz:", 25000, 0,
                                           100000, 1000)
        if okPressed:
            self.sampling_rate = i

    '''def getPCnumber(self):
        i, okPressed = QInputDialog.getInt(self, "Enter number of Pc´s", "How many different Pc´s?", 10, 0,
                                           100, 1)
        if okPressed:
            self.PC_Number = i
            self.PC_Array = [[0]*10 for i in range(self.PC_Number)]
    '''
    # creating file upload dialog
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.upload_data(fileName)

    def upload_data(self, fileName):
        #LFP, HIGH, Interval_inspected, Labels = load_data(fileName)
        #print(LFP, HIGH, Interval_inspected, Labels)
        mat = sp.loadmat(fileName)
        self.RAW = np.array(mat['RAW'])
        self.HIGH = np.array(mat['HIGH'])
        self.Labels = np.array(mat['Labels'])
        self.Interval_inspected = np.array(mat['Interval_inspected'])
        self.Filename = fileName
        self.plot_data()

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "train_data.mat",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            sp.savemat("train_data.mat", {'Names': self.names,
                                          'compLFP': self.RAW,
                                          'compHIGH': self.HIGH,
                                          'compLabels': self.Labels}, do_compression=True)
            mat = sp.loadmat("train_data.mat")
            print(mat)

    # updating plot for raw data
    def plot_data(self):
        raw_data = self.RAW[0]
        high_data = self.HIGH[0]
        #print(raw_data, high_data)

        time = np.arange(len(self.RAW[0]))

        self.canvas.high_axes.cla()
        self.canvas.lfp_axes.cla()
        self.canvas.high_axes.plot(time, high_data, 'r')
        self.canvas.lfp_axes.plot(time, raw_data, 'r')
        self.canvas.draw()

    # activates span selection
    def select_cs(self):
        self.span = SpanSelector(self.canvas.lfp_axes, self.onselect, 'horizontal', useblit=True,
                                 interactive=True, props=dict(alpha=0.5, facecolor='tab:blue'))

    # Loop for selecting CS and uploading new file
    def onselect(self, min_value, max_value):
        # while under 10 selected CS, assign both values to array x_values
        if self.value_counter < 10:
            self.x_values[self.value_counter][0] = int(min_value)
            self.x_values[self.value_counter][1] = int(max_value)
            self.value_counter += 1
            # print(min_value, max_value)
            print(self.value_counter)
            self.select_cs()
        # with the 10th CS, show messagebox
        else:
            replybox = QMessageBox.question(self, "Are all CS chosen correctly?",
                                            "Press yes to upload next file and no if you want to restart!",
                                            QMessageBox.Yes, QMessageBox.No)
            if replybox == QMessageBox.Yes:
                # if it is the 10th PC, save lists to train_data
                self.create_labels()
                if self.PC_Counter == 1:
                    self.saveFileDialog()
                    return

                # if its not the 10th PC: clear values, call concatenate and open new file
                self.PC_Array[self.PC_Counter] = self.x_values
                self.PC_Counter += 1
                self.x_values = [[0] * 2 for i in range(10)]
                self.value_counter = 0
                #self.concatenate_segments()
                self.openFileNameDialog()
                self.plot_data()

            # if not happy with CS, reset values and plot again
            elif replybox == QMessageBox.No:
                self.x_values = [[0] * 2 for i in range(10)]
                self.value_counter = 0
                self.plot_data()

    #TODO: implement correct delete last CS function
    def delete_last_CS(self):
        if self.x_values[self.backwardscounter][0] == 0:
            self.backwardscounter -= 1
        else:
            self.x_values[self.backwardscounter][0] = 0
            self.x_values[self.backwardscounter][1] = 0
            self.value_counter -= 1

    def create_labels(self):
        labels = np.zeros_like(self.RAW)
        for i in range(self.CSNumber):
            labels[0][self.x_values[i][0]:self.x_values[i][1]] = 1
        self.Labels = labels

    '''    # concatenate cs data in the file
    def concatenate_segments(self):

        starts = np.where(np.concatenate([0], np.diff(seg) == 1))[0]
        ends = np.where(np.concatenate(([0], np.diff(seg) == -1)))[0]
        if seg[-1] == 1:
            ends = np.concatenate(ends, len(seg) - 1)
        for s, e in zip(starts, ends):
            if sum(self.Labels[s:e]) == 0:
                seg[s:e] = False
        self.names.append(self.Filename)
        self.compLFP.append(self.RAW[seg])
        self.compHIGH.append(self.HIGH[seg])
        self.compLabels.append(self.Labels[seg])
        print(self.names, self.compLFP, self.compHIGH, self.compLABELS)
    '''
    # FUNCTIONS SECOND TAB
    # creating upload for files to detect on and plotting detected spikes third tab
    def create_detect_cs_box(self):
        detect_cs_layout = QGridLayout()
        detect_cs_layout.setColumnStretch(0, 0)
        detect_cs_layout.setColumnStretch(1, 0)

        detect_upload_button = QPushButton("Upload files to detect on")
        detect_upload_button.clicked.connect(self.upload_detection_file)

        detect_upload_weights_button = QPushButton("Upload your downloaded weights from Colab")
        detect_upload_weights_button.clicked.connect(self.upload_weights)

        detecting_button = QPushButton('Detect CS')
        # detecting_button.clicked.connect(self.detect_CS(self.detect_LFP, self.detect_HIGH, self.weights))

        #detect_cs_layout.addWidget(self.canvas2, 3, 0)
        detect_cs_layout.addWidget(detect_upload_button, 0, 1)
        detect_cs_layout.addWidget(detect_upload_weights_button, 1, 1)
        detect_cs_layout.addWidget(detecting_button, 2, 1)

        self.detect_cs_box.setLayout(detect_cs_layout)

    # creating file upload dialog

    def upload_detection_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            mat = sp.loadmat(fileName)
            self.detect_LFP = np.array(mat['RAW'])
            self.detect_HIGH = np.array(mat['HIGH'])

    def upload_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.weights = sp.loadmat(fileName)


    # TODO: Create Functions for detecting cs and uploading files, maybe postprocessing

    # FUNCTIONS THIRD TAB

def create():
    app = QApplication(sys.argv)
    main = Frame()
    main.show()
    sys.exit(app.exec_())


create()
