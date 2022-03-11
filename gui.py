
# from typing import List, Any

from PyQt5.QtWidgets import (QApplication, QComboBox, QDesktopWidget, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QVBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
                             QTextEdit, QWidget, QCheckBox)
from PyQt5.QtGui import QPainter, QIcon, QDesktopServices
from PyQt5.QtCore import QUrl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sp
import numpy as np
import sys
from CS import detect_CS, norm_LFP, norm_high_pass, get_field_mat
import uneye

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.high_axes = fig.add_subplot(211)
        self.high_axes.get_xaxis().set_visible(False)
        # self.high_axes.set_ylabel('High-pass signal')
        self.lfp_axes = fig.add_subplot(212, sharex=self.high_axes, sharey=self.high_axes)
        # self.lfp_axes.set_ylabel('Low field potential')
        #self.lfp_label_axes.get_yaxis().set_visible(False)
        self.lfp_axes.get_xaxis().set_visible(True)
        self.lfp_axes.set_xlabel('Time')

        super(MplCanvas, self).__init__(fig)

class MplCanvas2(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig2 = Figure(figsize=(width, height), dpi=dpi)
        self.CS = fig2.add_subplot(221)
        self.LFP = fig2.add_subplot(223)
        # self.clusters.get_xaxis().set_visible(False)
        # self.high_axes.set_ylabel('High-pass signal')
        self.clusters = fig2.add_subplot(222)
        # self.lfp_axes.set_ylabel('Low field potential')
        self.SS = fig2.add_subplot(224)


        super(MplCanvas2, self).__init__(fig2)


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
        self.LFP_varname = 'RAW'
        self.HIGH_varname = 'HIGH'
        self.SS_varname = 'SS'
        self.LFP = []
        self.HIGH = []
        self.Labels = []
        self.Interval_inspected = []
        self.upload_LFP = []
        self.upload_HIGH = []
        self.PC_Counter = 0
        self.PC_Number = 10
        self.CSNumber = 10
        self.PC_Array = [[[0] * 2 for i in range(10)] for j in range(self.PC_Number)]
        self.sampling_rate = 25000
        self.sampling_rate_SS = 1000
        self.x_values = [[0] * 2 for i in range(self.CSNumber)]
        self.value_counter = 0
        self.backwardscounter = 9
        self.ID = []
        self.detect_LFP = []
        self.detect_HIGH = []
        self.weights = []
        self.output = []
        self.CS_onset = []
        self.CS_offset = []
        self.cluster_ID = []
        self.embedding = []
        self.n_clusters = []
        self.ss_train = []
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
                       'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 
                       'tab:olive', 'tab:cyan']
        # [#1f77b4ff]

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

        self.canvas2 = MplCanvas2(self, width=40, height=20, dpi=100)
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

        # Create third tab
        self.tab_postprocessing.layout = QGridLayout(self)

        # Groupboxes
        self.select_show_data_box = QGroupBox("Select clusters to show")
        self.cluster_plotting_box = QGroupBox("Plotting")

        # Add Groupboxes to third tab
        self.create_show_data_box()
        self.create_cluster_plotting_box()

        # Layout for third tab
        self.tab_postprocessing.layout.addWidget(self.select_show_data_box, 0, 0)
        self.tab_postprocessing.layout.addWidget(self.cluster_plotting_box, 0, 1)

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

        next_cell_button = QPushButton('Proceed to next cell')
        next_cell_button.clicked.connect(self.goto_next_cell)

        save_button = QPushButton('Save your selected CS')
        save_button.clicked.connect(self.saveFileDialog)

        select_cs_layout.addWidget(toolbar, 0, 0)
        select_cs_layout.addWidget(self.canvas, 2, 0)
        select_cs_layout.addWidget(labeling_button, 0, 1)
        select_cs_layout.addWidget(delete_button, 0, 2)
        select_cs_layout.addWidget(next_cell_button, 1, 1)
        select_cs_layout.addWidget(save_button, 1, 2)

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

        goto_Colab_button = QPushButton("AFTER LABELING: go to colab sheet to train algorithm")
        goto_Colab_button.setToolTip('Plase finish labeling data before going to the website')
        goto_Colab_button.clicked.connect(self.open_Colab)

        information_layout.addWidget(textedit, 0, 0)
        information_layout.addWidget(goto_Colab_button, 1, 0)

        self.information_box.setLayout(information_layout)

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
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload data", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.upload_data(fileName)

    def upload_data(self, fileName):
        mat = sp.loadmat(fileName)
#         self.upload_LFP = np.array(mat['RAW'])
#         self.upload_HIGH = np.array(mat['HIGH'])
        self.upload_LFP = np.array(mat[self.LFP_varname])
        self.upload_HIGH = np.array(mat[self.HIGH_varname])
        self.ID = np.append(self.ID, fileName)
        self.LFP.append(self.upload_LFP[0])
        self.HIGH.append(self.upload_HIGH[0])
        self.plot_data()

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save file", "train_data.mat",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        if fileName:
            self.create_labels()
            sp.savemat(fileName, {'ID': np.array(self.ID, dtype=object),
                                  'LFP': self.LFP,
                                  'HIGH': self.HIGH,
                                  'Labels': self.Labels}, do_compression=True)

    def open_Colab(self):
        QDesktopServices.openUrl(QUrl('https://colab.research.google.com/drive/1g1MzZz5h30Uov9tIbrarwwm02WD7xU6B#scrollTo=plKVE-vH_SLt'))

    # updating plot for raw data
    def plot_data(self):
        raw_data = self.upload_LFP[0]
        high_data = self.upload_HIGH[0]
        #print(raw_data, high_data)

        time = np.linspace(0, len(self.upload_LFP[0])/self.sampling_rate, len(self.upload_LFP[0]))

        self.canvas.high_axes.cla()
        self.canvas.lfp_axes.cla()
        self.canvas.high_axes.plot(time, high_data, 'tab:blue', lw=0.4)
        self.canvas.high_axes.set_ylabel('High-pass signal')
        self.canvas.lfp_axes.plot(time, raw_data, 'tab:blue', lw=0.4)
        self.canvas.lfp_axes.set_ylabel('Low field potential')
        self.canvas.lfp_axes.set_xlabel('time [s]')
        self.canvas.draw()

    # activates span selection
    def select_cs(self):
        self.span = SpanSelector(self.canvas.lfp_axes, self.onselect, 'horizontal', useblit=True,
                                 interactive=True, props=dict(alpha=0.5, facecolor='tab:blue'))

    # Loop for selecting CS and uploading new file
    def onselect(self, min_value, max_value):
        # while under 10 selected CS, assign both values to array x_values
        if self.value_counter < self.CSNumber:
            self.x_values[self.value_counter][0] = int(min_value)
            self.x_values[self.value_counter][1] = int(max_value)
            self.value_counter += 1
            print(min_value, max_value)
            print(self.value_counter)
            self.select_cs()
        # with the 10th CS, show messagebox
        else:
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Maximum amount of CS chosen.")
            errorbox.setText("Increase Number of CS per Cell or move to the next one!")
            errorbox.exec_()

    def goto_next_cell(self):
        replybox = QMessageBox.question(self, "Are all CS chosen correctly?",
                                        "Press yes to upload next file and no if you want to restart!",
                                        QMessageBox.Yes, QMessageBox.No)
        if replybox == QMessageBox.Yes:
            self.create_labels()
            self.PC_Array[self.PC_Counter] = self.x_values
            self.PC_Counter += 1
            self.x_values = [[0] * 2 for i in range(10)]
            self.value_counter = 0
            self.openFileNameDialog()

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
        labels = np.zeros_like(self.upload_LFP[0])
        for i in range(self.CSNumber):
            labels[self.x_values[i][0]:self.x_values[i][1]] = 1
        self.Labels.append(labels)

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
        detecting_button.clicked.connect(self.detect_CS_starter)

        detect_cs_layout.addWidget(detect_upload_button, 0, 1)
        detect_cs_layout.addWidget(detect_upload_weights_button, 1, 1)
        detect_cs_layout.addWidget(detecting_button, 2, 1)

        self.detect_cs_box.setLayout(detect_cs_layout)

    # creating file upload dialog

    def upload_detection_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload files for CS detection", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            mat = sp.loadmat(fileName)
            self.detect_LFP = get_field_mat(mat,['RAW'])
            self.detect_LFP = norm_LFP(self.detect_LFP, self.sampling_rate)
            self.detect_HIGH = get_field_mat(mat, ['HIGH'])
            self.detect_HIGH = norm_high_pass(self.detect_HIGH)
            self.mat = mat

    def upload_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload weights", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.weights = fileName

    def detect_CS_starter(self):
        runningbox = QMessageBox()
        runningbox.setWindowTitle("Running")
        runningbox.setText("Detecting complex spikes....")
        runningbox.exec_()
        
        output = detect_CS(self.weights, self.detect_LFP, self.detect_HIGH)
        runningbox.done(1)
        
        cs_onset = output['cs_onset']
        cs_offset = output['cs_offset']
        cluster_ID = output['cluster_ID']
        embedding = output['embedding']
        
        # sort clusters by cluster size
        clusters = np.unique(cluster_ID)
        n_clusters = len(clusters)
        cluster_size = np.zeros(n_clusters)
        cluster_ID_sorted = np.zeros_like(cluster_ID)
        for i in range(n_clusters):
            cluster_size[i] =  sum(cluster_ID == clusters[i])
        print(clusters, n_clusters, cluster_size)
        clusters_sorted_idx = np.argsort(cluster_size)[::-1]
        cluster_size_sorted = np.sort(cluster_size)[::-1]
        for i in range(n_clusters):
            cluster_ID_sorted[cluster_ID==clusters[clusters_sorted_idx[i]]] = i+1
        clusters_sorted = np.sort(np.unique(cluster_ID_sorted))
        print(np.unique(cluster_ID_sorted), clusters, clusters_sorted)
        
        self.CS_onset = cs_onset
        self.CS_offset = cs_offset
        self.cluster_ID = cluster_ID_sorted
        self.embedding = embedding
        self.clusters = clusters_sorted
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size_sorted

        self.savedetectFileDialog()

    def savedetectFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save detected data", 'output.mat',
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        if fileName:
            sp.savemat(fileName, {'CS_onset': self.CS_onset,
                                  'CS_offset': self.CS_offset,
                                  'cluster_ID': self.cluster_ID,
                                  'embedding': self.embedding}, do_compression=True)
    # FUNCTIONS THIRD TAB
    
    def align_spikes(self, spikes, alignment, l1=300, l2=300):
        N = len(alignment)
        spikes_aligned = np.zeros([N, l1+l2+1]) * np.nan
        for i in range(N):
            if alignment[i] - l1 < 0:
                frag = np.hstack([np.zeros([l1-alignment[i]+1])*np.nan, spikes[0:alignment[i]+l2]])
            elif alignment[i]+l2+1 > spikes.shape[0]:
                frag = np.hstack([spikes[alignment[i]-l1:-1], np.zeros(alignment[i]+l2+2-self.ss_train.shape[0])*np.nan])
            else:
                frag = spikes[alignment[i]-l1:alignment[i]+l2+1]
            spikes_aligned[i, :] = frag
        
        return spikes_aligned       
    
    def create_show_data_box(self):
        show_data_layout = QGridLayout()

        plotting_button = QPushButton('Plot data')
        plotting_button.clicked.connect(self.plot_detected_data)

        saving_button = QPushButton('Save selected cluster data')
        saving_button.clicked.connect(self.save_selected_cluster)

        create_cluster_selection_button = QPushButton('Select CS clusters')

        select_widget = QWidget()
        self.checkbox_widget = QWidget()
        self.checkbox_widget.setStyleSheet('border: 1px solid black;')

        layout = QVBoxLayout()
        layout.addWidget(create_cluster_selection_button)
        layout.addWidget(self.checkbox_widget)
        layout.addStretch()
        select_widget.setLayout(layout)
        show_data_layout.addWidget(plotting_button, 0, 0)
        show_data_layout.addWidget(select_widget, 1, 0)
        show_data_layout.addWidget(saving_button, 2, 0)

        self.select_show_data_box.setLayout(show_data_layout)

        create_cluster_selection_button.clicked.connect(self.generate_cluster_list)

    def add_checkbox(self):
        
        self.checkbox_widget.setLayout(self.checkbox_layout)

    def generate_cluster_list(self):
        self.is_cluster_selected = []
        self.checkbutton = []
        self.checkbox_layout = QVBoxLayout()
        self.add_checkbox()
        print(self.n_clusters)
        if self.n_clusters:
            for i in range(self.n_clusters):
                checkbox = QCheckBox("Cluster {} ({})".format(i+1, self.cluster_size[i].astype(int)))
                checkbox.setStyleSheet('color: C0')
                self.checkbutton.append(checkbox)
                self.is_cluster_selected.append(True)
                self.checkbutton[i].setCheckState(self.is_cluster_selected[i])
                self.checkbutton[i].setTristate(False)
                self.checkbox_layout.addWidget(self.checkbutton[i])
                self.checkbutton[i].toggled.connect(self.checkbutton_clicked)

    def checkbutton_clicked(self):
        self.set_cluster_selected()
        print(self.is_cluster_selected)

    def set_cluster_selected(self):
        n = len(self.checkbutton)
        for i in range(n):
            self.is_cluster_selected[i] = self.checkbutton[i].isChecked()


    def create_cluster_plotting_box(self):
        cluster_plotting_layout = QGridLayout()

        toolbar = NavigationToolbar(self.canvas2, self)

        cluster_plotting_layout.addWidget(toolbar, 0, 0)
        cluster_plotting_layout.addWidget(self.canvas2, 1, 0)

        self.cluster_plotting_box.setLayout(cluster_plotting_layout)


    def plot_detected_data(self):

        cluster_ID = self.cluster_ID
        cs_offset = self.CS_offset
        cs_onset = self.CS_onset
        embedding = self.embedding
        n_clusters = self.n_clusters

        time = 1000

        #TODO: Different colors for different clusters, coorect plotting
        self.canvas2.CS.cla()
        self.canvas2.LFP.cla()
        self.canvas2.clusters.cla()
        # self.canvas2.simple_spikes.cla()
        # self.canvas2.onset.cla()
        # self.canvas2.clusters.cla()
        # self.canvas2.onset_lfp.cla()
        # self.canvas2.clusters.plot(cs_offset, cs_onset, 'tab:blue', lw=0.4)
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.clusters.scatter(embedding[idx,0], embedding[idx,1],  c=self.colors[i], linewidth=None)
        self.canvas2.clusters.set_xlabel('Dimension 1')
        self.canvas2.clusters.set_ylabel('Dimension 2')
        self.canvas2.clusters.set_title('CS clusters')
        
        t1 = 5
        t2 = 20
        t = np.arange(-t1, t2, 1000/(self.sampling_rate+1))
        # plot CS
        cs_aligned = self.align_spikes(self.detect_HIGH, self.CS_onset, l1=t1*int(self.sampling_rate/1000), l2=t2* int(self.sampling_rate/1000))
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.CS.plot(t, cs_aligned[idx, :].T, c=self.colors[i], lw=0.4)
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.CS.plot(t, cs_aligned.mean(0), c=self.colors[i], lw=2)
        self.canvas2.CS.set_xlabel('Time from CS onset [ms]')
        self.canvas2.CS.set_title('CS')
        
        # plot LFP
        lfp_aligned = self.align_spikes(self.detect_LFP, self.CS_onset, l1=t1*int(self.sampling_rate/1000), l2=t2* int(self.sampling_rate/1000))
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.LFP.plot(t, lfp_aligned[idx, :].T, c=self.colors[i], lw=0.4)
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.LFP.plot(t, lfp_aligned.mean(0), c=self.colors[i], lw=2)
        self.canvas2.LFP.set_xlabel('Time from CS onset [ms]')
        self.canvas2.LFP.set_title('LFP')
            
        # plot SS
        if self.SS_varname:
            self.ss_train = get_field_mat(self.mat,[self.SS_varname])
            clusters = np.unique(self.cluster_ID)
            cs_onset_downsample = (self.CS_onset/self.sampling_rate*1000).astype(int)
            ss_aligned = self.align_spikes(self.ss_train, cs_onset_downsample, 50, 50)
            offset = 0
            for i in range(self.n_clusters):
                # [iy, ix] = np.where(ss_aligned==1)
                [iy, ix] = np.where(ss_aligned[cluster_ID==clusters[i], :]==True)   
                self.canvas2.SS.plot(t[ix], iy+offset, '.', c=self.colors[i])
                offset = offset + (cluster_ID==i+1).sum()
            self.canvas2.SS.set_xlabel('Time from CS onset [ms]')
            self.canvas2.SS.set_title('SS')
            
        # self.canvas2.onset.plot(time, embedding, 'tab:blue', lw=0.4)
        # self.canvas2.onset.set_xlabel('CS onset')
        # self.canvas2.simple_spikes.plot()
        # self.canvas2.simple_spikes.set_xlabel('Simple Spikes')
        self.canvas2.draw()

    def save_selected_cluster(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save selected cluster data", 'clusters.mat',
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        if fileName:
            self.get_selected_clusters()
            sp.savemat(fileName, {'CS_onset': self.CS_onset,
                                  'CS_offset': self.CS_offset,
                                  'cluster_ID': self.cluster_ID,
                                  'embedding': self.embedding}, do_compression=True)

    def get_selected_clusters(self):
        newarray = []
        selected_indices = np.where(np.array(self.is_cluster_selected)==True)
        for i in self.cluster_ID:
            if self.cluster_ID[i] == selected_indices:
                newarray.append(self.cluster_ID[i])


def create():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    main = Frame()
    main.show()
    sys.exit(app.exec_())


create()
