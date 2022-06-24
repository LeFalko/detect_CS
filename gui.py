
# from typing import List, Any

from PyQt5.QtWidgets import (QApplication, QComboBox, QDesktopWidget, QDialog, QFileDialog, QSizePolicy, QFormLayout, QGridLayout, QGroupBox, QSpinBox, 
                             QHBoxLayout, QVBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QToolButton, QTabWidget,
                             QTextEdit, QWidget, QListWidget, QCheckBox, QLineEdit, QScrollBar, QStyle, QShortcut)
from PyQt5.QtGui import QIcon, QDesktopServices, QPixmap, QColor, QImage, QKeySequence
from PyQt5.QtCore import QUrl, QSize, Qt, QRect, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mplcolors
import scipy.io as sp
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import numpy as np
import sys
import os
from CS import detect_CS, norm_LFP, norm_high_pass, get_field_mat, create_random_intervals, concatenate_segments
import uneye

# add byte images 
import io
import base64 
from PIL import Image, ImageQt
from pic2str import logo

# this creates a .py file containing a variable of byte data from an image.
# import base64
# def pic2str(file, functionName):
#     pic = open(file, 'rb')
#     content = '{} = {}\n'.format(functionName, base64.b64encode(pic.read()))
#     pic.close()

#     with open('pic2str.py', 'a') as f:
#         f.write(content)
        
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.high_axes = fig.add_subplot(211)
        self.high_axes.get_xaxis().set_visible(False)
        self.high_axes.set_title('Action potential')
        self.lfp_axes = fig.add_subplot(212, sharex=self.high_axes, sharey=self.high_axes)
        self.lfp_axes.get_xaxis().set_visible(True)
        self.lfp_axes.set_xlabel('Time')
        self.lfp_axes.set_title('LFP')

        super(MplCanvas, self).__init__(fig)
        
class MplCanvas2(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig2 = Figure(figsize=(width, height), dpi=dpi)
        self.CS = fig2.add_subplot(221)
        self.CS.set_title('CS', loc='left')
        self.LFP = fig2.add_subplot(223)
        self.LFP.set_title('LFP', loc='left')
        self.CS_clusters = fig2.add_subplot(222)
        self.CS_clusters.set_title('Feature space', loc='left')
        self.SS = fig2.add_subplot(224)
        self.SS.set_title('SS', loc='left')
        self.ax2 = self.SS.twinx()

        super(MplCanvas2, self).__init__(fig2)


# Initializing GUi window and setting size
class Frame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 500
        self.top = 200
        self.width = 1600
        self.height = 1000
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.title = 'EPICS'
        self.setWindowTitle(self.title)

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        qtRectangle.moveTop(10)
        self.move(qtRectangle.topLeft())

        self.table_widget = Content(self)
        self.setCentralWidget(self.table_widget)


# creating content for the Frame
class Content(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QGridLayout()
        "self.popup_window = QWidget()"

        # Initialize arrays and helper values
        self.LFP_varname = 'LFP'
        self.HIGH_varname = 'HIGH'
        self.SS_varname = 'SS_train'
        self.Label_varname = 'Labels'
        self.LFP = []
        self.HIGH = []
        self.Labels = []
        self.Intervals_inspected = []
        self.upload_LFP = []
        self.upload_HIGH = []
        self.upload_Label = []
        self.label = []
        self.interval_inspected = []
        self.upload_fileName = []
        self.fileNames = []
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
        self.detect_folder = "No folder selected"
        self.output_folder = "No folder selected"
        self.output_suffix = '_output'
        self.logName = 'log'
        self.outputName = "No output"
        self.detect_fileName = "No file"
        self.CS_onset = []
        self.CS_offset = []
        self.cluster_ID = []
        self.cluster_ID_save = []
        self.embedding = []
        self.n_clusters = []
        self.ss_train = []
        self.ss_sort = 'cluster'
        self.sigma = 5
        self.t1 = 5
        self.t2 = 20
        self.t1_ss = 50
        self.t2_ss = 50
        self.ms1 = 3
        self.ms2 = 3
        # self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
        #                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Selecting CS span
        self.is_clicked = False
        self.cs_span = np.empty(2)
        self.cs_spans = np.array([])
        self.cs_spans_all = []
        self.cs_patch = []
        self.cs_patch2 = []

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab_label_data = QWidget()
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
        self.tabs.addTab(self.tab_label_data, "Label data")
        self.tabs.addTab(self.tab_detect, "Detect CS")
        self.tabs.addTab(self.tab_postprocessing, "Post-processing")

        # Create first tab
        self.tab_label_data.layout = QGridLayout(self)

        # Groupboxes
        self.data_input_box = QGroupBox("Data input")
        self.save_label_box = QGroupBox("Save labels of current recording")
        self.save_box = QGroupBox("Save training data")
        self.loaded_files_box = QGroupBox("Loaded files")
        self.select_cs_box = QGroupBox("Select CSs")
        self.after_labeling_box = QGroupBox("After labeling")

        # List of loaded files
        self.loaded_file_listWidget = QListWidget() 
        self.loaded_file_listWidget.setFixedHeight(self.loaded_files_box.height()-70)
        self.loaded_file_listWidget.update()
        
        # Add groupboxes to first tab
        self.create_data_input_box()
        # self.create_information_box()
        self.create_save_label_box()
        self.create_save_box()
        self.create_loaded_files_box()
        self.create_select_cs_box()
        self.create_after_labeling_box()

        # layout for first tab
        left_panel = QWidget()
        layout = QGridLayout()
        layout.addWidget(self.data_input_box, 0, 0)
        # layout.addWidget(self.information_box, 1, 0)
        layout.addWidget(self.loaded_files_box, 2, 0)
        layout.addWidget(self.save_label_box, 3, 0)
        layout.addWidget(self.save_box, 4, 0)
        layout.addWidget(self.after_labeling_box, 5, 0)
        left_panel.setLayout(layout)
        # self.tab_preprocessing.layout.addWidget(self.data_input_box, 0, 1)
        self.tab_label_data.layout.addWidget(left_panel, 0, 0)
        self.tab_label_data.layout.addWidget(self.select_cs_box, 0, 1)
        self.tab_label_data.layout.setColumnStretch(0, 4)
        self.tab_label_data.layout.setRowStretch(0, 0)
        self.tab_label_data.layout.setRowStretch(1, 4)
        # self.tab_preprocessing.layout.setRowStretch(0, 4)
        self.tab_label_data.setLayout(self.tab_label_data.layout)
        
        # Keyboard shotcuts
        self.shortcut_Q = QShortcut(QKeySequence('Q'), self, self.set_max_xlim)
        self.shortcut_W = QShortcut(QKeySequence('W'), self, lambda: self.set_zoom_xlim(1.0))
        self.shortcut_E = QShortcut(QKeySequence('E'), self, lambda: self.set_zoom_xlim(0.05))
        self.shortcut_R = QShortcut(QKeySequence('R'), self, lambda: self.zoom(self.zoom_ratio))
        self.shortcut_T = QShortcut(QKeySequence('T'), self, lambda: self.zoom(1/self.zoom_ratio))
        self.shortcut_D = QShortcut(QKeySequence('D'), self, lambda: self.scroll.setValue(self.scroll.value() - 1))
        self.shortcut_F = QShortcut(QKeySequence('F'), self, lambda: self.scroll.setValue(self.scroll.value() + 1))
        self.shortcut_C = QShortcut(QKeySequence('C'), self, self.go_to_prev_CS)
        self.shortcut_V = QShortcut(QKeySequence('V'), self, self.go_to_next_CS)
        

        # Create second tab
        self.tab_detect.layout = QGridLayout(self)

        # Groupboxes
        self.detect_cs_box = QGroupBox("Detect CS")

        # Add Groupboxes to second tab
        self.create_detect_cs_box()

        # layout for second tab
        self.tab_detect.layout.addWidget(self.detect_cs_box, 0, 0)
        self.tab_detect.layout.setColumnStretch(0, 1)

        self.tab_detect.setLayout(self.tab_detect.layout)

        # Create third tab
        self.tab_postprocessing.layout = QGridLayout(self)

        # Groupboxes
        # self.select_show_data_box = QGroupBox("Select clusters to show")
        self.load_files_for_plot_box = QGroupBox("Load files for plot")
        self.select_show_data_box = QWidget()
        self.cluster_plotting_box = QGroupBox("Plotting")

        # Add Groupboxes to third tab
        self.create_load_files_for_plot_box()
        self.create_show_data_box()
        self.create_cluster_plotting_box()

        # Layout for third tab
        self.tab_postprocessing.layout.addWidget(self.load_files_for_plot_box, 0, 0, 1, 2)
        self.tab_postprocessing.layout.addWidget(self.select_show_data_box, 1, 0)
        self.tab_postprocessing.layout.addWidget(self.cluster_plotting_box, 1, 1)

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
        
        widget2 = QWidget()
        layout2 = QGridLayout()

        labeling_button = QPushButton('Select CS')
        labeling_button.clicked.connect(self.select_cs)

        delete_button = QPushButton('Delete last Selection')
        delete_button.clicked.connect(self.delete_last_CS)

        next_cell_button = QPushButton('Proceed to next cell')
        next_cell_button.clicked.connect(self.goto_next_cell)

        save_button = QPushButton('Save your selected CS')
        save_button.clicked.connect(self.saveFileDialog)
        
        layout2.addWidget(labeling_button, 0,0)
        layout2.addWidget(delete_button, 0,1)
        layout2.addWidget(next_cell_button, 1,0)
        layout2.addWidget(save_button, 1,1)
        widget2.setLayout(layout2)
        
        minWidth = 150
        
        max_button = QPushButton("Full (Q)")
        max_button.clicked.connect(self.set_max_xlim)
        max_button.setMinimumWidth(minWidth)
        second_button = QPushButton("1s (W)")
        second_button.clicked.connect(lambda: self.set_zoom_xlim(1.0))
        second_button.setMinimumWidth(minWidth)
        millisecond_button = QPushButton("50ms (E)")
        millisecond_button.clicked.connect(lambda: self.set_zoom_xlim(0.05))
        millisecond_button.setMinimumWidth(minWidth)
        self.zoom_ratio = 2/3
        zoomin_button = QPushButton("Zoom in (R)")
        zoomin_button.clicked.connect(lambda: self.zoom(self.zoom_ratio))
        zoomin_button.setMinimumWidth(minWidth)
        zoomout_button = QPushButton("Zoom out (T)")
        zoomout_button.clicked.connect(lambda: self.zoom(1/self.zoom_ratio))
        zoomout_button.setMinimumWidth(minWidth)
        prev_cs_button = QPushButton("Previous CS (C)")
        prev_cs_button.clicked.connect(self.go_to_prev_CS)
        prev_cs_button.setMinimumWidth(minWidth)
        next_cs_button = QPushButton("Next CS (V)")
        next_cs_button.clicked.connect(self.go_to_next_CS)
        next_cs_button.setMinimumWidth(minWidth)
        self.cs_counter = QLabel()
        self.cs_counter.setText('{} CSs selected'.format(self.cs_spans.T.shape[0]))
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(30, 30))
        info_label.setToolTip(self.info_select_CS())
        
        scale_widget = QWidget()
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(info_label)
        scale_layout.addStretch()
        scale_layout.addWidget(max_button)
        scale_layout.addWidget(second_button)
        scale_layout.addWidget(millisecond_button)
        scale_layout.addWidget(zoomin_button)
        scale_layout.addWidget(zoomout_button)
        scale_layout.setContentsMargins(0,0,0,0)
        scale_widget.setLayout(scale_layout)
        
        cs_widget = QWidget()
        cs_layout = QHBoxLayout()
        cs_layout.addStretch()
        cs_layout.addWidget(self.cs_counter)
        cs_layout.addWidget(prev_cs_button)
        cs_layout.addWidget(next_cs_button)
        cs_layout.setContentsMargins(0,0,0,0)
        cs_widget.setLayout(cs_layout)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(scale_widget)
        ctrl_layout.addWidget(cs_widget)
        ctrl_layout.setContentsMargins(0,0,0,0)
        
        ctrl = QWidget()
        ctrl.setLayout(ctrl_layout)
        # ctrl.setStyleSheet('border: 1px solid red;')

        select_cs_layout.addWidget(ctrl, 0, 0)
        select_cs_layout.addWidget(self.canvas, 1, 0)
        self.scroll = QScrollBar(Qt.Horizontal)
        self.step = 20
        # self.setupSlider(0, 0, 0)
        select_cs_layout.addWidget(self.scroll, 3, 0)
        self.canvas_ylim = (0, 1)
        self.select_cs_box.setLayout(select_cs_layout)

    # creating a box in the first tab containing sampling rate input and file upload
    def create_data_input_box(self):
        data_input_layout = QHBoxLayout()
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(30)
        info_label.setToolTip(self.info_data_input())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)
        # info_label.setStyleSheet('border: 1px solid blue;')
        
        upload_button = QPushButton("Add PC for manual labeling")
        upload_button.setToolTip('Upload and plot a file for labeling')
        upload_button.clicked.connect(self.openFileNameDialog)
        
        setting_button = QPushButton("Set parameters")
        setting_button.clicked.connect(self.open_setting_box)
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(setting_button)
        layout.addWidget(upload_button)
        layout.setContentsMargins(0,0,10,10)
        widget.setLayout(layout)
        data_input_layout.addWidget(info_label)
        data_input_layout.addWidget(widget)
        data_input_layout.setSpacing(0)
        data_input_layout.setContentsMargins(0,10,0,10)
        

        self.data_input_box.setLayout(data_input_layout)
        # self.data_input_box.setStyleSheet('border: 1px solid blue;')
        
    def create_save_label_box(self):
        save_label_layout = QVBoxLayout()
        
        save_label_button = QPushButton('Save CS labels')
        save_label_button.clicked.connect(self.saveCurrentFile)
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(30)
        info_label.setToolTip(self.info_save_label())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)
        
        save_label_layout = QHBoxLayout()
        save_label_layout.addWidget(info_label)
        save_label_layout.addWidget(save_label_button)
        save_label_layout.setSpacing(0)
        save_label_layout.setContentsMargins(0,10,10,10)
        
        self.save_label_box.setLayout(save_label_layout)

    def create_save_box(self):
        # save_layout = QVBoxLayout()
        
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.saveFileDialog)
        # save_layout.addWidget(save_button)
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(30)
        info_label.setToolTip(self.info_save_data())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)
        
        layout = QHBoxLayout()
        layout.addWidget(info_label)
        layout.addWidget(save_button)
        layout.setSpacing(0)
        layout.setContentsMargins(0,10,10,10)
        
        self.save_box.setLayout(layout)

    # creating the box in the first tab containing user information
    def open_setting_box(self):
        dialog = QDialog()
        dialog.setWindowTitle("Setting")
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.close)
        layout = QFormLayout()
        layout.addRow(QLabel("Sampling rate [Hz]"), self.set_samplingRate())
        layout.addRow(QLabel("Action potential variable name"), self.set_HighVarname())
        layout.addRow(QLabel("LFP variable name"), self.set_LFPVarname())
        layout.addRow(QLabel("CS label variable name"), self.set_LabelVarname())
        # layout.addRow(QLabel("Max. CSs to select"), self.set_maxCSs())
        layout.addRow(ok_button)
        dialog.setLayout(layout)
        dialog.exec_()
        dialog.show()
    
    def create_information_box(self):
        information_layout = QVBoxLayout()

        textedit = QTextEdit()
        # textedit.resize(300, 200)
        textedit.setPlainText("Separate files individual PCs (unfiltered raw data) \n"
                              "Ask for cut-off frequencies: upper cut off and lower cut off, "
                              "sampling rate or use default values (use from our paper)")

        information_layout.addWidget(textedit)

        self.information_box.setLayout(information_layout)
        
    def create_loaded_files_box(self):
        layout = QVBoxLayout()
        self.loaded_file_listWidget.clear()
        self.loaded_file_listWidget.addItems(self.ID)
        self.loaded_file_listWidget.setCurrentRow(len(self.ID)-1)
        # print('loaded_file_listWidget.currentRow()', self.loaded_file_listWidget.currentRow())
                
        plot_file_button = QPushButton("Plot")
        plot_file_button.clicked.connect(self.set_current_file)        
                
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_loaded_file)
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(30)
        info_label.setToolTip(self.info_loaded_files())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(info_label)
        button_layout.addWidget(plot_file_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(remove_button)
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0,0,0,0)
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        button_widget.setFixedHeight(50)
        
        layout.addWidget(button_widget)
        # height = self.loaded_files_box.height()-button_widget.height()-60
        # self.loaded_file_listWidget.setFixedHeight(height)
        layout.addWidget(self.loaded_file_listWidget)
        layout.addStretch()
        # button_widget.sizeHint()
        
        self.loaded_files_box.setLayout(layout)
        self.loaded_files_box.update()
        self.set_current_file()
        
    def remove_loaded_file(self):
        idx = self.loaded_file_listWidget.currentRow()
        print('index removed',idx)
        if self.LFP:
            self.loaded_file_listWidget.takeItem(idx)
            self.ID.pop(idx)
            self.LFP.pop(idx)
            self.HIGH.pop(idx)
            self.Labels.pop(idx)
            self.cs_spans_all.pop(idx)
            
            self.canvas.high_axes.cla()
            self.canvas.lfp_axes.cla()
            self.canvas.draw_idle()
            self.loaded_files_box.update()
    
    def set_current_file(self):
        idx = self.loaded_file_listWidget.currentRow()
        # print('current index: ',idx)
        if self.LFP:
            self.upload_LFP = self.LFP[idx]
            self.upload_HIGH = self.HIGH[idx]
            self.label = self.Labels[idx]
            self.upload_fileName = self.ID[idx]
            self.cs_spans = self.cs_spans_all[idx]
            print('cell: ',self.ID[idx])
            print('cs_spans: ',self.cs_spans)
            self.plot_data()
            self.cs_counter.setText('{} CSs selected'.format(self.cs_spans.T.shape[0]))
    
    def create_after_labeling_box(self):
        after_labeling_layout = QHBoxLayout()
        
        goto_Colab_button = QPushButton("TRAIN ALGORITHM")
        goto_Colab_button.setToolTip('Please finish labeling data before going to Colab')
        goto_Colab_button.clicked.connect(self.open_Colab)

        # Load byte data
        byte_data = base64.b64decode(logo)
        image_data = io.BytesIO(byte_data)
        image = Image.open(image_data)

        # PIL to QPixmap
        qImage = ImageQt.ImageQt(image)
        image = QPixmap.fromImage(qImage)
        goto_Colab_button.setIcon(QIcon((image)))
        # goto_Colab_button.setIcon(QIcon(('./img/colab_logo.png')))
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(30)
        info_label.setToolTip(self.info_after_labeling())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)
        
        after_labeling_layout.addWidget(info_label)
        after_labeling_layout.addWidget(goto_Colab_button)
        after_labeling_layout.setSpacing(0)
        after_labeling_layout.setContentsMargins(0,10,10,10)
        
        self.after_labeling_box.setLayout(after_labeling_layout)
    
    # creating input for setting parameters
    def set_samplingRate(self):
        def changeSamplingRate():
            self.sampling_rate = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1000, 100000)
        spinbox.setValue(self.sampling_rate)
        spinbox.valueChanged.connect(changeSamplingRate)
        return spinbox
    
    def set_HighVarname(self):
        def changeText():
            self.HIGH_varname = lineedit.text()
        lineedit = QLineEdit(self.HIGH_varname)
        lineedit.textChanged.connect(changeText)
        return lineedit
    
    def set_LFPVarname(self):
        def changeText():
            self.LFP_varname = lineedit.text()
        lineedit = QLineEdit(self.LFP_varname)
        lineedit.textChanged.connect(changeText)
        return lineedit
    
    def set_LabelVarname(self):
        def changeText():
            self.Label_varname = lineedit.text()
        lineedit = QLineEdit(self.Label_varname)
        lineedit.textChanged.connect(changeText)
        return lineedit

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
        if not self.LFP_varname in mat.keys():
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Variable [" + self.LFP_varname +"] not found.")
            errorbox.exec_()
        elif not self.HIGH_varname in mat.keys():
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Variable [" + self.HIGH_varname +"] not found.")
            errorbox.exec_() 
        else:
            self.upload_fileName = fileName.split('.')[-2].split('/')[-1]
            self.upload_LFP = np.array(mat[self.LFP_varname][0])
            self.upload_HIGH = np.array(mat[self.HIGH_varname][0])
            if self.Label_varname in mat.keys():
                self.label = np.array(mat[self.Label_varname][0])
                print('Label is loaded')  
                print(self.Label_varname)                 
                self.labels_to_spans()
            else:
                print('CS labels not available')
                self.cs_spans = np.array([[]])
            self.interval_inspected = np.zeros_like(self.upload_LFP)
            self.cs_span = np.zeros(2)
            
            self.cs_patch = []
            self.cs_patch2 = []
            self.ID.append(self.upload_fileName)
            self.LFP.append(self.upload_LFP)
            self.HIGH.append(self.upload_HIGH)
            self.Labels.append(self.label)
            self.cs_spans_all.append(self.cs_spans)
            self.Intervals_inspected.append(self.interval_inspected)
            self.plot_data()
            self.create_loaded_files_box()
            
    def saveCurrentFile(self):
        if self.ID:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self, "Save file", self.upload_fileName+".mat",
                                                      "All Files (*);;MATLAB Files (*.mat)", options=options)
            if fileName:
                LFP = self.upload_LFP
                HIGH = self.upload_HIGH
                Label = self.label
                
                sp.savemat(fileName, {self.LFP_varname: LFP,
                                      self.HIGH_varname: HIGH,
                                      self.Label_varname: Label,}, do_compression=True)
        else:
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Warning")
            errorbox.setText("Upload a PC recording")
            errorbox.exec_()
            

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save file", "train_data.mat",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        if fileName:
            bigLFP = []
            bigHIGH = []
            bigLabels = []
            for i in range(len(self.ID)):
                print(i)
                print(self.LFP[i])
                print(self.Intervals_inspected[i])
                lfp_norm = norm_LFP(self.LFP[i],self.sampling_rate)
                high_norm = norm_high_pass(self.HIGH[i])
                compLFP,compHIGH,compLabels = concatenate_segments(lfp_norm, high_norm, self.Intervals_inspected[i], self.Labels[i])
                bigLFP = np.concatenate((bigLFP,compLFP))
                bigHIGH = np.concatenate((bigHIGH,compHIGH))
                bigLabels = np.concatenate((bigLabels,compLabels))
            # sp.savemat(fileName, {'ID': np.array(self.ID, dtype=object),
            #                       'LFP': self.LFP,
            #                       'HIGH': self.HIGH,
            #                       'Labels': self.Labels,
            #                       'sampling_rate': self.sampling_rate}, do_compression=True)
            sp.savemat(fileName, {'ID': np.array(self.ID, dtype=object),
                                  'LFP': bigLFP,
                                  'HIGH': bigHIGH,
                                  'Labels': bigLabels,
                                  'sampling_rate': self.sampling_rate}, do_compression=True)


    def open_Colab(self):
        # QDesktopServices.openUrl(QUrl('https://colab.research.google.com/drive/1g1MzZz5h30Uov9tIbrarwwm02WD7xU6B#scrollTo=plKVE-vH_SLt'))
        # QDesktopServices.openUrl(QUrl('https://colab.research.google.com/drive/1WenM8VYNQSxknWoSlv7wqimavASXvo50?authuser=5#scrollTo=wZ0-3PDAz5qr'))
        QDesktopServices.openUrl(QUrl('https://colab.research.google.com/drive/1WenM8VYNQSxknWoSlv7wqimavASXvo50'))
    # updating plot for raw data
    def plot_data(self):
        raw_data = self.upload_LFP
        high_data = self.upload_HIGH
        #print(raw_data, high_data)

        self.t = np.linspace(0, len(self.upload_HIGH)/self.sampling_rate, len(self.upload_HIGH))

        self.canvas.high_axes.cla()
        self.canvas.lfp_axes.cla()
        self.canvas.high_axes.plot(self.t, high_data, 'tab:blue', lw=0.4)
        print('cs_spans_all: ', self.cs_spans_all)
        for i in range(self.cs_spans.shape[1]):
            ylim = self.canvas.high_axes.get_ylim()
            
            # print('cs_spans,i', self.cs_spans,self.cs_spans.shape,i)
            print('cs_spans[i]',self.cs_spans[:, i])
            patch = patches.Rectangle((self.t[self.cs_spans[0, i]], ylim[0]), np.diff(self.t[self.cs_spans[:,i]]), np.diff(ylim), linewidth=1, edgecolor='k', facecolor='r', alpha=0.2, zorder=2)
            patch2 = patches.Rectangle((self.t[self.cs_spans[0, i]], ylim[0]), np.diff(self.t[self.cs_spans[:,i]]), np.diff(ylim), linewidth=1, edgecolor='k', facecolor='r', alpha=0.2, zorder=2)
            self.canvas.high_axes.add_patch(patch)
            self.canvas.lfp_axes.add_patch(patch2)
        self.canvas.high_axes.set_xlim([0, self.t[-1]])
        self.canvas.high_axes.set_ylabel('Action potential')
        self.canvas.lfp_axes.plot(self.t, raw_data, 'tab:blue', lw=0.4)
        self.canvas.lfp_axes.set_xlim([0, self.t[-1]])
        self.canvas.lfp_axes.set_ylabel('LFP')
        self.canvas.lfp_axes.set_xlabel('time [s]')
        self.canvas.high_axes.set_title(self.upload_fileName)
        self.canvas.draw()
        self.canvas_ylim = self.canvas.lfp_axes.get_ylim()
    #     self.canvas.mpl_connect('button_release_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.click_control)
        self.canvas.mpl_connect('motion_notify_event', self.draw_span)
        self.canvas.mpl_connect('button_release_event', self.set_cs_offset)
    
    # Selecting CSs 
    def click_control(self, event):
        if self.canvas.high_axes.patches or self.canvas.lfp_axes.patches:
            # for i in range(self.cs_spans.T.shape[0]):
            for i in range(len(self.canvas.high_axes.patches)):
                # if (event.xdata >=self.t[self.cs_spans[0, i]]) and (event.xdata<=self.t[self.cs_spans[1, i]]):
                x1 = self.canvas.high_axes.patches[i].get_x()
                x2 = self.canvas.high_axes.patches[i].get_x() + self.canvas.high_axes.patches[i].get_width()[0]
                print('x1,x2,event.xdata',x1,x2,event.xdata)
                x_max = max([x1, x2])
                x_min = min([x1, x2])
                if (event.xdata >= x_min) and (event.xdata <= x_max):  
                    self.canvas.high_axes.patches.pop(i)
                    self.canvas.lfp_axes.patches.pop(i)
                    self.canvas.draw_idle()
                    self.get_all_patches()
                    return print('remove')
        self.set_cs_onset(event)

    def set_cs_onset(self, event):
        self.is_clicked = True
        self.cs_span[0] = event.xdata
        self.cs_span[1] = event.xdata
        ylim = self.canvas.high_axes.get_ylim()
        self.cs_patch = patches.Rectangle((self.cs_span[0], ylim[0]), np.diff(self.cs_span), np.diff(ylim), linewidth=1, edgecolor='k', facecolor='r', alpha=0.2, zorder=2)
        self.cs_patch2 = patches.Rectangle((self.cs_span[0], ylim[0]), np.diff(self.cs_span), np.diff(ylim), linewidth=1, edgecolor='k', facecolor='r', alpha=0.2, zorder=2)
        self.canvas.high_axes.add_patch(self.cs_patch)
        self.canvas.lfp_axes.add_patch(self.cs_patch2)
        
    def draw_span(self, event):
        if self.is_clicked:
            self.cs_span[1] = event.xdata
            self.cs_patch.set_width(np.diff(self.cs_span))
            self.cs_patch2.set_width(np.diff(self.cs_span))
            self.canvas.draw_idle()
        
    def set_cs_offset(self, event):
        if self.is_clicked:
            self.cs_span[1] = event.xdata
            self.cs_patch.set_width(np.diff(self.cs_span))
            self.cs_patch2.set_width(np.diff(self.cs_span))
            
            if abs(np.diff(self.cs_span))<0.001: # if span is too short, it doesn't count
                # print('diff',self.cs_span[0],self.cs_span[1],event.xdata,abs(np.diff(self.cs_span)))
                self.canvas.high_axes.patches.pop()
                self.canvas.lfp_axes.patches.pop()
            self.canvas.draw_idle()
            self.is_clicked = False
            self.cs_span = np.empty(2)
            self.cs_patch = []
            self.cs_patch2 = []
            self.get_all_patches()
        
    def get_all_patches(self):
        onset = np.array([])
        offset = np.array([])
        if self.canvas.high_axes.patches:
            for patch in self.canvas.high_axes.patches:
                x1 = np.absolute(self.t-patch.get_x()).argmin().astype(int)
                x2 = np.absolute(self.t-patch.get_x()-patch.get_width()).argmin().astype(int)
                onset = np.append(onset, np.min([x1, x2]))
                offset = np.append(offset, np.max([x1, x2]))
            cs_spans = np.vstack((onset, offset))
            
            idx = np.argsort(onset)
            self.cs_spans = cs_spans[:, idx].astype(int)
            print('cs_spans:',self.cs_spans, self.cs_spans.shape, np.argsort(onset))
            self.create_label()
            self.cs_counter.setText('{} CSs selected'.format(self.cs_spans.T.shape[0]))
            
    # Zoom and slider
    def setupSlider(self, minimum, maximum, step, x0=0):
        # print('setupSlider', self.lims)
        self.scroll.setMinimum(minimum)
        self.scroll.setMaximum(maximum)
        self.scroll.setPageStep(step)
        self.scroll.setValue(x0)
        try:
            self.scroll.valueChanged.disconnect()
        except:
            pass
        self.scroll.valueChanged.connect(self.zoom_update)
        
    def zoom_update(self, evt=None):
        l1 = self.lims[0] + self.scroll.value() * np.diff(self.lims) / self.step 
        l2 = l1 +  np.diff(self.lims)
        self.canvas.lfp_axes.set_xlim(l1,l2)
        self.canvas.lfp_axes.set_ylim(self.canvas_ylim)
        # print('update',self.scroll.value(), l1,l2)
        self.canvas.draw_idle()
        
    def set_max_xlim(self):
        if len(self.upload_HIGH)>0:
            l = len(self.upload_HIGH)-1
            self.canvas.high_axes.set_xlim(self.t[0], self.t[l])
            self.canvas.lfp_axes.set_xlim(self.t[0], self.t[l])
            self.lims = np.array([0, len(self.upload_LFP)/self.sampling_rate])
            self.setupSlider(0, 0, 0)
            self.canvas.draw_idle()
        else:
            print('no plot data')
            self.canvas.high_axes.set_xlim(0, 1)
            self.canvas.lfp_axes.set_xlim(0, 1)
            self.lims = np.array([0, 1])
            self.setupSlider(0, 0, 0)
            self.canvas.draw_idle()
        
    def set_zoom_xlim(self, width):
        maximum = np.floor(len(self.upload_LFP) / self.sampling_rate * self.step / width).astype(int)
        minimum = 0
        lims = self.canvas.lfp_axes.get_xlim()
        xlim = [lims[0], lims[0]+width]
        x0 = np.floor(xlim[0] * self.step / width).astype(int)
        # print(xlim[0], x0)
        self.scroll.setValue(x0)
        self.lims = np.array([0, width])
        self.setupSlider(minimum, maximum, self.step, x0)
        self.zoom_update()
        self.canvas.draw_idle()
        
    def zoom(self, z):
        lims = self.canvas.lfp_axes.get_xlim()
        center = np.mean(lims)
        w = z * (lims[1] - center)
        width = z * np.diff(lims)[0]
        maximum = np.floor(len(self.upload_LFP) / self.sampling_rate * self.step / width).astype(int)
        minimum = 0
        lims_new = np.round([center-w, center+w], 5)
        xlim = [lims_new[0], lims_new[0]+width]
        x0 = np.floor(xlim[0] * self.step / width).astype(int)
        print('zoom', xlim[0], x0, width)
        self.scroll.setValue(x0)
        self.lims = np.array([0, width])
        self.zoom_update()
        self.setupSlider(minimum, maximum, self.step, x0)
        # self.zoom_update()
        self.canvas.draw_idle()
        
    def go_to_prev_CS(self):
        if self.cs_spans.T.shape[0]>0:
            width = 0.05
            center = np.mean(self.canvas.high_axes.get_xlim())
            idx_center = np.absolute(self.t-center).argmin().astype(int)
            # print('center',center, idx_center, self.cs_spans[0,:])
        
            idx0 = np.where(self.cs_spans[0,:]<idx_center)[0]
            if len(idx0)>0:
                idx = idx0.max()
                self.lims = (self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2)
                print('prev cs',self.lims, idx)
                self.canvas.lfp_axes.set_xlim([self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2])
                self.canvas.draw_idle()
                # self.set_zoom_xlim(width)
        else:
            print('No more previouse CSs')
                
    def go_to_next_CS(self):
        if self.cs_spans.T.shape[0]>0:
            width = 0.05
            center = np.mean(self.canvas.high_axes.get_xlim())
            idx_center = np.absolute(self.t-center).argmin().astype(int)
            # print('center',center, idx_center, self.cs_spans[0,:])
       
            idx0 = np.where(self.cs_spans[0,:]>idx_center)[0]
            if len(idx0)>0:
                idx = idx0.min()
                self.lims = (self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2)
                print('next cs',self.lims, idx)
                self.canvas.lfp_axes.set_xlim([self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2])
                self.canvas.draw_idle()
                # self.set_zoom_xlim(width)
        else:
            print('No more next CSs')
                
    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_R:
    #         self.zoom(self.zoom_ratio)
    #     elif event.key() == Qt.Key_T:
    #         print('T')
    #         self.zoom(1/self.zoom_ratio)
    #     elif event.key()==Qt.Key_F:
    #         print('F')
    #         self.scroll.setValue(self.scroll.value() + 1)
    #     elif event.key()==Qt.Key_D:
    #         self.scroll.setValue(self.scroll.value() - 1)
    #     elif event.key() == Qt.Key_Q:
    #         self.set_max_xlim()
    #     elif event.key() == Qt.Key_W:
    #         self.set_zoom_xlim(1.0)
    #     elif event.key() == Qt.Key_E:
    #         self.set_zoom_xlim(0.05)
    #     elif event.key() == Qt.Key_C:
    #         self.go_to_prev_CS()
    #     elif event.key() == Qt.Key_V:
    #         self.go_to_next_CS()

    # activates span selection
    def select_cs(self):
        self.span = SpanSelector(self.canvas.high_axes, self.onselect, 'horizontal', useblit=True,
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
            self.create_label()
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

    # delete last CS function
    def delete_last_CS(self):
        if self.x_values[self.backwardscounter][0] == 0:
            self.backwardscounter -= 1
        else:
            self.x_values[self.backwardscounter][0] = 0
            self.x_values[self.backwardscounter][1] = 0
            self.value_counter -= 1

    def create_label(self):
        label = np.zeros_like(self.upload_LFP)
        for i in range(self.cs_spans.T.shape[0]):
            label[self.cs_spans[0,i]:self.cs_spans[1,i]] = 1
        self.label = label
        idx = self.loaded_file_listWidget.currentRow()
        self.Labels[idx] = self.label
        self.cs_spans_all[idx] = self.cs_spans
        
        self.interval_inspected = create_random_intervals(self.sampling_rate, self.upload_LFP, self.label)
        self.Intervals_inspected[idx] = self.interval_inspected
        print(self.cs_spans_all)
        # self.Labels.append(labels)
        
    def labels_to_spans(self):
        onset = np.where(np.diff(self.label.astype(int))==1)[0]
        offset = np.where(np.diff(-self.label.astype(int))==1)[0]
        print('onset',onset)
        print('offset',offset)
        print(len(onset), len(offset), len(self.label))
        if len(onset) > len(offset):
            offset[len(onset)] = len(self.label)
        self.cs_spans = np.zeros([2, len(onset)],dtype=int)
        for i in range(len(onset)):
            self.cs_spans[0, i] = onset[i]
            self.cs_spans[1, i] = offset[i]
        print('self.cs_spans',self.cs_spans)
        
    # explanation texts for the first tab
    def info_data_input(self):
        text = """1. Set initial parameters for labeling CSs. 
2. Upload your recordings (stored in .mat format).
    - high band-passed action potential (1 x time)
    - low band-passed LFP signal (1 x time)
    Note: although not recommended, in case no LFP signal is available, try using the high band-passed signal also as LFP. """
        return text
    
    def info_loaded_files(self):
        text = """Loaded files are stored here. 
    - plot/remove the selected file"""
        return text
    
    def info_save_label(self):
        text = """Save CS labels of the current recording, together with aciton potential and LFP."""
        return text
    
    def info_save_data(self):
        text = """CSs of all files are concatenated and saved in one large row vector."""
        return text
    
    def info_after_labeling(self):
        text1 = """After saving the CS labels, 
click this button to use Google Colab's resource for training the network."""
        return text1
    
    def info_select_CS(self):
        text = """- Find CSs by zooming in and drag-select the onset & offset of CSs. Selected CSs will be shaded in red. 
- To delete the selection, click the selected red area. 
- We recoomend selecting ~10 CSs per cell from the beginning, middle and end of the recording session.
        
Keyboard shortcuts: 
        Set a range : Full: Q,  1s: W,  1ms: E 
        Zoom in: R,  zoom out: T
        Move forward: F,  move backward: D
        Go to previous CS: C,  go to next CS: V"""
        return text

    # FUNCTIONS SECOND TAB
    # creating upload for files to detect on and plotting detected spikes third tab
    def create_detect_cs_box(self):
        width = 500
        detect_cs_layout = QVBoxLayout()
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(30, 30))
        info_label.setToolTip(self.info_detect_CS())
        info_label.setFixedHeight(30)
        
        setting_button = QPushButton("Set parameters")
        setting_button.clicked.connect(self.open_setting_box2)
        
        widget1 = QWidget()
        layout1 = QHBoxLayout()
        layout1.addWidget(info_label)
        layout1.addWidget(setting_button)
        layout1.addStretch()
        widget1.setLayout(layout1)
        widget1.setContentsMargins(0,0,0,0)
        
        detect_upload_button = QPushButton("Upload a PC recording")
        detect_upload_button.clicked.connect(self.upload_detection_file)
        detect_upload_button.setFixedWidth(width)
        
        self.detect_upload_label = QLabel()
        self.detect_upload_label.setText(self.detect_fileName)

        detect_upload_weights_button1 = QPushButton("Upload your downloaded weights from Colab")
        detect_upload_weights_button1.clicked.connect(self.upload_weights)
        detect_upload_weights_button1.setFixedWidth(width)
        
        self.detect_upload_weights_label1 = QLabel()
        self.detect_upload_weights_label1.setText(self.outputName)
        
        detect_upload_weights_button2 = QPushButton("Upload your downloaded weights from Colab")
        detect_upload_weights_button2.clicked.connect(self.upload_weights)
        detect_upload_weights_button2.setFixedWidth(width)
        
        self.detect_upload_weights_label2 = QLabel()
        self.detect_upload_weights_label2.setText(self.outputName)

        detecting_button1 = QPushButton('Detect CS')
        detecting_button1.clicked.connect(self.detect_CS_starter)
        detecting_button1.setFixedWidth(width)
        
        detecting_button2 = QPushButton('Detect CS')
        detecting_button2.clicked.connect(self.start_serial_CS_detection)
        detecting_button2.setFixedWidth(width)
        
        select_detect_folder_button = QPushButton("Select folder")
        select_detect_folder_button.clicked.connect(self.select_detect_folder)
        select_detect_folder_button.setFixedWidth(width)
        
        self.select_detect_folder_label = QLabel(self.detect_folder)
        
        select_output_folder_button = QPushButton("Select folder to save output")
        select_output_folder_button.clicked.connect(self.select_output_folder)
        select_output_folder_button.setFixedWidth(width)
        
        self.select_output_folder_label = QLabel(self.detect_folder)
        
        output_suffix_widget = QWidget()
        output_suffix_widget.setMaximumWidth(width)
        output_suffix_layout = QHBoxLayout()
        output_file_label = QLabel('output file name: ')
        output_file_label.setToolTip("The name of the output file will be based on each of the input files.")
        output_file_label.setFixedWidth(150)
        filename_label = QLabel('your_filename')
        filename_label.setFixedWidth(100)
        self.output_line = QLineEdit(self.output_suffix)
        self.output_line.setFixedWidth(100)

        output_suffix_widget.setLayout(output_suffix_layout)
        
        log_widget = QWidget()
        log_widget.setMaximumWidth(width)
        log_layout = QHBoxLayout()
        log_file_label = QLabel('log file name: ')
        log_file_label.setFixedWidth(150)
        self.log_line = QLineEdit(self.logName)
        self.log_line.setFixedWidth(100)

        log_widget.setLayout(log_layout)
        
        input_widget = QWidget()
        input_widget.setMaximumWidth(width)
        input_layout = QGridLayout()
        input_layout.addWidget(output_file_label, 0, 0)
        input_layout.addWidget(QLabel('your_filename'), 0, 1)
        input_layout.addWidget(self.output_line, 0, 2)
        input_layout.addWidget(QLabel('.mat'), 0, 3)
        log_label = QLabel('log name: ')
        log_label.setToolTip("Information about detected CSs for each file will be saved in this file.")
        input_layout.addWidget(log_label, 1, 0)
        input_layout.addWidget(self.log_line, 1, 2)
        input_layout.addWidget(QLabel('.csv'), 1, 3)
        input_widget.setLayout(input_layout)
        
        single_file_box = QGroupBox("Single file")
        single_file_layout = QVBoxLayout()
        single_file_layout.addWidget(detect_upload_button)
        single_file_layout.addWidget(self.detect_upload_label)
        single_file_layout.addWidget(detect_upload_weights_button1)
        single_file_layout.addWidget(self.detect_upload_weights_label1)
        single_file_layout.addStretch()
        single_file_layout.addWidget(detecting_button1)
        single_file_box.setLayout(single_file_layout)
        
        folder_box = QGroupBox("Multiple files in a folder")
        folder_layout = QVBoxLayout()
        folder_layout.addWidget(select_detect_folder_button)
        folder_layout.addWidget(self.select_detect_folder_label)
        folder_layout.addWidget(select_output_folder_button)
        folder_layout.addWidget(self.select_output_folder_label)
        folder_layout.addWidget(detect_upload_weights_button2)
        folder_layout.addWidget(self.detect_upload_weights_label2)
        folder_layout.addWidget(input_widget)
        folder_layout.addWidget(detecting_button2)
        folder_box.setLayout(folder_layout)
        
        widget2 = QWidget()
        layout2 = QGridLayout()
        
        widget3 = QWidget()
        layout3 = QHBoxLayout()
        layout3.addStretch()
        layout3.addWidget(single_file_box)
        layout3.addStretch()
        layout3.addWidget(folder_box)
        layout3.addStretch()
        widget3.setLayout(layout3)
        
        layout2.addWidget(widget3, 0, 0, 1, 2)

        layout2.setVerticalSpacing(100)
        widget2.setLayout(layout2)
        
        
        detect_cs_layout.addWidget(widget1)
        detect_cs_layout.addSpacing(100)
        detect_cs_layout.addWidget(widget2)
        detect_cs_layout.addStretch()

        self.detect_cs_box.setLayout(detect_cs_layout)
        
    def open_setting_box2(self):
        dialog = QDialog()
        dialog.setWindowTitle("Setting")
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.close)
        layout = QFormLayout()
        layout.addRow(QLabel("Set variable names before loading files."))
        layout.addRow(QLabel("Sampling rate [Hz]"), self.set_samplingRate())
        layout.addRow(QLabel("Action potential variable name"), self.set_HighVarname())
        layout.addRow(QLabel("LFP variable name"), self.set_LFPVarname())
        layout.addRow(QLabel("SS train variable name"), self.set_SSVarname())
        # layout.addRow(QLabel("Max. CSs to select"), self.set_maxCSs())
        layout.addRow(ok_button)
        dialog.setLayout(layout)
        dialog.exec_()
        dialog.show()

    # creating file upload dialog

    def upload_detection_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload files for CS detection", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        print(fileName)
        
        self.load_detection_data(fileName)
            
    def load_detection_data(self, fileName):
        if fileName:
            mat = sp.loadmat(fileName)
            if not self.LFP_varname in mat.keys():
                errorbox = QMessageBox()
                errorbox.setWindowTitle("Error")
                errorbox.setText("Variable [" + self.LFP_varname +"] not found.")
                errorbox.exec_()
            elif not self.HIGH_varname in mat.keys():
                errorbox = QMessageBox()
                errorbox.setWindowTitle("Error")
                errorbox.setText("Variable [" + self.HIGH_varname +"] not found.")
                errorbox.exec_() 
            else:
                self.detect_LFP = get_field_mat(mat,[self.LFP_varname])
                self.detect_LFP = norm_LFP(self.detect_LFP, self.sampling_rate)
                self.detect_HIGH = get_field_mat(mat, [self.HIGH_varname])
                self.detect_HIGH = norm_high_pass(self.detect_HIGH)
                self.mat = mat
                
                ext = fileName.split('.')[-1]
                self.detect_fileName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
                print(self.detect_fileName)
                self.load_file_label.setText(self.detect_fileName)
                self.detect_upload_label.setText(self.detect_fileName)
            
    def select_detect_folder(self):
        self.detect_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if len(self.detect_folder) == 0:
            self.detect_folder = "No folder selected"
        print('selected detect folder: ', self.detect_folder)
        self.select_detect_folder_label.setText(self.detect_folder)
        return self.detect_folder
    
    def select_output_folder(self):
        self.output_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if len(self.output_folder) == 0:
            self.output_folder = "No folder selected"
        print('selected save folder:', self.output_folder)
        self.select_output_folder_label.setText(self.output_folder)
        return self.output_folder

    # def get_simpleSpikes(self):
    #     text, okPressed = QInputDialog.getText(self, "Enter SS data name", "Your data:", QLineEdit.Normal, "SS")
    #     if okPressed and text != '':
    #         self.SS_varname = text

    # def get_integerSS(self):
    #     i, okPressed = QInputDialog.getInt(self, "Enter sampling rate", "Sampling rate in Hz:", 1000, 0,
    #                                        10000, 100)
    #     if okPressed and i > 0:
    #         self.sampling_rate_SS = i

    def upload_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload weights", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.weights = fileName
            self.detect_upload_weights_label1.setText(self.weights.split('/')[-1])
            self.detect_upload_weights_label2.setText(self.weights.split('/')[-1])

    def detect_CS_starter(self):
        if self.detect_fileName == "No file":
            print('Detect file not selected.')
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Detect file not selected.")
            errorbox.exec_()
        elif self.weights == []:
            print('Weight file not selected.')
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Weight file not selected.")
            errorbox.exec_()
        else:            
            self.runningbox = QMessageBox()
            self.runningbox.show()
            self.runningbox.setWindowTitle("Running")
            self.runningbox.setText("No CSs detected")
            # runningbox.exec()
    
            self.process_detect_CS()
            
            self.runningbox.done(1)
            print("\a")
    
            cs_infobox = QMessageBox()
            cs_infobox.setText('{} CSs found'.format(len(self.CS_onset)))
            cs_infobox.exec()
            
            self.save_detectFileDialog()
        
    def process_detect_CS(self):
        output = detect_CS(self.weights, self.detect_LFP, self.detect_HIGH)
        print('Detecting CSs...')
        cs_onset = output['cs_onset']
        cs_offset = output['cs_offset']
        cluster_ID = output['cluster_ID']
        embedding = output['embedding']
        
        self.sort_clusters(cluster_ID)
        
        self.CS_onset = cs_onset
        self.CS_offset = cs_offset
        self.embedding = embedding
        
    def sort_clusters(self, cluster_ID):
        # sort clusters by cluster size
        clusters = np.unique(cluster_ID)
        n_clusters = len(clusters)
        cluster_size = np.zeros(n_clusters)
        cluster_ID_sorted = np.zeros_like(cluster_ID)
        print('clusters: ', clusters)
        print('n_clusters: ', n_clusters)
        for i in range(n_clusters):
            cluster_size[i] =  (cluster_ID == clusters[i]).sum()
        print('cluster_size: ', cluster_size)
        clusters_sorted_idx = np.argsort(cluster_size)[::-1]
        cluster_size_sorted = np.sort(cluster_size)[::-1]
        for i in range(n_clusters):
            cluster_ID_sorted[cluster_ID==clusters[clusters_sorted_idx[i]]] = i+1
        clusters_sorted = np.sort(np.unique(cluster_ID_sorted))
        # print(np.unique(cluster_ID_sorted), clusters, clusters_sorted)
        
        self.cluster_ID = cluster_ID_sorted
        self.cluster_ID_save = cluster_ID_sorted.copy()
        self.clusters = clusters_sorted
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size_sorted
        self.clusters_selected = [i+1 for i in range(n_clusters)]
        self.is_cluster_selected = [True for i in range(n_clusters)]
        

    def save_detectFileDialog(self):
        self.output_suffix = self.output_line.text()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save detected data", self.detect_fileName.split('.')[-2]+ self.output_suffix + '.mat',
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        print('self.detect_fileName: ',self.detect_fileName)
        print('fileName: ',fileName)
        self.save_detectedCS(fileName)
        # if fileName:
        #     sp.savemat(fileName, {'CS_onset': self.CS_onset,
        #                           'CS_offset': self.CS_offset,
        #                           'cluster_ID': self.cluster_ID,
        #                           'embedding': self.embedding}, do_compression=True)
            
        #     ext =  fileName.split('.')[-1]
        #     self.outputName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
            
        #     self.load_output_label.setText(self.outputName)
            
    def save_detectedCS(self, fileName):
        if fileName:
            sp.savemat(fileName, {'CS_onset': self.CS_onset,
                                  'CS_offset': self.CS_offset,
                                  'cluster_ID': self.cluster_ID,
                                  'embedding': self.embedding}, do_compression=True)
            
            ext =  fileName.split('.')[-1]
            self.outputName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
            
            self.load_output_label.setText(self.outputName)
            
    def start_serial_CS_detection(self):
        
        def msgButtonClick(click):
            if click.text() == 'Proceed':
                
                self.output_suffix = self.output_line.text()
                print('self.output_suffix',self.output_suffix)
                print(click.text())
                self.process_serial_CS_detection(matfiles, correct_file_list)
            else:
                print(click.text())
        
        if self.detect_folder =="No folder selected" or self.detect_folder == "":
            print('Training folder not selected.')
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Training folder not selected.")
            errorbox.exec_()
            print('detect_folder',self.detect_folder)
        elif self.output_folder =="No folder selected" or self.output_folder == "":
            print('Output folder not selected.')
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Output folder not selected.")
            errorbox.exec_()
        elif self.weights == []:
            print('Weights not selected.')
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Weights not selected.")
            errorbox.exec_()
        else:
        
            _filenames = next(os.walk(self.detect_folder), (None, None, []))[2]
            # print('filenames', _filenames)
            matfiles = [filename for filename in _filenames if filename[-4:] == '.mat']
            # print(matfiles)
            correct_file_list = self.make_correct_file_list(matfiles)
            idx = np.where(np.invert(correct_file_list))[0]
            failed_file_list = [matfiles[i] for i in idx]
            text1 = '{}/{} files will be inspected. \n'.format(len(matfiles)-len(idx), len(matfiles))
            text2 = 'Following {} files did not match the format and will not be inspected. Check the format again:\n\n'.format(len(idx))
            if len(idx)>0:
                text3 = '\n'.join(failed_file_list)
            else:
                text3 = 'all files will be inspected.'
            
            message_box = QMessageBox()
            message_box.setText(text1+text2+text3)
            message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            message_box.buttonClicked.connect(msgButtonClick)
            buttonY = message_box.button(QMessageBox.Ok)
            buttonY.setText('Proceed')
            buttonN = message_box.button(QMessageBox.Cancel)
            buttonN.setText('Cancel')
            message_box.exec()
            
        return
    
    
    def make_correct_file_list(self, matfiles):
        n_files = len(matfiles)
        correct_high_shape_list = np.zeros(n_files, dtype=bool)
        correct_lfp_shape_list = np.zeros(n_files, dtype=bool)
        is_same_shape_list = np.zeros(n_files, dtype=bool)
        for i, file in enumerate(matfiles):
            # print('file',file)
            matfile = sp.whosmat(self.detect_folder + '/' + file)
            high_contain_list = list(map(lambda var: True if var[0] == self.HIGH_varname else False, matfile))
            print(high_contain_list)
            if sum(high_contain_list) > 0:
                high_shape = matfile[high_contain_list.index(True)][1]
                # print('high_shape',high_shape)
                if len(high_shape) == 2:
                    correct_high_shape = (high_shape[0] == 1 and  high_shape[1] > 1)
                else:
                    correct_high_shape = False
            else:
                correct_high_shape = False
            # print('correct_high_shape', correct_high_shape)
            correct_high_shape_list[i] = correct_high_shape
            
            lfp_contain_list = list(map(lambda var: True if var[0] == self.LFP_varname else False, matfile))
            print(lfp_contain_list)
            if sum(lfp_contain_list) > 0:
                lfp_shape = matfile[lfp_contain_list.index(True)][1]
                if len(lfp_shape) == 2:
                    correct_lfp_shape = (lfp_shape[0] == 1 and  lfp_shape[1] > 1)
                else:
                    correct_lfp_shape == False
            else:
                correct_lfp_shape = False
            # print('correct_lfp_shape', correct_lfp_shape)
            correct_lfp_shape_list[i] = correct_lfp_shape
            
            if correct_high_shape and correct_lfp_shape:
                is_same_shape = high_shape[1] == lfp_shape[1]
            else:
                is_same_shape = False
            is_same_shape_list[i] = is_same_shape 
            
        # print('correct_high_shape_list',correct_high_shape_list)
        # print('correct_lfp_shape_list',correct_lfp_shape_list)
        # print('is_same_shape_list',is_same_shape_list)
        
        correct_file_list = correct_high_shape_list * correct_lfp_shape_list * is_same_shape_list
        print('correct_file_list',correct_file_list)
        
        return correct_file_list
    
    def process_serial_CS_detection(self, matfiles, correct_file_list):
        idx = np.where(correct_file_list)[0]
        files_new = [matfiles[i] for i in idx]
        n_files = len(files_new)
        logfile = pd.DataFrame(files_new, columns=['file name'])
        n_cs = np.zeros(n_files, dtype=int)
        n_clusters = np.zeros(n_files, dtype=int)
        cluster_size = np.zeros(n_files, dtype=object)
        
        for i, file in enumerate(files_new):                
            fileName = self.detect_folder + '/' + files_new[i]
            print('process...', correct_file_list[i], fileName)
            print('output_folder:',self.output_folder)
            
            runningbox = QMessageBox()
            runningbox.setFixedWidth(300)
            runningbox.show()
            runningbox.setWindowTitle("{}/{} files...".format(i+1, len(files_new)))
            
            self.load_detection_data(fileName)
            self.process_detect_CS()
            
            n_cs[i] = len(self.CS_onset)
            n_clusters[i] = self.n_clusters
            cluster_size[i] = self.cluster_size.astype(int)
            
            runningbox.done(1)
            
            saveName = self.output_folder + '/' + files_new[i].split('.')[-2] + self.output_suffix + '.mat'
            print('saveName',saveName)
            self.save_detectedCS(saveName)
        
        logfile['#CS'] = n_cs.tolist()
        logfile['#clusters'] = n_clusters.tolist()
        logfile['cluster size'] = cluster_size.tolist()
        print(logfile)
        
        logfile.to_csv(self.output_folder + '/' + self.logName + '.csv', index=False)
             
    # explanation texts for the second tab
    def info_detect_CS(self):
        text = """1. Upload a file in which you want to detect CSs (stored in .mat format as in labeling process).
    - high band-passed action potential (1 x time)
    - low band-passed LFP signal (1 x time)
    - SS train (1 x time, 1 if the spike is fired, otherwise 0)
    SS train is optional and not used for CS detection, but useful for post-pr4ocessing. 
    Sampling frequency of the SS train can be lower than the other signals.
    Although not recommended, in case no LFP signal is available, try using the same high band-passed signal as LFP.
2. Upload the weights of the network trained in Google Colab.
3. Detect CSs.
    The output is saved in .mat format with the following variables:
        - CS_onset: Times of CS start (1 x # of CSs)
        - CS_offset: Times of CS end (1 x # of CSs)
        - cluster_ID: Cluster ID for each CS (1 x # of CSs)
        - embedding: Two dimensional representation of CS feature space (# of CSs x 2)"""
        return text

    # FUNCTIONS THIRD TAB
    def align_spikes(self, spikes, alignment, l1=300, l2=300):
        N = alignment.size
        spikes_aligned = np.zeros([N, l1+l2+1]) * np.nan
        for i in range(N):
            if alignment[i] - l1 < 0:
                frag = np.hstack([np.zeros([l1-alignment[i]+1])*np.nan, spikes[0:alignment[i]+l2]])
            elif alignment[i]+l2+1 > spikes.shape[0]:
                frag = np.hstack([spikes[alignment[i]-l1:-1], np.zeros(alignment[i]+l2+2-spikes.shape[0])*np.nan])
            else:
                frag = spikes[alignment[i]-l1:alignment[i]+l2+1]
            spikes_aligned[i, :] = frag
        
        return spikes_aligned       
    
    def create_load_files_for_plot_box(self):
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(30)
        info_label.setToolTip(self.info_loading_files_for_plot())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)
        info_label.setContentsMargins(0,10,0,0)
        
        width = 220
        
        setting_box = QPushButton("Set parameters")
        setting_box.setFixedWidth(width)
        setting_box.clicked.connect(self.open_setting_box2)
        
        load_file_widget = QWidget()
        load_file_layout = QHBoxLayout()
        load_file_layout.setContentsMargins(0, 10, 10, 0)
        load_file_layout.setSpacing(10)
        load_file_button = QPushButton("Load a PC recording")
        load_file_button.clicked.connect(self.upload_detection_file)
        load_file_button.setFixedWidth(width)
        
        self.load_file_label = QLabel(self.detect_fileName)
        load_file_layout.addWidget(load_file_button)
        load_file_layout.addWidget(self.load_file_label)
        load_file_widget.setLayout(load_file_layout)        
        
        load_output_widget = QWidget()
        load_output_layout = QHBoxLayout()
        load_output_layout.setContentsMargins(0, 10, 10, 10)
        load_output_layout.setSpacing(10)
        load_output_button = QPushButton("Load output")
        load_output_button.clicked.connect(self.open_OutputDialog)
        load_output_button.setFixedWidth(width)
        
        self.load_output_label = QLabel(self.outputName)
        load_output_layout.addWidget(load_output_button)
        load_output_layout.addWidget(self.load_output_label)
        load_output_widget.setLayout(load_output_layout)
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(setting_box)
        layout.addWidget(load_file_widget)
        layout.addWidget(load_output_widget)
        layout.setContentsMargins(0,0,0,0) 
        layout.setSpacing(0)
        widget.setLayout(layout)

        load_box_layout = QHBoxLayout()
        load_box_layout.addWidget(info_label)
        load_box_layout.addWidget(widget)
        load_box_layout.setContentsMargins(0,0,0,0) 
        load_box_layout.setSpacing(0)
        self.load_files_for_plot_box.setLayout(load_box_layout)
        # self.load_files_for_plot_box.setStyleSheet("border: 1px solid black;")        
        
    def create_show_data_box(self):
        show_data_layout = QVBoxLayout()
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(20, 20))
        info_label.setFixedWidth(20)
        info_label.setToolTip(self.info_select_clusters())
        info_label.setAlignment(Qt.AlignTop)
        info_label.setAlignment(Qt.AlignHCenter)

        create_cluster_selection_button = QPushButton('Select CS clusters')
        create_cluster_selection_button.clicked.connect(self.generate_cluster_list)
        
        widget = QWidget()
        layout1 = QHBoxLayout()
        layout1.addWidget(info_label)
        layout1.addWidget(create_cluster_selection_button)
        layout1.setSpacing(0)
        layout1.setContentsMargins(0,0,0,0)
        widget.setLayout(layout1)

        select_widget = QWidget()
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout()
        
        layout2 = QVBoxLayout()
        layout2.addWidget(widget)
        layout2.addWidget(self.checkbox_widget)
        layout2.addStretch()
        layout2.setContentsMargins(0,0,0,0)
        
        select_widget.setLayout(layout2)
        # select_widget.setStyleSheet('border:1px solid black;')
        
        saving_button = QPushButton('Save selected cluster data')
        saving_button.clicked.connect(self.save_selected_cluster)

        select_cluster_box = QGroupBox("Select clusters")
        select_cluster_box_layout = QVBoxLayout()
        select_cluster_box_layout.addWidget(select_widget)
        select_cluster_box_layout.addWidget(saving_button)

        select_cluster_box.setLayout(select_cluster_box_layout)
        
        show_data_layout.addWidget(select_cluster_box)

        self.select_show_data_box.setLayout(show_data_layout)
        # self.select_show_data_box.setStyleSheet('border: 1px solid red;')
        
    def open_setting_box3(self):
        dialog = QDialog()
        dialog.setWindowTitle("Set parameters")
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.close)
        layout = QFormLayout()
        
        # layout.addRow(QLabel("Sampling rate [Hz]"), self.set_samplingRate())
        # layout.addRow(QLabel("Action potential variable name"), self.set_HighVarname())
        # layout.addRow(QLabel("LFP variable name"), self.set_LFPVarname())
        
        cs_xlim_widget = QWidget()
        cs_xlim_layout = QHBoxLayout()
        cs_xlim_layout.setContentsMargins(0,0,0,0)
        cs_xlim_layout.addWidget(self.set_time_before_CS())
        cs_xlim_layout.addStretch()
        cs_xlim_layout.addWidget(QLabel('to'))
        cs_xlim_layout.addStretch()
        cs_xlim_layout.addWidget(self.set_time_after_CS())
        cs_xlim_widget.setLayout(cs_xlim_layout)
        cs_xlim_label = QLabel("Time range for CS & LFP [ms]")
        cs_xlim_label.setToolTip("Time range of action potential and LFP from CS onset")
        
        # layout.addRow(QLabel("SS train variable name"), self.set_SSVarname())
        layout.addRow(QLabel("SS sampling rate [Hz]"), self.set_samplingRate_SS())
        layout.addRow(cs_xlim_label, cs_xlim_widget)      
                
        ms_label1 = QLabel("Marker size for feature space")
        layout.addRow(ms_label1, self.set_ms1())
        
        ss_sort_label = QLabel("SS raster sort by")
        ss_sort_label.setToolTip("Sort SSs by CS cluster or time")
        layout.addRow(ss_sort_label, self.set_SS_sorting())

        sigma_label = QLabel("Gaussian kernel size [ms]")
        sigma_label.setToolTip("Used for computing SS firing rate")
        layout.addRow(sigma_label, self.set_sigma())
        
        ms_label2 = QLabel("Marker size for SS raster")
        layout.addRow(ms_label2, self.set_ms2())
        
        ss_xlim_widget = QWidget()
        ss_xlim_layout = QHBoxLayout()
        ss_xlim_layout.setContentsMargins(0,0,0,0)
        ss_xlim_layout.addWidget(self.set_time_before_CS_for_SS())
        ss_xlim_layout.addStretch()
        ss_xlim_layout.addWidget(QLabel('to'))
        ss_xlim_layout.addStretch()
        ss_xlim_layout.addWidget(self.set_time_after_CS_for_SS())
        ss_xlim_widget.setLayout(ss_xlim_layout)
        
        ss_xlim_label = QLabel("Time range for SS raster [ms]")
        ss_xlim_label.setToolTip("Time range of SS raster from CS onset")
        
        layout.addRow(ss_xlim_label, ss_xlim_widget)
        layout.addRow(ok_button)
        dialog.setLayout(layout)
        dialog.exec_()
        dialog.show()
        
    def set_samplingRate_SS(self):
        def changeSamplingRate():
            self.sampling_rate_SS = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(100, 10000)
        spinbox.setValue(self.sampling_rate_SS)
        spinbox.valueChanged.connect(changeSamplingRate)
        return spinbox
    
    def set_SSVarname(self):
        def changeText():
            self.SS_varname = lineedit.text()
        lineedit = QLineEdit(self.SS_varname)
        lineedit.textChanged.connect(changeText)
        return lineedit
    
    def set_SS_sorting(self):
        def changeText():
            self.ss_sort = combobox.currentText()
        combobox = QComboBox()
        combobox.addItem('cluster')
        combobox.addItem('time')
        combobox.currentIndexChanged.connect(changeText)
        combobox.setCurrentText(self.ss_sort)
        return combobox
    
    def set_sigma(self):
        def changeSigma():
            self.sigma = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 20)
        spinbox.setValue(self.sigma)
        spinbox.valueChanged.connect(changeSigma)
        return spinbox
    
    def set_time_before_CS(self):
        def change():
            self.t1 = -spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(-20, -1)
        spinbox.setValue(-self.t1)
        spinbox.valueChanged.connect(change)
        return spinbox
    
    def set_time_after_CS(self):
        def change():
            self.t2 = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 20)
        spinbox.setValue(self.t2)
        spinbox.valueChanged.connect(change)
        return spinbox
    
    def set_time_before_CS_for_SS(self):
        def change():
            self.t1_ss = -spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(-100, -1)
        spinbox.setValue(-self.t1_ss)
        spinbox.valueChanged.connect(change)
        return spinbox
    
    def set_time_after_CS_for_SS(self):
        def change():
            self.t2_ss = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 100)
        spinbox.setValue(self.t2_ss)
        spinbox.valueChanged.connect(change)
        return spinbox
    
    def set_ms1(self):
        def change():
            self.ms1 = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 10)
        spinbox.setValue(self.ms1)
        spinbox.valueChanged.connect(change)
        return spinbox
    
    def set_ms2(self):
        def change():
            self.ms2 = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 10)
        spinbox.setValue(self.ms2)
        spinbox.valueChanged.connect(change)
        return spinbox
    
    def open_OutputDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload data", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.upload_output(fileName)
            
    def upload_output(self, fileName):
        output = sp.loadmat(fileName)
        if not 'CS_onset' in output.keys():
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Variable [" + 'CS_onset' +"] not found.")
            errorbox.exec_()
        elif not 'CS_offset' in output.keys():
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Variable [" + 'CS_offset' +"] not found.")
            errorbox.exec_()
        elif not 'cluster_ID' in output.keys():
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Variable [" + 'Cluster_ID' +"] not found.")
            errorbox.exec_()
        elif not 'embedding' in output.keys():
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Variable [" + 'embedding' +"] not found.")
            errorbox.exec_()
        else:    
            cs_onset = output['CS_onset'].flatten()
            cs_offset = output['CS_offset'].flatten()
            cluster_ID = output['cluster_ID'].flatten()
            embedding = output['embedding']
            print('cs_onset shape, cluster_ID shape, embedding.shape: ',cs_onset.shape, cluster_ID.shape, embedding.shape)
            
            self.CS_onset = cs_onset
            self.CS_offset = cs_offset
            self.embedding = embedding
            
            self.sort_clusters(cluster_ID)
            
            ext = fileName.split('.')[-1]
            self.outputName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
            self.load_output_label.setText(self.outputName)
        4
    def generate_cluster_list(self):
        self.is_cluster_selected = []
        self.combobox = []
        self.checkbutton = []
        index = self.checkbox_layout.count()
        while(index > 0):
            # print('self.checkbox_layout',self.checkbox_layout)
            # print('self.checkbox_layout.itemAt(index-1)',index,self.checkbox_layout.itemAt(index-1))
            myWidget = self.checkbox_layout.itemAt(index-1).widget()
            myWidget.setParent(None)
            index -=1
        
        print('n_clusters: ',self.n_clusters)
        if self.n_clusters:
            for i in range(self.n_clusters):
                checkbox = QCheckBox("n={}".format(self.cluster_size[i].astype(int)))
                checkbox.setFixedWidth(90)
                self.checkbutton.append(checkbox)
                self.is_cluster_selected.append(True)
                self.checkbutton[i].setCheckState(self.is_cluster_selected[i])
                self.checkbutton[i].setTristate(False)
                combobox = QComboBox()
                for j in range(self.n_clusters):
                    color = QPixmap(50,50)
                    color.fill(QColor(self.colors[(j)%len(self.colors)]))
                    icon = QIcon(color)
                    combobox.addItem(icon, 'cluster {}'.format(j+1))
                combobox.setCurrentIndex(self.clusters[i]-1)
                combobox.setFixedWidth(120)
                combobox.setStyleSheet('selection-background-color: rgb(245, 245, 245); selection-color: rgb(0, 0, 0)')
                self.combobox.append(combobox)
                # textcolor = 'color: ' + self.colors[self.combobox[i].currentIndex()]
                textcolor = 'color: black'
                checkbox.setStyleSheet(textcolor + ';border: none;')
                check_widget = QWidget()
                check_layout = QHBoxLayout()
                check_layout.addWidget(self.checkbutton[i])
                check_layout.addWidget(self.combobox[i])
                check_layout.setContentsMargins(0,0,0,0)
                check_layout.setSpacing(0)
                check_widget.setLayout(check_layout)
                self.checkbox_layout.addWidget(check_widget)
                self.checkbutton[i].toggled.connect(self.checkbutton_clicked)
                self.clusters_selected[i] = self.combobox[i].currentIndex()+1
            update_button = QPushButton("Update")
            update_button.clicked.connect(self.update_clusters)
            
            self.checkbox_layout.addWidget(update_button)
            print(self.checkbox_layout.children())
            self.checkbox_widget.setLayout(self.checkbox_layout)
        else:
            self.checkbox_layout.addWidget(QLabel("No cluster"))
            self.checkbox_widget.setLayout(self.checkbox_layout)
        # print('self.checkbox_layout',self.checkbox_layout.count())
        # print('self.checkbox_widget',self.checkbox_widget.children())
        # self.checkbox_widget.setStyleSheet('border: 1px solid black;')

    def checkbutton_clicked(self):
        self.set_cluster_selected()
        print('is_cluster_selected: ', self.is_cluster_selected)

    def set_cluster_selected(self):
        n = len(self.checkbutton)
        for i in range(n):
            self.is_cluster_selected[i] = self.checkbutton[i].isChecked()

    def create_cluster_plotting_box(self):
        cluster_plotting_layout = QGridLayout()
        
        navi = QWidget()
        toolbar = NavigationToolbar(self.canvas2, self)
        
        setting_button = QPushButton("Setting for plot")
        setting_button.clicked.connect(self.open_setting_box3)
        
        plotting_button = QPushButton('Plot data')
        plotting_button.clicked.connect(self.plot_detected_data)
        
        info_label = QLabel()
        info_icon = self.style().standardIcon(getattr(QStyle, 'SP_MessageBoxInformation'))
        info_label.setPixmap(info_icon.pixmap(30, 30))
        info_label.setToolTip(self.info_plot_detected_CS())
        info_label.setFixedHeight(30)

        navi_layout = QHBoxLayout()
        navi_layout.addWidget(info_label)
        navi_layout.addWidget(setting_button)
        navi_layout.addWidget(plotting_button)
        navi_layout.addWidget(toolbar)
        navi_layout.setContentsMargins(0, 0, 0, 0)
        navi.setLayout(navi_layout)
        
        cluster_plotting_layout.addWidget(navi, 1, 0)
        cluster_plotting_layout.addWidget(self.canvas2, 2, 0)

        self.cluster_plotting_box.setLayout(cluster_plotting_layout)
        
    def update_clusters(self):
        for i in range(len(self.combobox)):
            idx = self.cluster_ID == i + 1
            self.cluster_ID_save[idx] = self.combobox[i].currentIndex()+1
            self.clusters_selected[i] = self.combobox[i].currentIndex()+1
        print('cluster_ID: ', self.cluster_ID)
        print('cluster_ID_save: ', self.cluster_ID_save)
        
        self.plot_detected_data()


    def plot_detected_data(self):
        
        self.canvas2.CS.cla()
        self.canvas2.LFP.cla()
        self.canvas2.CS_clusters.cla()
        self.canvas2.SS.cla()
        self.canvas2.ax2.cla()
        self.canvas2.draw()
        
        if len(self.detect_HIGH) == 0:
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("Action potential not found.")
            errorbox.exec_()
        elif len(self.detect_LFP) == 0:
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("LFP not found.")
            errorbox.exec_()
        elif self.cluster_ID_save.size==0:
            errorbox = QMessageBox()
            errorbox.setWindowTitle("Error")
            errorbox.setText("No detected CS was found.")
            errorbox.exec_()
        else:
            cluster_ID = self.cluster_ID_save
            embedding = self.embedding
    
            for i in range(len(self.is_cluster_selected)):
                if not self.is_cluster_selected[i]:
                    cluster_ID[self.cluster_ID == i+1] = 0
            
            for i in np.unique(cluster_ID[cluster_ID!=0]):
                idx = cluster_ID == i
                self.canvas2.CS_clusters.plot(embedding[idx,0], embedding[idx,1], '.',  c=self.colors[(i-1)%len(self.colors)], ms=self.ms1)
            self.canvas2.CS_clusters.set_xlabel('Dimension 1')
            self.canvas2.CS_clusters.set_ylabel('Dimension 2')
            self.canvas2.CS_clusters.set_title('Feature space', loc='left')
    
            t = np.arange(-self.t1, self.t2, 1000/(self.sampling_rate+1))
            p = 0.6
            wht = np.array([1., 1., 1., 1.])
            
            # plot CS
            cs_aligned = self.align_spikes(self.detect_HIGH, self.CS_onset, l1=self.t1*int(self.sampling_rate/1000), l2=self.t2* int(self.sampling_rate/1000))
            for i in np.unique(cluster_ID[cluster_ID!=0]):
                idx = cluster_ID == i
                color = mplcolors.to_rgba_array(self.colors[(i-1)%len(self.colors)])
                self.canvas2.CS.plot(t, cs_aligned[idx, :].T, c=color*p+wht*(1-p), lw=0.4)
            for i in np.unique(cluster_ID[cluster_ID!=0]):
                idx = cluster_ID == i
                self.canvas2.CS.plot(t, cs_aligned[idx, :].mean(0), c=self.colors[(i-1)%len(self.colors)], lw=1.5)
            self.canvas2.CS.set_xlim((-self.t1, self.t2))
            self.canvas2.CS.set_xlabel('Time from CS onset [ms]')
            self.canvas2.CS.set_title('CS', loc='left')
            self.canvas2.CS.get_xaxis().set_ticks([])
            
            # plot LFP
            lfp_aligned = self.align_spikes(self.detect_LFP, self.CS_onset, l1=self.t1*int(self.sampling_rate/1000), l2=self.t2* int(self.sampling_rate/1000))
            for i in np.unique(cluster_ID[cluster_ID!=0]):
                idx = cluster_ID == i
                color = mplcolors.to_rgba_array(self.colors[(i-1)%len(self.colors)])
                self.canvas2.LFP.plot(t, lfp_aligned[idx, :].T, c=color*p+wht*(1-p), lw=0.4)
            for i in np.unique(cluster_ID[cluster_ID!=0]):
                idx = cluster_ID == i
                self.canvas2.LFP.plot(t, lfp_aligned[idx, :].mean(0), c=self.colors[(i-1)%len(self.colors)], lw=1.5)
            self.canvas2.LFP.set_xlim((-self.t1, self.t2))
            self.canvas2.LFP.set_xlabel('Time from CS onset [ms]')
            self.canvas2.LFP.set_title('LFP', loc='left')
                
            # plot SS
            
            if self.SS_varname in self.mat.keys():
    
                t_ss = np.arange(-self.t1_ss, self.t2_ss, 1000/(self.sampling_rate_SS+1))
                self.ss_train = get_field_mat(self.mat,[self.SS_varname])
                # clusters = np.unique(cluster_ID[cluster_ID!=0])
                cs_onset_downsample = (self.CS_onset/self.sampling_rate*1000).astype(int)
                ss_aligned = self.align_spikes(self.ss_train, cs_onset_downsample, self.t1_ss, self.t2_ss)
                offset = 0
                print('ss_sort: ', self.ss_sort)
                if self.ss_sort == 'cluster':
                    for i in np.unique(cluster_ID[cluster_ID!=0]):
                        [iy, ix] = np.where(ss_aligned[cluster_ID==i, :]==1)   
                        color = mplcolors.to_rgba_array(self.colors[(i-1)%len(self.colors)])
                        self.canvas2.SS.plot(t_ss[ix], iy+offset, '.', c=color*p+wht*(1-p), ms=self.ms2)
                        offset = offset + (cluster_ID==i).sum()
                elif self.ss_sort == 'time':
                    [iy, ix] = np.where(ss_aligned[cluster_ID!=0, :]==1) 
                    for i in np.unique(cluster_ID[cluster_ID!=0]):
                        idx = np.in1d(iy, np.where(cluster_ID[cluster_ID!=0]==i)[0])
                        color = mplcolors.to_rgba_array(self.colors[(i-1)%len(self.colors)])
                        self.canvas2.SS.plot(t_ss[ix[idx]], iy[idx], '.', c=color*p+wht*(1-p), ms=self.ms2)
                self.canvas2.SS.set_ylim((0, cluster_ID[cluster_ID!=0].size))
                
                ss_conv = gaussian_filter1d(ss_aligned, self.sigma * self.sampling_rate_SS/1000, order=0)*1000
                for i in np.unique(cluster_ID[cluster_ID!=0]):
                    self.canvas2.ax2.plot(t_ss, np.nanmean(ss_conv[cluster_ID==i, :], 0), c=self.colors[(i-1)%len(self.colors)], lw=2)
                
                self.canvas2.ax2.set_ylabel('SS firing rate [spikes/s]')
                self.canvas2.SS.set_xlabel('Time from CS onset [ms]')
                self.canvas2.SS.set_ylabel('CS')
                self.canvas2.SS.set_title('SS', loc='left')
                self.canvas2.SS.set_xlim([-self.t1_ss, self.t2_ss])
    
            self.canvas2.draw()

    def save_selected_cluster(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        print(self.detect_fileName)
        if self.detect_fileName == 'No file' or len(self.detect_fileName)==0:
            print('error:',self.detect_fileName, 'to save')
        else:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save selected cluster data", self.detect_fileName.split('.')[-2]+'_cs_clusters.mat',
                                                      "All Files (*);;MATLAB Files (*.mat)", options=options)
    
            if fileName:
                self.get_selected_clusters()
                sp.savemat(fileName, {'CS_onset': self.save_CS_onset,
                                      'CS_offset': self.save_CS_offset,
                                      'cluster_ID': self.save_cluster_ID,
                                      'embedding': self.save_embedding}, do_compression=True)
                print(fileName + ' saved')

    def get_selected_clusters(self):
        selected_clusters = self.clusters[np.where(np.array(self.is_cluster_selected)==True)]
        selected_indices = np.where(np.isin(np.array(self.cluster_ID), selected_clusters))

        self.save_cluster_ID = self.cluster_ID_save[selected_indices]
        self.save_CS_onset = self.CS_onset[selected_indices]
        self.save_CS_offset = self.CS_offset[selected_indices]
        self.save_embedding = self.embedding[selected_indices, :].squeeze()

    # explanation texts for the third tab
    def info_loading_files_for_plot(self):
        text = """1. Load PC recording
    The same data as the one used in the CS detection process.
2. Load output
    The output file produced by using the above-mentioned file.
Note: After the CS detection process, the files are already loaded here."""
        return text
    
    def info_select_clusters(self):
        text = """1. Select CS clusters.
    - Show CS clusters
    Select which clusters you want to save by changing the checkbox.
    Merge clusters by changing the cluster ID/color. 
    - Update to see the new plot. 
2. Save the selected clusters.
    The selected CS clusters are saved in the save way as in the CS detection process:
        - CS_onset: Times of CS start (1 x # of CSs)
        - CS_offset: Times of CS end (1 x # of CSs)
        - cluster_ID: Newly selected and labeled cluster ID for each CS (1 x # of CSs)
        - embedding: Two dimensional representation of CS feature space (# of CSs x 2)"""
        return text
    
    def info_plot_detected_CS(self):
        text = """1. Set parameters for plotting.
    If no SS train is available, leave the variable name empty.
2. Plot detected clusters. Each cluster is plotted in a different color.
    - CS: High band-passed action potential aligned to CS onset.
    - LFP: Low band-passed LFP aligned to CS onset.
    - Feature space: Clustering is based on this dimensionally reduced feature space (using UMAP).
    - SS: SS raster and firing rate aligned to CS onset."""
        return text

def create():
    print("App running...")
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    main = Frame()
    main.show()
    sys.exit(app.exec_())

create()
