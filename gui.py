
# from typing import List, Any

from PyQt5.QtWidgets import (QApplication, QComboBox, QDesktopWidget, QDialog, QFileDialog, QSizePolicy, QFormLayout, QGridLayout, QGroupBox, QSpinBox, 
                             QHBoxLayout, QVBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QComboBox, QPushButton, QToolButton, QTabWidget,
                             QTextEdit, QWidget, QListWidget, QCheckBox, QLineEdit, QScrollBar)
from PyQt5.QtGui import QPainter, QIcon, QDesktopServices, QPixmap
from PyQt5.QtCore import QUrl, QSize, Qt, QRect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mplcolors
import scipy.io as sp
from scipy.ndimage import gaussian_filter1d
import numpy as np
import sys
from CS import detect_CS, norm_LFP, norm_high_pass, get_field_mat, create_random_intervals, concatenate_segments
import uneye

# from IPython import embed; 

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.high_axes = fig.add_subplot(211)
        self.high_axes.get_xaxis().set_visible(False)
        self.lfp_axes = fig.add_subplot(212, sharex=self.high_axes, sharey=self.high_axes)
        self.lfp_axes.get_xaxis().set_visible(True)
        self.lfp_axes.set_xlabel('Time')

        super(MplCanvas, self).__init__(fig)
        
class MplCanvas2(FigureCanvas):
    def __init__(self, parent=None, width=60, height=20, dpi=100):
        fig2 = Figure(figsize=(width, height), dpi=dpi)
        self.CS = fig2.add_subplot(221)
        self.LFP = fig2.add_subplot(223)
        self.CS_clusters = fig2.add_subplot(222)
        self.SS = fig2.add_subplot(224)
        self.ax2 = self.SS.twinx()

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

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
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
        self.LFP_varname = 'RAW'
        self.HIGH_varname = 'HIGH'
        self.SS_varname = 'SS'
        self.LFP = []
        self.HIGH = []
        self.Labels = []
        self.Intervals_inspected = []
        self.upload_LFP = []
        self.upload_HIGH = []
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
        self.outputName = "No output"
        self.detect_fileName = "No file"
        self.CS_onset = []
        self.CS_offset = []
        self.cluster_ID = []
        self.embedding = []
        self.n_clusters = []
        self.ss_train = []
        self.sigma = 5
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
        # self.information_box = QGroupBox("Please note:")
        self.save_box = QGroupBox("Save training data")
        self.loaded_files_box = QGroupBox("Loaded files")
        self.select_cs_box = QGroupBox("Select complex spikes")
        self.after_labeling_box = QGroupBox("After labeling")

        # List of loaded files
        self.loaded_file_listWidget = QListWidget() 
        self.loaded_file_listWidget.setFixedHeight(self.loaded_files_box.height()-70)
        # self.loaded_file_listWidget.setStyleSheet('border: 1px solid red;')
        # self.loaded_file_listWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        print(self.loaded_file_listWidget.height())

        self.loaded_file_listWidget.update()
        
        # Add groupboxes to first tab
        self.create_data_input_box()
        # self.create_information_box()
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
        layout.addWidget(self.save_box, 3, 0)
        layout.addWidget(self.after_labeling_box, 4, 0)
        left_panel.setLayout(layout)
        # self.tab_preprocessing.layout.addWidget(self.data_input_box, 0, 1)
        self.tab_label_data.layout.addWidget(left_panel, 0, 0)
        self.tab_label_data.layout.addWidget(self.select_cs_box, 0, 1)
        self.tab_label_data.layout.setColumnStretch(0, 4)
        self.tab_label_data.layout.setRowStretch(0, 0)
        self.tab_label_data.layout.setRowStretch(1, 4)

        # self.tab_preprocessing.layout.setRowStretch(0, 4)
        self.tab_label_data.setLayout(self.tab_label_data.layout)

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
        
        scale_widget = QWidget()
        scale_layout = QHBoxLayout()
        scale_layout.addStretch()
        scale_layout.addWidget(max_button)
        scale_layout.addWidget(second_button)
        scale_layout.addWidget(millisecond_button)
        scale_layout.addWidget(zoomin_button)
        scale_layout.addWidget(zoomout_button)
        scale_widget.setLayout(scale_layout)
        
        cs_widget = QWidget()
        cs_layout = QHBoxLayout()
        cs_layout.addStretch()
        cs_layout.addWidget(self.cs_counter)
        cs_layout.addWidget(prev_cs_button)
        cs_layout.addWidget(next_cs_button)  
        cs_widget.setLayout(cs_layout)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(scale_widget)
        ctrl_layout.addWidget(cs_widget)
        
        ctrl = QWidget()
        ctrl.setLayout(ctrl_layout)

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
        data_input_layout = QVBoxLayout()
        
        upload_button = QPushButton("Add PC for manual labeling")
        upload_button.setToolTip('Upload and plot the first file for labeling')
        upload_button.clicked.connect(self.openFileNameDialog)
        
        setting_button = QPushButton("Set parameters")
        setting_button.clicked.connect(self.open_setting_box)

        data_input_layout.addWidget(setting_button)
        data_input_layout.addWidget(upload_button)

        self.data_input_box.setLayout(data_input_layout)
        
    def create_save_box(self):
        save_layout = QVBoxLayout()
        
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.saveFileDialog)
        
        save_layout.addWidget(save_button)
        
        self.save_box.setLayout(save_layout)

    # creating the box in the first tab containing user information
    def open_setting_box(self):
        dialog = QDialog()
        dialog.setWindowTitle("Setting")
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.close)
        layout = QFormLayout()
        layout.addRow(QLabel("Sampling rate [Hz]"), self.set_samplingRate())
        layout.addRow(QLabel("High-passed action potential variable name"), self.set_HighVarname())
        layout.addRow(QLabel("LFP variable name"), self.set_LFPVarname())
        layout.addRow(QLabel("Max. CSs to select"), self.set_maxCSs())
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
        print(self.loaded_file_listWidget.currentRow())
                
        plot_file_button = QPushButton("Plot")
        plot_file_button.clicked.connect(self.set_current_file)        
                
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_loaded_file)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(plot_file_button)
        button_layout.addWidget(remove_button)
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        
        button_widget.setFixedHeight(70)
        # button_widget.setStyleSheet('border: 1px solid red;')
        height = self.loaded_files_box.height()-button_widget.height()-60
        self.loaded_file_listWidget.setFixedHeight(height)
        layout.addWidget(self.loaded_file_listWidget)
        layout.addStretch()
        layout.addWidget(button_widget)
        # button_widget.sizeHint()
        
        self.loaded_files_box.setLayout(layout)
        self.loaded_files_box.update()
        self.set_current_file()
        
    def remove_loaded_file(self):
        idx = self.loaded_file_listWidget.currentRow()
        print(idx)
        if self.LFP:
            self.loaded_file_listWidget.takeItem(idx)
            self.ID.pop(idx)
            self.LFP.pop(idx)
            self.HIGH.pop(idx)
            self.Labels.pop(idx)
            
            self.canvas.high_axes.cla()
            self.canvas.lfp_axes.cla()
            self.canvas.draw_idle()
            self.loaded_files_box.update()
    
    def set_current_file(self):
        idx = self.loaded_file_listWidget.currentRow()
        print(idx)
        if self.LFP:
            self.upload_LFP = self.LFP[idx]
            self.upload_HIGH = self.HIGH[idx]
            self.label = self.Labels[idx]
            self.cs_spans = self.cs_spans_all[idx]
            print(self.ID[idx])
            print('self.cs_spans',self.cs_spans)
            self.plot_data()
            self.cs_counter.setText('{} CSs selected'.format(self.cs_spans.T.shape[0]))
    
    def create_after_labeling_box(self):
        after_labeling_layout = QHBoxLayout()
        
        goto_Colab_button = QPushButton("TRAIN ALGORITHM")
        goto_Colab_button.setToolTip('Plase finish labeling data before going to Colab')
        goto_Colab_button.clicked.connect(self.open_Colab)
        goto_Colab_button.setIcon(QIcon(('./img/colab_logo.png')))
        
        after_labeling_layout.addWidget(goto_Colab_button)
        
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
    
    def set_maxCSs(self):
        def changeMaxCSs():
            self.PC_Number = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 20)
        spinbox.setValue(self.PC_Number)
        spinbox.valueChanged.connect(changeMaxCSs)
        return spinbox

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
        self.upload_fileName = fileName.split('.')[-2].split('/')[-1]
        self.upload_LFP = np.array(mat[self.LFP_varname][0])
        self.upload_HIGH = np.array(mat[self.HIGH_varname][0])
        self.interval_inspected = np.zeros_like(self.upload_LFP)
        self.cs_span = np.zeros(2)
        self.cs_spans = np.array([[]])
        self.cs_patch = []
        self.ID.append(self.upload_fileName)
        self.LFP.append(self.upload_LFP)
        self.HIGH.append(self.upload_HIGH)
        self.Labels.append(self.label)
        self.cs_spans_all.append(self.cs_spans)
        self.Intervals_inspected.append(self.interval_inspected)
        self.plot_data()
        self.create_loaded_files_box()

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

        self.t = np.linspace(0, len(self.upload_LFP)/self.sampling_rate, len(self.upload_LFP))

        self.canvas.high_axes.cla()
        self.canvas.lfp_axes.cla()
        self.canvas.high_axes.plot(self.t, high_data, 'tab:blue', lw=0.4)
        print('cs_spans_all', self.cs_spans_all)
        for i in range(self.cs_spans.shape[1]):
            ylim = self.canvas.high_axes.get_ylim()
            
            print('cs_spans,i', self.cs_spans,self.cs_spans.shape,i)
            print('cs_spans[i]',self.cs_spans[:, i])
            patch = patches.Rectangle((self.t[self.cs_spans[0, i]], ylim[0]), np.diff(self.t[self.cs_spans[:,i]]), np.diff(ylim), linewidth=1, edgecolor='k', facecolor='r', alpha=0.2, zorder=2)
            self.canvas.high_axes.add_patch(patch)
        self.canvas.high_axes.set_xlim([0, self.t[-1]])
        self.canvas.high_axes.set_ylabel('High-pass signal')
        self.canvas.lfp_axes.plot(self.t, raw_data, 'tab:blue', lw=0.4)
        self.canvas.lfp_axes.set_xlim([0, self.t[-1]])
        self.canvas.lfp_axes.set_ylabel('Local field potential')
        self.canvas.lfp_axes.set_xlabel('time [s]')
        self.canvas.high_axes.set_title(self.upload_fileName)
        self.canvas.draw()
        self.canvas_ylim = self.canvas.lfp_axes.get_ylim()
    #     self.canvas.mpl_connect('button_release_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.click_control)
        self.canvas.mpl_connect('motion_notify_event', self.draw_span)
        self.canvas.mpl_connect('button_release_event', self.set_cs_offset)
    
    # Selecting CS 
    def click_control(self, event):
        if self.canvas.high_axes.patches:
            # for i in range(self.cs_spans.T.shape[0]):
            for i in range(len(self.canvas.high_axes.patches)):
                # if (event.xdata >=self.t[self.cs_spans[0, i]]) and (event.xdata<=self.t[self.cs_spans[1, i]]):
                x1 = self.canvas.high_axes.patches[i].get_x()
                x2 = self.canvas.high_axes.patches[i].get_x() + self.canvas.high_axes.patches[i].get_width()[0]
                print('x1,x2,event.xdata',x1,x2,event.xdata)
                if (event.xdata >= x1) and (event.xdata <= x2):  
                    self.canvas.high_axes.patches.pop(i)
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
        self.canvas.high_axes.add_patch(self.cs_patch)
        
    def draw_span(self, event):
        if self.is_clicked:
            self.cs_span[1] = event.xdata
            self.cs_patch.set_width(np.diff(self.cs_span))
            self.canvas.draw_idle()
        
    def set_cs_offset(self, event):
        if self.is_clicked:
            self.cs_span[1] = event.xdata
            self.cs_patch.set_width(np.diff(self.cs_span))
            
            if abs(np.diff(self.cs_span))<0.001: # if span is too short, it doesn't count
                print('diff',self.cs_span[0],self.cs_span[1],event.xdata,abs(np.diff(self.cs_span)))
                self.canvas.high_axes.patches.pop()
            self.canvas.draw_idle()
            self.is_clicked = False
            self.cs_span = np.empty(2)
            self.cs_patch = []
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
            print('self.cs_spans',self.cs_spans, self.cs_spans.shape, np.argsort(onset))
            self.create_labels()
            self.cs_counter.setText('{} CSs selected'.format(self.cs_spans.T.shape[0]))
            
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
        l = len(self.upload_HIGH)-1
        self.canvas.high_axes.set_xlim(self.t[0], self.t[l])
        self.canvas.lfp_axes.set_xlim(self.t[0], self.t[l])
        self.lims = np.array([0, len(self.upload_LFP)/self.sampling_rate])
        self.setupSlider(0, 0, 0)
        self.canvas.draw_idle()
        
    def set_zoom_xlim(self, width):
        maximum = np.floor(len(self.upload_LFP) / self.sampling_rate * self.step / width).astype(int)
        minimum = 0
        lims = self.canvas.lfp_axes.get_xlim()
        xlim = [lims[0], lims[0]+width]
        x0 = np.floor(xlim[0] * self.step / width).astype(int)
        print(xlim[0], x0)
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
        width = 0.05
        center = np.mean(self.canvas.high_axes.get_xlim())
        idx_center = np.absolute(self.t-center).argmin().astype(int)
        print('center',center, idx_center, self.cs_spans[0,:])
        if self.cs_spans.T.shape[0]>0:
            idx0 = np.where(self.cs_spans[0,:]<idx_center)[0]
            if len(idx0)>0:
                idx = idx0.max()
                self.lims = (self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2)
                print('prev cs',self.lims, idx)
                self.canvas.lfp_axes.set_xlim([self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2])
                self.canvas.draw_idle()
                # self.set_zoom_xlim(width)
                
    def go_to_next_CS(self):
        width = 0.05
        center = np.mean(self.canvas.high_axes.get_xlim())
        idx_center = np.absolute(self.t-center).argmin().astype(int)
        print('center',center, idx_center, self.cs_spans[0,:])
        if self.cs_spans.T.shape[0]>0:
            idx0 = np.where(self.cs_spans[0,:]>idx_center)[0]
            if len(idx0)>0:
                idx = idx0.min()
                self.lims = (self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2)
                print('next cs',self.lims, idx)
                self.canvas.lfp_axes.set_xlim([self.t[self.cs_spans[0,idx]]-width/2, self.t[self.cs_spans[0,idx]]+width/2])
                self.canvas.draw_idle()
                # self.set_zoom_xlim(width)
                
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.zoom(self.zoom_ratio)
        elif event.key() == Qt.Key_T:
            self.zoom(1/self.zoom_ratio)
        elif event.key()==Qt.Key_F:
            self.scroll.setValue(self.scroll.value() + 1)
        elif event.key()==Qt.Key_S:
            self.scroll.setValue(self.scroll.value() - 1)
        elif event.key() == Qt.Key_Q:
            self.set_max_xlim()
        elif event.key() == Qt.Key_W:
            self.set_zoom_xlim(1.0)
        elif event.key() == Qt.Key_E:
            self.set_zoom_xlim(0.05)
        elif event.key() == Qt.Key_C:
            self.go_to_prev_CS()
        elif event.key() == Qt.Key_V:
            self.go_to_next_CS()

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

    # delete last CS function
    def delete_last_CS(self):
        if self.x_values[self.backwardscounter][0] == 0:
            self.backwardscounter -= 1
        else:
            self.x_values[self.backwardscounter][0] = 0
            self.x_values[self.backwardscounter][1] = 0
            self.value_counter -= 1

    def create_labels(self):
        labels = np.zeros_like(self.upload_LFP)
        for i in range(self.cs_spans.T.shape[0]):
            labels[self.cs_spans[0,i]:self.cs_spans[1,i]] = 1
        self.label = labels
        idx = self.loaded_file_listWidget.currentRow()
        self.Labels[idx] = self.label
        self.cs_spans_all[idx] = self.cs_spans
        
        self.interval_inspected = create_random_intervals(self.sampling_rate, self.upload_LFP, self.label)
        self.Intervals_inspected[idx] = self.interval_inspected
        print(self.cs_spans_all)
        # self.Labels.append(labels)

    # FUNCTIONS SECOND TAB
    # creating upload for files to detect on and plotting detected spikes third tab
    def create_detect_cs_box(self):
        width = 500
        detect_cs_layout = QGridLayout()
        detect_cs_layout.setColumnStretch(0, 0)
        detect_cs_layout.setColumnStretch(1, 0)

        detect_upload_button = QPushButton("Upload a PC recording")
        detect_upload_button.clicked.connect(self.upload_detection_file)
        detect_upload_button.setFixedWidth(width)
        
        self.detect_upload_label = QLabel()
        self.detect_upload_label.setText(self.detect_fileName)

        detect_upload_weights_button = QPushButton("Upload your downloaded weights from Colab")
        detect_upload_weights_button.clicked.connect(self.upload_weights)
        detect_upload_weights_button.setFixedWidth(width)
        
        self.detect_upload_weights_label = QLabel()
        self.detect_upload_weights_label.setText(self.outputName)

        detecting_button = QPushButton('Detect CS')
        detecting_button.clicked.connect(self.detect_CS_starter)
        detecting_button.setFixedWidth(width)

        detect_cs_layout.addWidget(detect_upload_button, 0, 0)
        detect_cs_layout.addWidget(self.detect_upload_label, 0, 1)
        detect_cs_layout.addWidget(detect_upload_weights_button, 1, 0)
        detect_cs_layout.addWidget(self.detect_upload_weights_label, 1, 1)
        detect_cs_layout.addWidget(detecting_button, 2, 0)

        self.detect_cs_box.setLayout(detect_cs_layout)

    # creating file upload dialog

    def upload_detection_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload files for CS detection", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        print(fileName)
        
        if fileName:
            mat = sp.loadmat(fileName)
            self.detect_LFP = get_field_mat(mat,['RAW'])
            self.detect_LFP = norm_LFP(self.detect_LFP, self.sampling_rate)
            self.detect_HIGH = get_field_mat(mat, ['HIGH'])
            self.detect_HIGH = norm_high_pass(self.detect_HIGH)
            self.mat = mat
            
            ext = fileName.split('.')[-1]
            self.detect_fileName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
            print(self.detect_fileName)
            self.load_file_label.setText(self.detect_fileName)
            self.detect_upload_label.setText(self.detect_fileName)

    def get_simpleSpikes(self):
        text, okPressed = QInputDialog.getText(self, "Enter SS data name", "Your data:", QLineEdit.Normal, "SS")
        if okPressed and text != '':
            self.SS_varname = text

    def get_integerSS(self):
        i, okPressed = QInputDialog.getInt(self, "Enter sampling rate", "Sampling rate in Hz:", 1000, 0,
                                           10000, 100)
        if okPressed and i > 0:
            self.sampling_rate_SS = i

    def upload_weights(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload weights", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.weights = fileName
            self.detect_upload_weights_label.setText(self.weights.split('/')[-1])

    def detect_CS_starter(self):
        runningbox = QMessageBox()
        runningbox.setWindowTitle("Running")
        runningbox.setText("Detecting CSs....")
        runningbox.exec_()
        
        output = detect_CS(self.weights, self.detect_LFP, self.detect_HIGH)
        runningbox.done(1)
        
        cs_onset = output['cs_onset']
        cs_offset = output['cs_offset']
        cluster_ID = output['cluster_ID']
        embedding = output['embedding']
        print(cs_onset.shape, cluster_ID.shape, embedding.shape)
        
        # sort clusters by cluster size
        clusters = np.unique(cluster_ID)
        n_clusters = len(clusters)
        cluster_size = np.zeros(n_clusters)
        cluster_ID_sorted = np.zeros_like(cluster_ID)
        for i in range(n_clusters):
            cluster_size[i] =  (cluster_ID == clusters[i]).sum()
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

        self.save_detectFileDialog()

    def save_detectFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save detected data", self.detect_fileName.split('.')[-2]+'_output.mat',
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        print('self.detect_fileName',self.detect_fileName)
        print('fileName',fileName)
        if fileName:
            sp.savemat(fileName, {'CS_onset': self.CS_onset,
                                  'CS_offset': self.CS_offset,
                                  'cluster_ID': self.cluster_ID,
                                  'embedding': self.embedding}, do_compression=True)
            
            ext =  fileName.split('.')[-1]
            self.outputName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
            
            self.load_output_label.setText(self.outputName)

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
    
    def create_load_files_for_plot_box(self):
        width = 250
        load_file_widget = QWidget()
        load_file_layout = QHBoxLayout()
        load_file_layout.setContentsMargins(10, 10, 10, 0)
        load_file_layout.setSpacing(10)
        load_file_button = QPushButton("Load PC recording")
        load_file_button.clicked.connect(self.upload_detection_file)
        load_file_button.setFixedWidth(width)
        
        self.load_file_label = QLabel(self.detect_fileName)
        load_file_layout.addWidget(load_file_button)
        load_file_layout.addWidget(self.load_file_label)
        load_file_widget.setLayout(load_file_layout)        
        
        load_output_widget = QWidget()
        load_output_layout = QHBoxLayout()
        load_output_layout.setContentsMargins(10, 10, 10, 10)
        load_output_layout.setSpacing(10)
        load_output_button = QPushButton("Load output")
        load_output_button.clicked.connect(self.open_OutputDialog)
        load_output_button.setFixedWidth(width)
        
        self.load_output_label = QLabel(self.outputName)
        load_output_layout.addWidget(load_output_button)
        load_output_layout.addWidget(self.load_output_label)
        load_output_widget.setLayout(load_output_layout)

        # load_box_layout = QVBoxLayout()
        # load_box_layout.addWidget(load_file_button)
        # # load_box_layout.addWidget(loaded_file_widget)
        # load_box_layout.addWidget(load_output_button)
        
        load_box_layout = QVBoxLayout()
        load_box_layout.addWidget(load_file_widget)
        load_box_layout.addWidget(load_output_widget)
        load_box_layout.setContentsMargins(0,0,0,0) 
        load_box_layout.setSpacing(0)
        self.load_files_for_plot_box.setLayout(load_box_layout)
        # self.load_files_for_plot_box.setStyleSheet("border: 1px solid black;")
                
        
    def create_show_data_box(self):
        show_data_layout = QVBoxLayout()
        
        setting_button = QPushButton("Set parameters")
        setting_button.clicked.connect(self.open_setting_box2)
        
        load_file_button = QPushButton("Load PC recording")
        load_file_button.clicked.connect(self.upload_detection_file)
        
        loaded_file_widget = QListWidget()
        loaded_file_widget.setFixedHeight(34)
        print('height',load_file_button.sizeHint())
        
        load_output_button = QPushButton("Load output")
        load_output_button.clicked.connect(self.open_OutputDialog)

        # plotting_button = QPushButton('Plot data')
        # plotting_button.clicked.connect(self.plot_detected_data)

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
        
        load_box = QGroupBox("Load files")
        load_box_layout = QVBoxLayout()
        load_box_layout.addWidget(load_file_button)
        load_box_layout.addWidget(loaded_file_widget)
        load_box_layout.addWidget(load_output_button)
        load_box.setLayout(load_box_layout)
        
        select_cluster_box = QGroupBox("Select clusters")
        select_cluster_box_layout = QVBoxLayout()
        select_cluster_box_layout.addWidget(select_widget)
        select_cluster_box_layout.addWidget(saving_button)

        select_cluster_box.setLayout(select_cluster_box_layout)
        
        # show_data_layout.addWidget(load_box)
        show_data_layout.addWidget(select_cluster_box)

        self.select_show_data_box.setLayout(show_data_layout)

        create_cluster_selection_button.clicked.connect(self.generate_cluster_list)
        
    def open_setting_box2(self):
        dialog = QDialog()
        dialog.setWindowTitle("Set parameters")
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.close)
        layout = QFormLayout()
        layout.addRow(QLabel("SS train variable name"), self.set_SSVarname())
        layout.addRow(QLabel("SS sampling rate [Hz]"), self.set_samplingRate_SS())
        layout.addRow(QLabel("gaussian kernel size [ms]"), self.set_sigma())
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
    
    def set_sigma(self):
        def changeSamplingRate():
            self.sigma = spinbox.value()
        spinbox = QSpinBox()
        spinbox.setRange(1, 20)
        spinbox.setValue(self.sigma)
        spinbox.valueChanged.connect(changeSamplingRate)
        return spinbox
    
    def open_OutputDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Upload data", "",
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)
        if fileName:
            self.upload_output(fileName)
            ext = fileName.split('.')[-1]
            self.detect_fileName = fileName.split('.')[-2].split('/')[-1] + '.' + ext
            self.load_output_label.setText(self.detect_fileName)
        
    def upload_output(self, fileName):
        output = sp.loadmat(fileName)

        cs_onset = output['CS_onset'].squeeze()
        cs_offset = output['CS_offset'].squeeze()
        cluster_ID = output['cluster_ID'].squeeze()
        embedding = output['embedding'].squeeze()
        print(cs_onset.shape, cluster_ID.shape, embedding.shape)
        
        # sort clusters by cluster size
        clusters = np.unique(cluster_ID)
        n_clusters = len(clusters)
        cluster_size = np.zeros(n_clusters)
        cluster_ID_sorted = np.zeros_like(cluster_ID)
        for i in range(n_clusters):
            cluster_size[i] =  (cluster_ID == clusters[i]).sum()
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
                checkbox = QCheckBox("Cluster {} (n={})".format(i+1, self.cluster_size[i].astype(int)))
                textcolor = 'color: ' + self.colors[i]
                checkbox.setStyleSheet(textcolor)
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
        
        navi = QWidget()
        toolbar = NavigationToolbar(self.canvas2, self)
        
        setting_button = QPushButton("Setting for plot")
        setting_button.clicked.connect(self.open_setting_box2)
        
        plotting_button = QPushButton('Plot data')
        plotting_button.clicked.connect(self.plot_detected_data)

        navi_layout = QHBoxLayout()
        navi_layout.addWidget(setting_button)
        navi_layout.addWidget(plotting_button)
        navi_layout.addWidget(toolbar)
        navi.setLayout(navi_layout)
        
        cluster_plotting_layout.addWidget(navi, 1, 0)
        cluster_plotting_layout.addWidget(self.canvas2, 2, 0)

        self.cluster_plotting_box.setLayout(cluster_plotting_layout)


    def plot_detected_data(self):

        cluster_ID = self.cluster_ID
        cs_offset = self.CS_offset
        cs_onset = self.CS_onset
        embedding = self.embedding
        n_clusters = self.n_clusters

        self.canvas2.CS.cla()
        self.canvas2.LFP.cla()
        self.canvas2.CS_clusters.cla()
        self.canvas2.SS.cla()
        self.canvas2.ax2.cla()

        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.CS_clusters.plot(embedding[idx,0], embedding[idx,1], '.',  c=self.colors[i])
        self.canvas2.CS_clusters.set_xlabel('Dimension 1')
        self.canvas2.CS_clusters.set_ylabel('Dimension 2')
        self.canvas2.CS_clusters.set_title('CS clusters', loc='left')
        # self.canvas2.CS_clusters.get_xaxis().set_ticks([])

        t1 = 5
        t2 = 20
        t = np.arange(-t1, t2, 1000/(self.sampling_rate+1))
        p = 0.6
        wht = np.array([1., 1., 1., 1.])
        # plot CS
        cs_aligned = self.align_spikes(self.detect_HIGH, self.CS_onset, l1=t1*int(self.sampling_rate/1000), l2=t2* int(self.sampling_rate/1000))
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            color = mplcolors.to_rgba_array(self.colors[i])
            self.canvas2.CS.plot(t, cs_aligned[idx, :].T, c=color*p+wht*(1-p), lw=0.4)
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.CS.plot(t, cs_aligned[idx, :].mean(0), c=self.colors[i], lw=2)
        self.canvas2.CS.set_xlabel('Time from CS onset [ms]')
        self.canvas2.CS.set_title('CS', loc='left')
        self.canvas2.CS.get_xaxis().set_ticks([])
        
        # plot LFP
        lfp_aligned = self.align_spikes(self.detect_LFP, self.CS_onset, l1=t1*int(self.sampling_rate/1000), l2=t2* int(self.sampling_rate/1000))
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            color = mplcolors.to_rgba_array(self.colors[i])
            self.canvas2.LFP.plot(t, lfp_aligned[idx, :].T, c=color*p+wht*(1-p), lw=0.4)
        for i in range(self.n_clusters):
            idx = cluster_ID == i+1
            self.canvas2.LFP.plot(t, lfp_aligned[idx, :].mean(0), c=self.colors[i], lw=2)
        self.canvas2.LFP.set_xlabel('Time from CS onset [ms]')
        self.canvas2.LFP.set_title('LFP', loc='left')
            
        # plot SS
        if self.SS_varname:
            
            t1_ss = 50
            t2_ss = 50
            t_ss = np.arange(-t1_ss, t2_ss, 1000/(self.sampling_rate_SS+1))
            self.ss_train = get_field_mat(self.mat,[self.SS_varname])
            clusters = np.unique(self.cluster_ID)
            cs_onset_downsample = (self.CS_onset/self.sampling_rate*1000).astype(int)
            ss_aligned = self.align_spikes(self.ss_train, cs_onset_downsample, t1_ss, t2_ss)
            offset = 0
            for i in range(self.n_clusters):
                # [iy, ix] = np.where(ss_aligned==1)
                [iy, ix] = np.where(ss_aligned[cluster_ID==clusters[i], :]==True)   
                self.canvas2.SS.plot(t_ss[ix], iy+offset, '.', c=self.colors[i])
                offset = offset + (cluster_ID==i+1).sum()
            
            
            ss_conv = gaussian_filter1d(ss_aligned, self.sigma * self.sampling_rate_SS/1000, order=0)*1000
            for i in range(self.n_clusters):
                self.canvas2.ax2.plot(t_ss, np.nanmean(ss_conv[cluster_ID==clusters[i], :], 0), c=self.colors[i], lw=2)
            
            self.canvas2.ax2.set_ylabel('SS firing rate [spikes/s]')
            self.canvas2.SS.set_xlabel('Time from CS onset [ms]')
            self.canvas2.SS.set_ylabel('CS')
            self.canvas2.SS.set_title('SS', loc='left')
            self.canvas2.SS.set_xlim([-t1_ss, t2_ss])
            
            
        # self.canvas2.onset.plot(time, embedding, 'tab:blue', lw=0.4)
        # self.canvas2.onset.set_xlabel('CS onset')
        # self.canvas2.simple_spikes.plot()
        # self.canvas2.simple_spikes.set_xlabel('Simple Spikes')
        self.canvas2.draw()

    def save_selected_cluster(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save selected cluster data", self.detect_fileName+'_clusters.mat',
                                                  "All Files (*);;MATLAB Files (*.mat)", options=options)

        if fileName:
            self.get_selected_clusters()
            sp.savemat(fileName, {'CS_onset': self.save_CS_onset,
                                  'CS_offset': self.save_CS_offset,
                                  'cluster_ID': self.save_cluster_ID,
                                  'embedding': self.save_embedding}, do_compression=True)
            print(fileName + ' saved')

    def get_selected_clusters(self):
        # selected_clusters = np.where(np.array(self.is_cluster_selected)==True)
        selected_clusters = self.clusters[np.where(np.array(self.is_cluster_selected)==True)]
        selected_indices = np.where(np.isin(np.array(self.cluster_ID), selected_clusters))
        print(selected_clusters)
        print(selected_indices)
        self.save_cluster_ID = self.cluster_ID[selected_indices]
        self.save_CS_onset = self.CS_onset[selected_indices]
        self.save_CS_offset = self.CS_offset[selected_indices]
        self.save_embedding = self.embedding[selected_indices, :].squeeze()



def create():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    main = Frame()
    main.show()
    sys.exit(app.exec_())


create()
