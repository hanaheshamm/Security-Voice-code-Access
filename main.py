import glob
import math

import qdarkstyle
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import os
import numpy as np
import sounddevice as sd
import wavio as wv
from os import path
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.feature as lib

from joblib import dump, load

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "main.ui"))
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=400, height=300, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=(0.0, 0.0, 0.0, 0.0), tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        if parent:
            layout = QVBoxLayout(parent)
            layout.addWidget(self)
            layout.setStretch(0, 1)

        self.axes.clear()
        self.axes.set_facecolor('none')  # Set axes facecolor to transparent
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Frequency (Hz)')
        self.axes.set_title('Spectrogram')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.speaker = None
        self.previous_recordings = None
        self.setupUi(self)
        self.reference_audio_paths = ['recording_abdalla_1.wav', 'recording_abdalla_u1.wav', 'recording_abdalla_g8.wav']
        self.test_audio_path = 'recording.wav'
        self.mode = 0  # initialize sentence mode
        self.radioButton_3.setChecked(True)
        self.handle_buttons()
        self.similarity_score_arr = [[0] * 4]
        self.mpl_placeholder = self.findChild(QWidget, 'mplWidget')
        layout = QVBoxLayout(self.mpl_placeholder)
        self.sc = MplCanvas(self.mpl_placeholder, width=5, height=4, dpi=100)
        layout.addWidget(self.sc)
        layout.removeWidget(self.mpl_placeholder)
        self.features_list = []
        self.labels = []

    def extract_mfcc(self, file_path):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        speaker_features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

        return speaker_features

    def mode_changed(self):
        if self.radioButton_3.isChecked():  # radioButton_3 sentence mode
            self.mode = 0
        elif self.radioButton_2.isChecked():  # radioButton_2 fingerprint mode
            self.mode = 1
        print(self.mode)


    def give_access(self):
        self.person = self.comboBox.currentText()
        listView = self.findChild(QListView, 'access_list')
        model = listView.model()

        if model is None:
            model_list = QStringListModel([self.person])
            self.comboBox_2.addItem(self.person)
            listView.setModel(model_list)
            self.items = [self.person]
        else:
            # self.items = model.stringList()
            if self.person not in self.items:
                self.items.append(self.person)
                model.setStringList(self.items)
                # self.comboBox_2.addItem(self.person)
                if self.person not in [self.comboBox_2.itemText(i) for i in range(self.comboBox_2.count())]:
                    self.comboBox_2.addItem(self.person)

    def extract_features(self,audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=int(sr*0.02), hop_length=int(sr*0.01))
        return mfcc.T

    def detect_phrase(self,reference_paths, test_path, threshold=0.1):
        # Extract features from the test audio
        test_features = self.extract_features(test_path)

        # Initialize variables to store the best (minimum) distance
        min_distance = float('inf')
        best_ref_path = None
        distances = []
        probabilities = []

        # Loop through each reference file and calculate DTW distance
        for ref_path in reference_paths:
            ref_features = self.extract_features(ref_path)
            distance, _ = fastdtw(ref_features, test_features, dist=euclidean)
            normalized_distance = distance / len(test_features)
            distances.append(int(normalized_distance))

            # Update minimum distance and corresponding reference path
            if normalized_distance < min_distance:
                min_distance = normalized_distance
                best_ref_path = ref_path
        mean_distance = sum(distances) / len(distances)     
        squared_diff = [(distance - mean_distance)**2 for distance in distances]
        mean_squared_diff = sum(squared_diff) / len(squared_diff)
        # Calculate the standard deviation
        std_dev = math.sqrt(mean_squared_diff)
        # probability = math.exp(-0.5 * ((min_distance - mean_distance) / std_dev) ** 2)
        probabilities = [(math.exp(-0.5 * ((min - mean_distance) / std_dev) ** 2)) for min in distances]
        # print(f"min distance {best_ref_path}: {normalized_distance} , probability : {probability}")
        sum_probabilities = sum(probabilities)
        normalized_probabilities = [prob / sum_probabilities for prob in probabilities]
        normalized_probabilities = [1-prob for prob in normalized_probabilities]
        # self.add_data_to_model(normalized_probabilities)
        print(normalized_probabilities)
        print("(max(normalized_probabilities))" , max(normalized_probabilities))
        # self.label_5.setText(normalized_probabilities[0])
        # ['open_middle_door.wav', 'unlock_the_gate.wav', 'grant_me_access.wav']
        if normalized_probabilities:
            first_probability = normalized_probabilities[0]
            second_probability = normalized_probabilities[1]
            third_probability = normalized_probabilities[2]
            self.label_11.setText(f"open middle door: {first_probability:.4f}")
            self.label_12.setText(f"unlock the gate: {second_probability:.4f}")
            self.label_13.setText(f"grant me access: {third_probability:.4f}")

    def remove_access(self):
        selected_item = self.comboBox_2.currentText()
        listView = self.findChild(QListView, 'access_list')
        model = listView.model()

        if model is not None:
            # items = model.stringList()
            if selected_item in self.items:
                self.items.remove(selected_item)
                model.setStringList(self.items)


    def predict_all(self, file_path):
        new_recording_features = self.extract_mfcc(file_path)
        model = load('svm_model.joblib')
        prediction_score = model.predict_proba([new_recording_features])
        print("prediction ", prediction_score)
        # if prediction_score:
        first_probability = prediction_score[0][0]
        second_probability = prediction_score[0][1]
        third_probability = prediction_score[0][2]
        fourth_probability = prediction_score[0][3]
        fifth_probability = prediction_score[0][4]
        six_probability = prediction_score[0][5]
        seventh_probability = prediction_score[0][6]
        last_probability = prediction_score[0][7]

        self.label_22.setText(f"Abdallah: {first_probability:.4f}")
        self.label_16.setText(f"omar: {second_probability:.4f}")
        self.label_14.setText(f"Mayar: {third_probability:.4f}")
        self.label_19.setText(f"Hana: {fourth_probability:.4f}")
        self.label_17.setText(f"omda: {fifth_probability:.4f}")
        self.label_18.setText(f"nabil: {six_probability:.4f}")
        self.label_20.setText(f"merna: {seventh_probability:.4f}")
        self.label_23.setText(f"khaled: {last_probability:.4f}")




        return

    def record_audio(self, duration, filename="recording.wav"):
        """
        Records audio for `duration` seconds and saves it to `filename`.
        """

        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wv.write(filename, recording, fs, sampwidth=2)
        self.audio_read(filename)

        self.predict_all(filename)
        self.detect_phrase(self.reference_audio_paths, self.test_audio_path)

    def audio_read(self, filename):
        data, sf = librosa.load(filename)
        print("data: ", data)
        self.sc.axes.specgram(data, Fs=sf)
        self.sc.draw()

    def handle_buttons(self):
        self.recordButton.clicked.connect(lambda: self.record_audio(duration=3, filename="recording.wav"))
        self.comboBox.currentIndexChanged.connect(self.give_access)
        self.comboBox_2.currentIndexChanged.connect(self.remove_access)
        self.radioButton_3.clicked.connect(self.mode_changed)
        self.radioButton_2.clicked.connect(self.mode_changed)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
