import unittest
import os
from ..vibration import file_to_numpy_array, SlicedFile, get_spectrogram, plot_spectrogram, VibrationData, train_vibration_classifier
import numpy

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')
logger = logging.getLogger(__name__)

# Make results (almost) reproducible
from numpy.random import seed
seed(1)

class test_vibration(unittest.TestCase):

    def test_SlicedFile(self):

        slicedFile = SlicedFile(TEST_DATA_PATH + '8kHz_vibration.m4a')
        slice_number = 0
        for slice in slicedFile.slices:
            self.assertTrue(slice.endswith(str(slice_number)))
            slice_number+=1
        self.assertEqual(slice_number, 179)

    def test_file_to_numpy_array(self):

        slicedFile = SlicedFile(TEST_DATA_PATH + '8kHz_vibration.m4a', slice_length=3)
        data, metadata = file_to_numpy_array(slicedFile.slices[0], format='wav')

        self.assertEqual(type(data), numpy.ndarray)
        self.assertEqual(metadata['frequency'], 8000)
        self.assertEqual(metadata['length'], 3)
        self.assertEqual(data.shape[0], 24000)

    def test_VibrationData(self):
        
        slicedFile = SlicedFile(TEST_DATA_PATH + '8kHz_vibration.m4a', slice_length=3)
        vibrationdData = VibrationData.from_file(slicedFile.slices[0], format='wav')
        
        self.assertEqual(type(vibrationdData.data), numpy.ndarray)
        self.assertEqual(vibrationdData.length, 3)
        self.assertEqual(vibrationdData.frequency, 8000)

    def test_spectrogram(self):
        slicedFile = SlicedFile(TEST_DATA_PATH + '8kHz_vibration.m4a', slice_length=1)
        vibrationdData = VibrationData.from_file(slicedFile.slices[34], format='wav')
        spectrogram = get_spectrogram(vibrationdData)
        self.assertEqual(type(spectrogram), numpy.ndarray)
        self.assertEqual(spectrogram.shape, (4001,1))
        #plot_spectrogram(spectrogram)

    def test_spectrogram_extended(self):
        vibrationdData = VibrationData.from_file(TEST_DATA_PATH + '8kHz_vibration.m4a')
        spectrogram = get_spectrogram(vibrationdData)
        self.assertEqual(type(spectrogram), numpy.ndarray)
        self.assertEqual(spectrogram.shape, (4001,179))
        #plot_spectrogram(spectrogram)

    def test_vibration_classifier(self):

        # Prepare test data
        slicedFile = SlicedFile(TEST_DATA_PATH + '8kHz_vibration.m4a', slice_length=1)
        
        # Preapre labels
        labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0]

        # Prepare data
        data=[]
        for slice in slicedFile.slices:
            vibrationdData = VibrationData.from_file(slice, format='wav')
            spectrogram = get_spectrogram(vibrationdData)
            spectrogram_4096 = numpy.append(spectrogram,numpy.zeros((95,)))
            spectrogram_64x64 = spectrogram_4096.reshape(64,64,1)
            data.append(spectrogram_64x64)

        # Prepare train and test data:
        train_data = []
        test_data  = []
        train_labels = []
        test_labels  = []
        for i in range(0, len(data)):
                       
            # Use 25% of data as test 
            if i%4 == 0:
                test_data.append(data[i])
                test_labels.append(labels[i])
            else:
                train_data.append(data[i])
                train_labels.append(labels[i])

        # Train classifier
        vibrationClassifier = train_vibration_classifier(train_data   = train_data,
                                                         train_labels = train_labels,
                                                         test_data    = test_data,
                                                         test_labels  = test_labels,
                                                         epochs       = 3)
        
        self.assertTrue(hasattr(vibrationClassifier,'model'))  
        self.assertTrue(vibrationClassifier.loss < 0.2)
        self.assertTrue(vibrationClassifier.accuracy > 0.95)

        # Ok, now test for precision, recall(sensitivity) and specificity.
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for i in range(0, len(data)):
            predicted_class = vibrationClassifier.predict(data[i])
            if predicted_class == labels[i]:
                if labels[i] == 0:
                    TN += 1
                else:
                    TP += 1
            else:
                if labels[i] == 0:
                    FN += 1
                else:
                    FP += 1

        #print('True positives: "{}"'.format(TP))
        #print('False positives: "{}"'.format(FP))
        #print('True negatives: "{}"'.format(TN))
        #print('False negatives: "{}"'.format(FN))
        
        precision   = float(TP)/float(TP+FP)
        recall      = float(TP)/float(TP+FN)
        specificity = float(TN)/float(TN+FP)   

        #print('Precision: "{}"'.format(precision))
        #print('Recall (sensitivity): "{}"'.format(recall))
        #print('Specificity: "{}"'.format(specificity))
        
        self.assertTrue(precision > 0.98)
        self.assertTrue(recall > 0.98)
        self.assertTrue(specificity > 0.97)
