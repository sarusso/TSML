import unittest
import os
from ..vibration import file_to_numpy_array, SlicedFile, get_spectrogram, plot_spectrogram, VibrationData
import numpy

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

# Setup logging
import logging
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)

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
