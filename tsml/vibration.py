import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy import signal as scipy_signal
from skimage import exposure

# Logging
import logging
logger = logging.getLogger(__name__)


#======================
#  Data Structures
#======================

class SlicedFile(object):

    def __init__(self, file_name, format=None, slice_length=1, data_path=None):
    
        if not format and '.' in file_name:
            format = file_name.split('.')[-1]
        elif format:
            pass
        else:
            raise Exception('Cannot detect file format and no explicit format set')
        
        if format not in ['m4a']:
            raise Exception('Only m4a files are supported for now')
        
        # Create the audio segment object
        audioSegment = AudioSegment.from_file(file_name, format=format)
        
        # Support vars
        audio_length_ms = len(audioSegment)
        slice_length_ms = slice_length*1000
    
        # Define a temporary directory if no data path is porvided
        if not data_path:
            data_path = tempfile.TemporaryDirectory().name
        
        # Store data path internally
        self.path = data_path

        # Log
        logger.debug('Using data path="{}"'.format(data_path))

        # Create directory path if does not exists
        if not os.path.isdir(data_path):
            logger.debug('Data path does not exist, creating...')
            os.makedirs(data_path)

        # Store slice paths internally
        self.slices = []
        
        # Slice
        count = 0 
        for _ in range(0, int(audio_length_ms//slice_length_ms)):
            
            # Define the start within the audioSegment
            slice_start_ms = slice_length_ms * count
            
            # Create the slice
            slice_audioSegment = audioSegment[slice_start_ms:slice_start_ms+slice_length_ms]
            slice_path = '{}/slice_{:04d}'.format(data_path, count)
            with open(slice_path, 'wb') as output_file_name:
                slice_audioSegment.export(output_file_name, format="wav")

            # Add slice
            self.slices.append(slice_path)
            
            count+=1
        
        # Log
        logger.debug('Sliced file in "{}" slices'.format(count))


class VibrationData(object):
    
    def __init__(self, data, length, frequency):
        '''Length in second, frequency in Hz'''
        self.data      = data 
        self.length    = length
        self.frequency = frequency

    @staticmethod
    def from_file(file_name, format=None):
        data, metadata = file_to_numpy_array(file_name, format=format)
        return VibrationData(data=data, length=metadata['length'], frequency=metadata['frequency'])


#======================
#  Utility functions
#======================

def file_to_numpy_array(file_name, format=None, norm=False):

    if not format and '.' in file_name:
        format = file_name.split('.')[-1]
    elif format:
        pass
    else:
        raise Exception('Cannot detect file format and no explicit format set (file_name="{}")'.format(file_name))

    metadata = {}
    
    if format in ['m4a', 'wav', 'wave']:
        
        # Audio formats: create the AudioSegment Object
        audio = AudioSegment.from_file(file_name, format=format)
    
        # Now create the numpy array
        numpy_array = np.asarray(audio.get_array_of_samples()).astype('float32')
        
        # Normalize?
        if norm:
            numpy_array = numpy_array/abs(numpy_array).max()

        # Add metadata
        metadata = {'frequency':audio.frame_rate, 'length': len(audio)/1000.0}
        
    else:
        raise Exception('Format "{}" is not supported"'.format(format))

    # Return
    return numpy_array, metadata


def plot_spectrogram(spectrogram):
    _, ax = plt.subplots(figsize=(7, 6), dpi=80)
    ax.imshow(spectrogram, cmap='viridis', aspect='auto') 
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()



#======================
#  Analysis functions
#======================

def get_spectrogram(vibrationdData, log=False, norm=False, overlap=0, resolution=1.0):

    # Define the number of samples per segment
    nperseg = round(vibrationdData.frequency*resolution)
    
    
    logger.debug('VibrationdData data shape: {}'.format(vibrationdData.data.shape))
    logger.debug('nperseg: {}'.format(nperseg))
   
    # Call scipy spectrogram function. x = time series of measurement values, fs = sampling frequency of the x time series.
    f, t, spectrogram = scipy_signal.spectrogram(x         = vibrationdData.data,
                                                 fs        = vibrationdData.frequency,
                                                 nperseg   = nperseg,
                                                 noverlap  = overlap,
                                                 return_onesided = True,
                                                 mode      = 'magnitude')
    # Return types:
    # f(ndarray: )Array of sample frequencies.
    # t(ndarray): Array of segment times.
    # spectrogram(ndarray): the spectrogram. By default, the last axis of spectrogram corresponds to the segment times.

    # Standard or logarithmic spectrogram?
    if log:
        spectrogram = np.log(spectrogram)

    # Log
    logger.debug('Spectrogram FFT window resolution (s) = "{}"'.format(resolution))
    logger.debug('Spectrogram FFT samples per window = "{}"'.format(nperseg))
    logger.debug('Spectrogram FFT overlap = "{}"'.format(overlap))
    logger.debug('Spectrogram shape: "{}"'.format(spectrogram.shape))
    logger.debug('Spectrogram t(s) = "{}"'.format(t))
    logger.debug('Spectrogram f(Hz) = "{}"'.format(f))

    # Return
    return spectrogram
