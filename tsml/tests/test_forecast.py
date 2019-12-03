import unittest
import os
from ..forecast import Forecaster
import numpy
from pandas import datetime
from pandas import read_csv 

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'
EXTENDED_TESTING = os.environ.get('EXTENDED_TESTING', False)

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL') if os.environ.get('LOGLEVEL') else 'CRITICAL')
logger = logging.getLogger(__name__)

# Make results (almost) reproducible
from numpy.random import seed
seed(1)

class test_forecast(unittest.TestCase):

    def test_train_and_forecast(self):

        # Load dataset
        from pandas import read_csv 
        def parser(x):
            return datetime.strptime('190'+x, '%Y-%m')
        pd_dataframe = read_csv(TEST_DATA_PATH+'shampoo_sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        
        # Instantiate a Forecaster
        forecaster = Forecaster(window_datapoints=4, forecast_datapoints=3, test_ratio=0.28)
        
        # Train and evaluate the forecaster
        rmse_values = forecaster.train(pd_dataframe=pd_dataframe, epochs=10, neurons=100, plot=False)
        
        # Check RMSEs
        self.assertAlmostEqual(rmse_values[0], 140.3, places=1)
        self.assertAlmostEqual(rmse_values[1], 85.8, places=1)
        self.assertAlmostEqual(rmse_values[2], 110.2, places=1)

        # Ok, test forecast now 
        series_to_forecast = numpy.array([407.6, 682.0, 475.3, 581.3])

        forecast = forecaster.forecast(series_to_forecast)
        expected_forecast = [612.8421161651611, 660.1734531402587, 663.7969360828399]
        
        for i in range(len(expected_forecast)):
            self.assertAlmostEqual(forecast[i], expected_forecast[i], places=1)


    def test_train_and_forecast_temperature(self):


        if str(EXTENDED_TESTING).lower() == 'true':

            # Load data
            pd_dataframe = read_csv(TEST_DATA_PATH+'temp_long_1h.csv', index_col='epoch', header=0)
            clean_pd_dataframe = pd_dataframe['temp']
            
            # Initialize test parameters
            model_types = [1,2,3]
            windows     = [3,6,12,24]
            
            # Initalize results
            results={}
            
            # Test with 24h encoder
            encoder ='24h'
            for model_type in model_types:
                for window in windows:

                    # Instantiate and evaluate forecaster
                    forecaster = Forecaster(window_datapoints=window, forecast_datapoints=1, test_ratio=0.3, encoder=encoder)
                    rmse_values = forecaster.train(pd_dataframe=clean_pd_dataframe, epochs=2, model_type=model_type, plot=False, verbose=True)
    
                    # Append results
                    print('\nEncoder={}, model_type={}, window={}: {}\n'.format(encoder, model_type, window, rmse_values))
                    results['Encoder={}, model_type={}, window={}'.format(encoder, model_type, window)] = rmse_values
                        
            
            # Test with other encoders
            clean_pd_dataframe.reset_index(drop=True, inplace=True)
            for encoder in [None, 'diff',]:
                for model_type in model_types:
                    for window in windows:

                        # Instantiate and evaluate forecaster
                        forecaster = Forecaster(window_datapoints=window, forecast_datapoints=1, test_ratio=0.3, encoder=encoder)
                        rmse_values = forecaster.train(pd_dataframe=clean_pd_dataframe, epochs=2, model_type=model_type, plot=False, verbose=True)
        
                        # Append results
                        print('\nEncoder={}, model_type={}, window={}: {}\n'.format(encoder, model_type, window, rmse_values))
                        results['Encoder={}, model_type={}, window={}'.format(encoder, model_type, window)] = rmse_values
                
            # Check all results
            for item in results:
                rmse = results[item][0]
                print('{}: {}'.format(item,rmse))
                self.assertTrue(rmse<0.8)
            
        else:

            # Load data
            pd_dataframe = read_csv(TEST_DATA_PATH+'temp_long_1h.csv', index_col='epoch', header=0)
            clean_pd_dataframe = pd_dataframe['temp']
            clean_pd_dataframe.reset_index(drop=True, inplace=True)

            # Instantiate and evaluate forecaster
            forecaster = Forecaster(window_datapoints=24, forecast_datapoints=1, test_ratio=0.3, encoder='diff')
            rmse_values = forecaster.train(pd_dataframe=clean_pd_dataframe, epochs=3, model_type=3, plot=False, verbose=False)

            # Check results
            self.assertTrue(rmse_values[0]<0.28)

          
