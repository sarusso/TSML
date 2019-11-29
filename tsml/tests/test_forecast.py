import unittest
import os
from ..forecast import Forecaster
import numpy
from pandas import datetime

# Set test data path
TEST_DATA_PATH = '/'.join(os.path.realpath(__file__).split('/')[0:-1]) + '/test_data/'

# Setup logging
import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'CRITICAL'))
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
 