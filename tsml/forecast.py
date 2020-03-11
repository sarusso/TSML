from math import sqrt
from matplotlib import pyplot

# Pandas and Numpy
import pandas as pd
import numpy as np

# Keras and sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Logging
import logging
logger = logging.getLogger(__name__)

import sys
#======================
# Utility functions
#======================

def encode_data(data, encoder):
    '''Encode (compute diffs and normalize) data'''

    raw_values = data.values
    
    if encoder=='diff':
        # Compute diffs
        diff_series = []
        for i in range(1, len(raw_values)):
            value = raw_values[i] - raw_values[i - 1]
            diff_series.append(value)
        diff_series = pd.Series(diff_series)
        diff_values = diff_series.values
        first_pass_values = diff_values
        
    elif encoder=='24h':
        first_pass_values = []
        for i in range(0, len(raw_values)):
            hour = int(data.index[i] % (60*60*24) / 3600)
            first_pass_values.append([hour,raw_values[i]])

    elif encoder=='144m':
        first_pass_values = []
        for i in range(0, len(raw_values)):
            tenminute = int(data.index[i] % (60*6*24) / 3600)
            first_pass_values.append([tenminute,raw_values[i]])
            
    else:    
        first_pass_values = raw_values

    # Scale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))

    if encoder in ['24h', '144m']:
        scaled_values = scaler.fit_transform(first_pass_values)
        #scaled_values = first_pass_values
    else:
        scaled_values = scaler.fit_transform(first_pass_values.reshape(len(first_pass_values), 1))
        scaled_values = scaled_values.reshape(len(scaled_values), 1)

    # Hard debug
    #logger.debug('--- Encoding mapping ---') 
    #for i in range(1, len(raw_values)):
    #    logger.debug( '{}: {}-{} = {} -> {}'.format(i, raw_values[i], raw_values[i - 1], diff_series[i-1], scaled_values[i-1]))
    #logger.debug('------------------------')
    
    return scaler, scaled_values


def reshape_matrix_data_for_LSTM(X, features_per_timestep=1):
    '''Reshape matrix data for LSTM, as it must include the "features per timestep" dimension'''
    return  X.reshape(X.shape[0], features_per_timestep, int(X.shape[1]/features_per_timestep))


def reshape_array_data_for_LSTM(x, features_per_timestep=1):
    '''Reshape array data for LSTM, as it must include the "features per timestep" dimension '''
    return x.reshape(1, features_per_timestep, int(len(x)/features_per_timestep))


def split_data_vertically(data, cutoff):
    '''Split data vertically at a given cutoff'''
    X = data[:, 0:cutoff]
    y = data[:, cutoff:]
    return (X, y)


def create_train_and_test_data(data, window_datapoints, forecast_datapoints, cutoff, encoder):
    '''Create train and test datasets'''
    
    dataFrame = pd.DataFrame(data)
    cols      = []
    col_names = []

    # Window datapoints
    for j,i in enumerate(range(window_datapoints, 0, -1)):
        cols.append(dataFrame.shift(i))
        if encoder in ['24h', '144m']:
            col_names.append('window h({})'.format(j))
        col_names.append('window ({})'.format(j))
    
    # Forecast datapoints 
    for i in range(0, forecast_datapoints):
        shifted_cols = dataFrame.shift(-i)
        
        # Keep only the last col in case we are using more than one features
        shifted_cols = shifted_cols[shifted_cols.shape[1]-1]

        # Append
        cols.append(shifted_cols)
        col_names.append('forecast ({})'.format(i))
                    
    # Merge
    window_and_forecast_datapoints = pd.concat(cols, axis=1)
    window_and_forecast_datapoints.columns = col_names

    # Drop NaN introduced due to shifting
    window_and_forecast_datapoints.dropna(inplace=True)
    
    #print('=============================')
    #print(window_and_forecast_datapoints)
    #print('=============================')

    # Split data in train and test sets
    train = window_and_forecast_datapoints.values[0:-cutoff]
    test  = window_and_forecast_datapoints.values[-cutoff:]
    
    # Return
    return train, test


def decode_values(seed, scaler, encoded_values, decoder):
    '''Decode encoded values'''
    decoded_values = []
    
    if decoder == 'diff':
        prev_decoded_value = seed
        for encoded_value in encoded_values:
            decoded_value = prev_decoded_value + scaler.inverse_transform(encoded_value)[0][0]
            decoded_values.append(decoded_value)
            prev_decoded_value = decoded_value
    
    elif decoder in ['24h', '144m']:
        for encoded_value in encoded_values:
            encoded_value = [[0,encoded_value]]
            decoded_value = scaler.inverse_transform(encoded_value)[0][1] # Keep only the last
            decoded_values.append(decoded_value)

    else:
        for encoded_value in encoded_values:
            decoded_value = scaler.inverse_transform(encoded_value)[0][0]
            decoded_values.append(decoded_value)
    
    return decoded_values
  
        
def evaluate_model_on_encoded_data(model, encoded_test_data_in, encoded_test_data_out, initial_seed, scaler, encoder, features_per_timestep, plot=False):

    # Support vars    
    prev_seed    = None
    
    # Forecast on test data
    decoded_test_data_out_forecasted = []
    decoded_test_data_out            = []
    for i in range(len(encoded_test_data_in)):
        
        # Make the forecast
        x = encoded_test_data_in[i]
        x = reshape_array_data_for_LSTM(x, features_per_timestep)
        encoded_forecast_values = list(model.predict(x)[0])

        # Set the seed to decode the forecast
        if i != 0:
            # Use the last (decoded) value of the test data input array for this row
            this_seed = decode_values(seed=prev_seed, scaler=scaler, encoded_values=[encoded_test_data_in[i][-1]], decoder=encoder)[0]
        else:
            this_seed = initial_seed
        # Save this seed to be used on the next loop
        prev_seed = this_seed
        
        # Ok, now decode the forecast and the values (that we will use to compute accuracy/error):
        decoded_forecast_values = decode_values(seed=this_seed, scaler=scaler, encoded_values=encoded_forecast_values, decoder=encoder)
        decoded_values          = decode_values(seed=this_seed, scaler=scaler, encoded_values=encoded_test_data_out[i], decoder=encoder)

        # Append
        decoded_test_data_out_forecasted.append(decoded_forecast_values)
        decoded_test_data_out.append(decoded_values)

    # Example encoded_test_data_forecasts: [[-0.02597164, 0.019228103, -0.0863501], [-0.04299922, -0.06884324, 0.019511309], [-0.025245361, 0.0050590625, -0.044009812], [0.004433021, -0.07524977, -0.018381473], [-0.035056613, 0.022560129, -0.07090938], [-0.02765443, -0.07488325, 0.018706568], [-0.011006074, -0.0008233283, -0.06463229], [-0.047710918, -0.008823719, -0.008209936], [0.021406159, -0.08290595, -0.023111574], [-0.053629197, 0.06815379, -0.078827314]]
    # Example decoded_test_data_forecasts: [[369.9025218963623, 408.377844619751, 421.4563291549683], [363.20653839111327, 380.49629707336425, 419.03974266052245], [468.1772274017334, 503.2441867828369, 526.5076259613037], [350.81636276245115, 366.56503047943113, 395.9933675765991], [464.7171314239502, 503.9939678192139, 520.7867153167724], [428.49772720336915, 444.33456020355226, 482.68442516326905], [468.60248985290525, 502.2544368743896, 520.5571388244629], [597.8731384277344, 629.6005935668945, 661.4756927490234], [446.59925231933596, 460.50622711181643, 488.79673728942873], [702.9494953155518, 753.1938877105713, 768.0819778442383]]

    # Compute Root Mean Square Error
    rmse_values = []
    for i in range(len(decoded_test_data_out_forecasted[0])):
        actual    = [row[i] for row in decoded_test_data_out]
        predicted = [forecast[i] for forecast in decoded_test_data_out_forecasted]
        rmse      = sqrt(mean_squared_error(actual, predicted))
        rmse_values.append(rmse)
        logger.debug('RMSE @ forecasting step #{}: {}'.format(i+1, rmse))

    if plot:
        # Plot the test data series
        decoded_test_data_out_series = [row[0] for row in decoded_test_data_out]
        pyplot.plot(decoded_test_data_out_series)
        
        # Plot the forecasts, with the right offset
        for i in range(len(decoded_test_data_out_forecasted)):
            offset_start = i
            offset_end   = i + len(decoded_test_data_out_forecasted[i]) + 1
            xaxis = [x for x in range(offset_start, offset_end)]
            yaxis = [decoded_test_data_out_series[offset_start]] + decoded_test_data_out_forecasted[i]
            pyplot.plot(xaxis, yaxis, color='red')
        pyplot.show()   

    return rmse_values


#======================
#  Models
#======================

class Forecaster(object):

    def __init__(self, window_datapoints=3, forecast_datapoints=1, test_ratio=0.2, encoder='diff'):
        '''Initialize a Forecaster'''
        self.window_datapoints   = window_datapoints
        self.forecast_datapoints = forecast_datapoints
        self.test_ratio          = test_ratio
        self.encoder             = encoder
        self.model               = None

    def train(self, dataTimeSlotSerie=None, pd_dataframe=None, epochs=10, neurons=100, plot=False, verbose=False, model_type=1):
        '''Train on an input dataTimeSlotSeries'''

        if dataTimeSlotSerie:
            # Convert the training series to a pd dataframe
            #pd_dataframe = ...
            raise NotImplementedError('Pasing dataTimeSlotSeries is not yet implemented')
        
        if pd_dataframe is None:
            raise Exception('Missing pd_dataframe argument')
        

        # configure
        window_datapoints = self.window_datapoints
        forecast_datapoints = self.forecast_datapoints
        if self.encoder == 'diff':
            encoded_window_datapoints = window_datapoints-1
        else:
            encoded_window_datapoints = window_datapoints

        # Features  are the amount of features in every time step. In numerical time series data this is each timestamp
        # has one feature (the value), but if using other values
        features_per_timestep = 2 if self.encoder == '24h' else 1
        
        # Compute test datapoints based on the test_ratio
        test_datapoints = int(round(len(pd_dataframe)*self.test_ratio))
        logger.debug('Using "{}" datapoints for testing'.format(test_datapoints))

        # Encode data
        scaler, encoded_data = encode_data(data=pd_dataframe, encoder=self.encoder)

        # Create train and test data sets
        encoded_train_data, encoded_test_data = create_train_and_test_data(encoded_data, encoded_window_datapoints, forecast_datapoints, test_datapoints, encoder=self.encoder)

        # Split train and test data into input and output
        encoded_train_data_in, encoded_train_data_out = split_data_vertically(data=encoded_train_data, cutoff=encoded_window_datapoints*features_per_timestep)
        encoded_test_data_in,  encoded_test_data_out  = split_data_vertically(data=encoded_test_data,  cutoff=encoded_window_datapoints*features_per_timestep)

        # Timesteps are basically the "lookback" memory of the network.
        timesteps = encoded_window_datapoints

        if not self.model:
            # Neural network model topology
            logger.debug('Using model type="{}"'.format(model_type))
        
            if model_type==1:
                model = Sequential()
                model.add(LSTM(neurons, input_shape=(features_per_timestep, timesteps)))
                model.add(Dense(encoded_train_data_out.shape[1]))
                model.compile(loss='mean_squared_error', optimizer='adam')
                
            elif model_type==2:
                model = Sequential()
                model.add(LSTM(200, activation='relu', input_shape=(features_per_timestep, timesteps)))
                model.add(Dropout(0.15))
                model.add(Dense(encoded_train_data_out.shape[1]))
                model.compile(optimizer='adam', loss='mse')
    
            elif model_type==3:
                model = Sequential()
                model.add(LSTM(32, return_sequences=True, input_shape=(features_per_timestep, timesteps)))
                model.add(Dropout(0.05))
                model.add(LSTM(16, activation='relu'))
                model.add(Dropout(0.05))
                model.add(Dense(encoded_train_data_out.shape[1]))
                model.compile(optimizer=optimizers.RMSprop(clipvalue=1.0), loss='mae')
            self.model = model

        # Train the model
        self.model.fit(reshape_matrix_data_for_LSTM(encoded_train_data_in, features_per_timestep=features_per_timestep), encoded_train_data_out, epochs=epochs, verbose=verbose, shuffle=False)

        # Evaluate the model (compute RMSE). The initial seed allows to decode data and compute the error on actual numbers rather that on an encoded, neural network-friendly representation.
        if not test_datapoints:
            logger.info('Will not evaluate model as no test data at all')
            evaluate = False
        else: 
            if self.encoder == 'diff':
                try:
                    initial_seed = pd_dataframe[len(pd_dataframe) - (test_datapoints + (encoded_window_datapoints))]
                except KeyError:
                    logger.warning('Cannot evaluate model as no enough test data')
                    evaluate = False
            else:
                initial_seed = None

        if evaluate:
            rmse_values = evaluate_model_on_encoded_data(model, encoded_test_data_in, encoded_test_data_out, initial_seed, scaler, encoder=self.encoder, features_per_timestep=features_per_timestep, plot=plot)
        else:
            rmse_values = None

        # Save model and parameters internally
        self.scaler = scaler
        self.window_datapoints = window_datapoints
        self.forecast_datapoints = forecast_datapoints

        return rmse_values
        
    def forecast(self, series):
        '''Forecast the next n values based on an input dataTimeSlotSeries'''
        
        if not self.model:
            raise('Sorry, this forecaster is not trained yet')

        # Encode
        encoded_series=[]
        prev_datapoint = None
        for datapoint in series:
            if prev_datapoint is not None:
                encoded_series.append(self.scaler.transform(datapoint-prev_datapoint))
            prev_datapoint = datapoint
        logger.debug('Encoded series: "{}"'.format(encoded_series))

        # Predict using encoded series
        x = np.array(encoded_series)
        encoded_forecasts = self.model.predict(reshape_array_data_for_LSTM(x))[0]
        logger.debug('Encoded forecasts: "{}"'.format(encoded_forecasts))
        
        # Decode forecasts
        decoded_forecasts=[]
        prev_decoded_forecast = None
        for encoded_forecast in encoded_forecasts:

            if prev_decoded_forecast is None:
                forecast = series[-1] + float(self.scaler.inverse_transform(encoded_forecast))
            else:
                forecast = prev_decoded_forecast + float(self.scaler.inverse_transform(encoded_forecast))
            
            decoded_forecasts.append(forecast)
            prev_decoded_forecast =  forecast

        logger.debug('Decoded forecasts: "{}"'.format(decoded_forecasts))
        return decoded_forecasts
