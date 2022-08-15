
from email.mime import base
import numpy as np, pandas as pd
import math
import joblib
import sys
import os, warnings
os.environ['PYTHONHASHSEED']=str(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 
from sklearn.metrics import mean_squared_error


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

from algorithm.model.nbeats import NBeatsNet

MODEL_NAME = "Nbeats_Forecaster_Base"


model_pred_pipeline_fname = "model_pred_pipeline.save"
model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
train_history_fname = "train_history.json"
train_data_fname = "train_data.csv"
train_data_fname_zip = "train_data.zip"


COST_THRESHOLD = float('inf')

class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get('loss')
        if(loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val)):
            print("Cost is inf, so stopping training!!")
            self.model.stop_training = True



def get_data_based_model_params(train_data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''     
    return {
        "input_dim": train_data['X'].shape[2], 
        "exo_dim": train_data['E'].shape[2] if train_data['E'] is not None else 0, 
        "backcast_length": train_data['X'].shape[1],
        "forecast_length": train_data['y'].shape[1]
        }


def get_patience_factor(N): 
    # magic number - just picked through trial and error
    patience = int(35 - math.log(N, 1.5))
    return patience


class Forecaster(): 
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    MIN_VALID_SIZE = 10
    BATCH_SIZE = 64

    def __init__(self,
                 input_dim,
                 exo_dim,
                 backcast_length,
                 forecast_length,
                 thetas_dim,
                 hidden_layer_units,
                 share_weights_in_stack,
                 learning_rate=1e-4,
                 **kwargs
                 ):

        self.input_dim = input_dim
        self.exo_dim = exo_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.thetas_dim = thetas_dim
        self.hidden_layer_units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.loss='mse'        
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.model.compile_model(self.loss, self.learning_rate)


    def build_model(self, ):   
             
        num_generic_stacks=2
        # thetas_dim_per_stack=16 * (1 + self.exo_dim)
        thetas_dim_per_stack=self.thetas_dim
        nb_blocks_per_stack=2
        # share_weights_in_stack=False
        share_weights_in_stack=self.share_weights_in_stack
        # hidden_layer_units=8 * (1 + self.exo_dim)
        hidden_layer_units=self.hidden_layer_units
        nb_harmonics=None
        
        stack_types = []; thetas_dim = []
        for _ in range(num_generic_stacks):
            stack_types.append(self.GENERIC_BLOCK)
            thetas_dim.append(thetas_dim_per_stack)

        model = NBeatsNet(
            input_dim = self.input_dim,
            exo_dim = self.exo_dim,
            backcast_length = self.backcast_length,
            forecast_length = self.forecast_length,
            stack_types = stack_types,
            nb_blocks_per_stack = nb_blocks_per_stack,
            thetas_dim = thetas_dim,
            share_weights_in_stack = share_weights_in_stack,
            hidden_layer_units = hidden_layer_units,
            nb_harmonics = nb_harmonics
        )
        return model 
        
    
    
    def fit(self, train_data, validation_split=0.1, verbose=0, max_epochs=1000):

        train_X, train_y, train_E = train_data['X'], train_data['y'], train_data['E']
        # print("X/y shapes", train_X.shape, train_y.shape)
        # if train_E is not None: print("E shape", train_E.shape )
        if train_X.shape[0] < 100:  validation_split = None
        
        patience = get_patience_factor(train_X.shape[0])

        loss_to_monitor = 'loss' if validation_split is None else 'val_loss'
        early_stop_callback = EarlyStopping(monitor=loss_to_monitor, 
                                            min_delta = 1e-4, patience=patience) 

        learning_rate_reduction = ReduceLROnPlateau(monitor=loss_to_monitor, 
                                            patience=3, 
                                            factor=0.5, 
                                            min_lr=1e-7)

        inf_cost_callback = InfCostStopCallback()
        
        history = self.model.fit(
            x=[train_X, train_E] if train_E is not None else train_X, 
            y=train_y, 
            validation_split = validation_split,
            shuffle=False,
            verbose=verbose,
            epochs=max_epochs,  
            callbacks=[early_stop_callback, inf_cost_callback],
            batch_size=self.BATCH_SIZE)
        
        return history
    

    def predict(self, data):        
        X, E, y = data['X'], data['E'], data['y']  
        # print(X.shape, E.shape, y.shape)  ; sys.exit()
        preds = self.model.predict( x=[X, E] if E is not None else X, verbose=False )
        return preds
    

    def evaluate(self, data): 
        preds = self.predict(data)
        mse = mean_squared_error(data['y'].flatten(), preds.flatten())
        return mse
        


    def summary(self): 
        self.model.summary()
        
    
    def save(self, model_path): 
        model_params = {
            "input_dim": self.input_dim,
            "exo_dim": self.exo_dim,
            "backcast_length": self.backcast_length,
            "forecast_length": self.forecast_length,
            "thetas_dim": self.thetas_dim,
            "hidden_layer_units": self.hidden_layer_units,
            "share_weights_in_stack": self.share_weights_in_stack,
            "learning_rate": self.learning_rate
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))
        self.model.save_weights(os.path.join(model_path, model_wts_fname))
    
    @classmethod
    def load(cls, model_path): 
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        forecaster = cls(**model_params)
        forecaster.model.load_weights(os.path.join(model_path, model_wts_fname)).expect_partial()
        return forecaster


      

def save_model_artifacts(train_artifacts, model_artifacts_path): 
    # save model
    save_model(train_artifacts["model"], model_artifacts_path)
    # save model-specific prediction pipeline
    save_model_pred_pipeline(train_artifacts["model_pred_pipeline"], model_artifacts_path)
    # save traiing history
    save_training_history(train_artifacts["train_history"], model_artifacts_path)
    # save training data
    save_training_data(train_artifacts["train_data"], model_artifacts_path)

def save_model(model, model_path):    
    model.save(model_path) 
    

def save_model_pred_pipeline(pipeline, model_path): 
    joblib.dump(pipeline, os.path.join(model_path, model_pred_pipeline_fname))


def save_training_history(history, f_path): 
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = os.path.join(f_path, train_history_fname)
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)


def save_training_data(train_data, model_artifacts_path):
    compression_opts = { "method":'zip',  "archive_name": train_data_fname }      
    train_data.to_csv(os.path.join(model_artifacts_path, train_data_fname_zip), 
            index=False,  compression=compression_opts)


