#!/usr/bin/env python

import os, warnings, sys
import pprint
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd

import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.preprocessing.preprocessing_main as data_preprocessing

import algorithm.utils as utils
import algorithm.model.forecaster as forecaster
import algorithm.model.forecaster_pipeline as fcstr_pipeline
from algorithm.utils import get_model_config

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):      
    # set seeds
    utils.set_seeds(100)       
    # get preprocessing parameters
    pp_params = pp_utils.get_preprocess_params(data_schema)     
    # preprocess data (includes history and special_events if any)
    processed_history, data_preprocessor = preprocess_data(data, pp_params)    
    # print('processed_history \n', processed_history.head())    
    # Train model  - this will also do the preprocessing specific to the model
    model_artifacts = train_model(processed_history, pp_params, hyper_params) 
    # merge the two artifacts
    model_artifacts = {"data_preprocessor": data_preprocessor, **model_artifacts}
    return model_artifacts, data_preprocessor

    

def preprocess_data(data, pp_params):  
    print("Preprocessing data...") 
    history = data["history"] ; sp_events = data["sp_events"]  
    # history.at[80, 'Passengers'] = 'missing'
    
    data_preprocessor = data_preprocessing.DataPreprocessor(
            pp_params,  
            has_sp_events=True if sp_events is not None else False
        )    
    history = data_preprocessor.fit_transform(history=history, sp_events=sp_events) 
     
    # del history["__exog__missing__"]
    return history, data_preprocessor  



def train_model(processed_history, pp_params, hyper_params):        
            
    # determine the history length to use. needs to be updated if given shorter history 
    pp_params["hist_len_multiple_of_fcst_len"] = get_history_len_multiplier(
            processed_history, pp_params, hyper_params )    
    
    # model-specific preprocessing pipelines for training and prediction 
    model_train_pipeline, model_pred_pipeline = \
        fcstr_pipeline.get_forecaster_preprocess_pipelines(pp_params, model_cfg)
        
    train_data = model_train_pipeline.fit_transform(processed_history)      
    
    # --------------------------------------------------------------------------
    # get model hyper-parameters parameters      
    data_based_params = forecaster.get_data_based_model_params(train_data)
    model_params = { **data_based_params, **hyper_params }
    # print(model_params) ; sys.exit()
    # --------------------------------------------------------------------------
    # Create and train model       
    print('Fitting model ...')  
    model = forecaster.Forecaster(  **model_params )  
    # model.summary()  ; sys.exit()    
    train_history = model.fit(
        train_data = train_data,
        validation_split = model_cfg["valid_split"],
        max_epochs = 1000,
        verbose = 0, 
    )  
    # --------------------------------------------------------------------------
    # test predictions
    # preds =  model.predict(train_X, train_E)       
    # preds = preds.flatten(); actuals = train_y.flatten()    
    # r2 = r2_score(actuals,preds) ; print("r2", r2)  ; sys.exit()  
    # df = pd.DataFrame(np.vstack([actuals, preds]).T, columns=['Actuals', 'Preds'])
    # df.to_csv("predictions.csv", index=False)
    # --------------------------------------------------------------------------    
    train_artifacts = {
        "train_data": processed_history,
        "model_train_pipeline": model_train_pipeline, 
        "model_pred_pipeline": model_pred_pipeline, 
        "model": model,
        "train_history": train_history
    }
        
    return train_artifacts
    

def get_history_len_multiplier(data, pp_params, hyper_params): 
    '''
    update history_len_multiplier. If given history isnt enough, then make history_len_multiplier smaller. 
    '''
    num_epochs = data[pp_params['epoch_field']].nunique()
    max_multiples = num_epochs // int(pp_params['forecast_horizon_length']) - 1
    hist_len_multiple_of_fcst_len = min(hyper_params["max_hist_len_multiple_of_fcst_len"], max_multiples) 
    return hist_len_multiple_of_fcst_len


def save_training_artifacts(model_artifacts, data_preprocessor, model_artifacts_path):   
    # save data_preprocessor
    data_preprocessing.save_data_preprocessor(data_preprocessor, model_artifacts_path)      
    # save the model artifacts
    forecaster.save_model_artifacts(model_artifacts, model_artifacts_path)