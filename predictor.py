import pickle

import numpy as np
import pandas as pd

def predict(models, data):
    features_df = data.drop(['date', 'serial_number', 'model', 'failure'], axis=1)
    probabilities = models[0].predict_proba(features_df)
    for model in models[1:]:
       probabilities = np.add(probabilities, model.predict_proba(features_df)) 
    return pd.concat(
            [
                data[['serial_number','model']], pd.DataFrame({'failure_probability': probabilities[:, 1] / 2})
            ],
            axis=1
        )

def parse_data_json(data_json):
    return pd.read_json(data_json)

def get_json_from_source(json_source):
    json_file = open(json_source)
    data_json = json_file.read()
    json_file.close()
    return data_json

def get_models(models_source):
    models = []

    for model_name in models_source:
        model_file = open(model_name, 'rb')
        models.append(pickle.load(model_file))
        model_file.close()

    return models

def send_prediction_to_consumer(prediction_df):
    print(prediction_df.to_json(orient='records'))
    return True

def execute_by_db_update_event(json_source, models_source):
    data_json = get_json_from_source(json_source)
    models = get_models(models_source)
    data = parse_data_json(data_json)
    prediction = predict(models, data)
    #print('execute_by_db_update_event, prediction:\n', prediction)
    sucess = send_prediction_to_consumer(prediction)
    return prediction

    
#smart_short_0.json

if __name__ == '__main__':
    execute_by_db_update_event('smart_short_0.json', ['xgb_clf.pkl', 'lgbm_clf.pkl'])
