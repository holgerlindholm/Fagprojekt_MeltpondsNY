import pandas as pd
import numpy as np
import h5py
from tensorflow.keras.models import load_model
import os

current_dir = os.getcwd()

def predict_depth(data, num_p):

    # Choose correct model depending on number of neighbors
    if num_p == 9:
        model_path = os.path.join(current_dir, '7383_9P.h5')
    elif num_p == 1:
        model_path = os.path.join(current_dir, '6371_1P.h5')

    #load the keras model
    model = load_model(model_path)

    
    scaler_X_max = [
    7588.0, 9239.0, 9743.0, 10240.0, 7788.0, 9248.0, 9695.0, 9928.0,
    7944.0, 9873.0, 10432.0, 10544.0, 7772.0, 9552.0, 10192.0, 10448.0,
    7828.0, 9152.0, 9464.0, 9592.0, 7772.0, 9552.0, 10192.0, 10448.0,
    7944.0, 9873.0, 10432.0, 10544.0, 7584.0, 9128.0, 9808.0, 9928.0,
    7600.0, 9032.0, 9568.0, 9792.0]

    scaler_X_min = [
    1008.0, 1471.0, 2736.0, 3073.0, 1024.0, 1491.0, 2808.0, 3132.0,
    1048.0, 1463.0, 2747.0, 2935.0, 1048.0, 1463.0, 2839.0, 3129.0,
    1008.0, 1471.0, 2736.0, 3073.0, 1040.0, 1478.0, 2736.0, 3073.0,
    1052.0, 1461.0, 2747.0, 2935.0, 1024.0, 1491.0, 2755.0, 3010.0,
    1048.0, 1463.0, 2816.0, 3175.0]
    scaler_y_max = [-0.172625433312264]
    scaler_y_min = [-3.63776361780149]
    

    if num_p == 9:
        df_normalized = normalize_data(df, scaler_X_max, scaler_X_min)
    elif num_p == 1:
        df_normalized = normalize_data(df, scaler_X_max[:4], scaler_X_min[:4])

    # Predict using the model
    predictions = model.predict(df_normalized)

    # Denormalize the predictions
    predictions_denormalized = denormalize_data(predictions, scaler_y_max, scaler_y_min)

    # Replace predictions where df has zeros with zero
    for i in range(len(df)):
        if df.iloc[i].eq(0).any():
            predictions_denormalized[i] = 0

    return predictions_denormalized.tolist()

def normalize_data(df, scaler_max, scaler_min):
    # Convert scalers to numpy arrays for arithmetic operations
    scaler_max = np.array(scaler_max)
    scaler_min = np.array(scaler_min)
    
    # Normalize the dataframe using given scalers
    df_normalized = (df - scaler_min) / (scaler_max - scaler_min)
    
    # Replace NaN values with 0 (if any)
    df_normalized = np.nan_to_num(df_normalized, nan=0.0)
    
    return df_normalized.astype(np.float32)


def denormalize_data(df_normalized, scaler_max, scaler_min):
    # Convert scalers to numpy arrays for arithmetic operations
    scaler_max = np.array(scaler_max)
    scaler_min = np.array(scaler_min)

    # Denormalize the dataframe using given scalers
    df_denormalized = df_normalized * (scaler_max - scaler_min) + scaler_min
    
    return df_denormalized

