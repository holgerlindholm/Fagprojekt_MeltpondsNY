import pandas as pd
import numpy as np
import h5py
from tensorflow.keras.models import load_model
import os

dir_path = "C:/Users/35466/Desktop/Python_Projects/mp_machine_learning/"

def predict_depth(df):
    # Load the Keras model from h5 file
    model_path = os.path.join(dir_path, '7383_9P.h5')
    model = load_model(model_path)

    # Load scalers from CSV files
    scaler_X_max_path = os.path.join(dir_path, 'scaler_X_max.csv')
    scaler_X_min_path = os.path.join(dir_path, 'scaler_X_min.csv')
    scaler_y_max_path = os.path.join(dir_path, 'scaler_y_max.csv')
    scaler_y_min_path = os.path.join(dir_path, 'scaler_y_min.csv')

    scaler_X_max = pd.read_csv(scaler_X_max_path, index_col=0)
    scaler_X_min = pd.read_csv(scaler_X_min_path, index_col=0)
    scaler_y_max = pd.read_csv(scaler_y_max_path, index_col=0)
    scaler_y_min = pd.read_csv(scaler_y_min_path, index_col=0)

    # Normalize the data
    df_normalized = normalize_data(df, scaler_X_max, scaler_X_min)

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
    # Normalize the dataframe using given scalers
    df_normalized = (df - scaler_min) / (scaler_max - scaler_min)
    return df_normalized.values.astype(np.float32)

def denormalize_data(df_normalized, scaler_max, scaler_min):
    # Denormalize the dataframe using given scalers
    df_denormalized = df_normalized * (scaler_max - scaler_min) + scaler_min
    return df_denormalized

# Example usage:
if __name__ == "__main__":
    # Assume df is your DataFrame
    df = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [0.2, 0.3, 0.4, 0.5]
    })

    predictions = predict_depth(df)
    print(predictions)
