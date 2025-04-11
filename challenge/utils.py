"""
Utility functions and transformers for feature engineering in the flight delay prediction challenge.

This module provides functions and custom transformers for processing flight data,
including date/time feature extraction, season classification, and data preprocessing pipelines.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import os


FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

def get_period_day(date):
    """
    Classify a timestamp into a period of the day (morning, afternoon, or night).
    
    Parameters:
    -----------
    date : str
        String containing a datetime in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
    --------
    str
        The period of the day: 'mañana' (morning), 'tarde' (afternoon), or 'noche' (night/evening)
    """
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()

    if (date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif (date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif (
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
    ):
        return 'noche'


def is_high_season(fecha):
    """
    Determine if a given date falls within a high season period.
    
    High seasons are defined as:
    - December 15 to December 31
    - January 1 to March 3
    - July 15 to July 31
    - September 11 to September 30
    
    Parameters:
    -----------
    fecha : str
        String containing a datetime in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
    --------
    int
        1 if the date is in high season, 0 otherwise
    """
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0


def get_min_diff(data):
    """
    Calculate the time difference in minutes between the scheduled departure time (Fecha-O)
    and the scheduled arrival time (Fecha-I).
    
    Parameters:
    -----------
    data : pandas.Series or dict
        Row of data containing 'Fecha-O' and 'Fecha-I' datetime strings
    
    Returns:
    --------
    float
        Time difference in minutes
    """
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff


class DateFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that extracts features from datetime columns.
    
    This transformer creates three new features:
    - period_day: Time of day classification (morning, afternoon, night)
    - high_season: Binary indicator for high travel season
    - min_diff: Time difference in minutes between departure and arrival
    
    Adheres to sklearn's transformer interface with fit and transform methods.
    """
    
    def fit(self, X, y=None):
        """
        Fit method (no-op, returns self).
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : array-like, optional
            Target values, ignored
            
        Returns:
        --------
        self : object
            Returns self
        """
        return self

    def transform(self, X):
        """
        Transform the input data by adding date-related features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with at least 'Fecha-I' and 'Fecha-O' columns
            
        Returns:
        --------
        pandas.DataFrame
            Transformed data with only the new features:
            'period_day', 'high_season', and 'min_diff'
        """
        X = X.copy()
        X['period_day'] = X['Fecha-I'].apply(get_period_day)
        X['high_season'] = X['Fecha-I'].apply(is_high_season)
        X['min_diff'] = X.apply(get_min_diff, axis=1)
        return X[['period_day', 'high_season', 'min_diff']]

    def get_feature_names_out(self, input_features=None):
        return ['period_day', 'high_season', 'min_diff']


def get_feature_names(pipeline):
    """
    Extract feature names from a preprocessing pipeline.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        The preprocessing pipeline
        
    Returns:
    --------
    list
        List of feature names
    """
    col_transformer = pipeline.named_steps['preprocessor']
    feature_names = []
    for name, transformer, columns in col_transformer.transformers_:
        if name == 'remainder':
            continue  # skip remainder if any
        if hasattr(transformer, 'get_feature_names_out'):
            # Handle OneHotEncoder
            if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(transformer.get_feature_names_out())
        else:
            # Custom transformer – manually add names or column labels
            if isinstance(columns, list):
                feature_names.extend([f"{name}__{col}" for col in columns])
            else:
                feature_names.append(name)
    return feature_names


def save_the_pipe(pipeline, path=None):
    """
    Save the sklearn pipeline to a pickle file.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        The preprocessing pipeline to save
    path : str, optional
        The path where to save the pipeline. If None, defaults to 'model/pipeline.pickle'
        in the project root directory
        
    Returns:
    --------
    str
        Path to the saved pipeline file
    """
    # Default path is in the model directory at project root
    if path is None:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root and into model directory
        project_root = os.path.dirname(current_dir)
        path = os.path.join(project_root, 'model', 'pipeline.joblib')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the pipeline
    joblib.dump(pipeline, path)
    
    return path


def load_the_pipe(path=None):
    """
    Load a saved sklearn pipeline from a pickle file.
    
    Parameters:
    -----------
    path : str, optional
        The path to the saved pipeline. If None, defaults to 'model/pipeline.pickle'
        in the project root directory
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        The loaded pipeline
    """
    # Default path is in the model directory at project root
    if path is None:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root and into model directory
        project_root = os.path.dirname(current_dir)
        path = os.path.join(project_root, 'model', 'pipeline.pickle')
    
    # Load and return the pipeline

    return joblib.load(path)


# Pipeline definition for data preprocessing
# Define categorical features that will be one-hot encoded
categorical_features = ['OPERA', 'TIPOVUELO', 'MES']
categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Create a column transformer that applies different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),  # One-hot encode categorical features
        ('date', DateFeaturesAdder(), ['Fecha-I', 'Fecha-O'])  # Extract custom features from date columns
    ]
)

# Define the preprocessing pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
