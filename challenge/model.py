import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from challenge.utils import pipeline, get_feature_names, save_the_pipe, FEATURES_COLS
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pathlib import Path
import xgboost as xgb
import joblib
class DelayModel:
    """
    Model for predicting flight delays.
    
    This model uses features from flight data to predict whether a flight will be delayed.
    """

    def __init__(
        self
    ):
        """Initialize the DelayModel."""
        self._model = None  # Model should be saved in this attribute.
        self._pipeline = pipeline  # Store the preprocessing pipeline from utils
        self.threshold_in_minutes = 15
        self._model_path = Path(__file__).resolve().parents[1] / 'model' / 'delay_model.joblib'
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Create a copy of the data to avoid modifying the original
        data_copy = data.copy()
        features = self._pipeline.fit_transform(data_copy)
        save_the_pipe(self._pipeline)
        features_df = pd.DataFrame(features,columns=get_feature_names(self._pipeline))

        if target_column is not None:

            data_copy[target_column] = np.where(features_df['min_diff'] > self.threshold_in_minutes, 1, 0)
            y = pd.DataFrame(data_copy[target_column])

            return features_df[FEATURES_COLS].astype(int), y
        return features_df[FEATURES_COLS].astype(int)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        df = features.copy()
        df["target"] = target
        df = shuffle(df, random_state=111)
        y = df.pop('target')

        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(x_train, y_train)

        # Save the trained model
        self._model_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        joblib.dump(self._model, self._model_path)
        print(f"Model saved at {self._model_path}")

        return

    def load_model(self, data_path: str):
        """
        Load the model from disk if available, otherwise train it and save it.
        """
        if self._model_path.exists():
            self._model = joblib.load(self._model_path)
            print(f"Model loaded from {self._model_path}")
        else:
            # Train the model if it doesn't exist
            data = pd.read_csv(data_path)
            features_, target = self.preprocess(data, "delay")
            self.fit(features_, target)
            print(f"Model trained and saved at {self._model_path}")
    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            data_path = Path(__file__).resolve().parents[1] / 'data' / 'data.csv'
            self.load_model(data_path)

        predictions = self._model.predict(features)
        return [int(x) for x in predictions]