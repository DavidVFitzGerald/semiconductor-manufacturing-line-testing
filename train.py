import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from get_data import download_and_extract_data
from preprocessing import ConstantColumnDropper, CorrelationFilter, HighNaNColumnDropper


def load_data(url):
    download_and_extract_data(url)
    data_path = r"data\secom.data"
    df = pd.read_csv(data_path, sep=" ", header=None)
    labels_path = r"data\secom_labels.data"
    labels_df = pd.read_csv(labels_path, sep=" ", header=None)
    return df, labels_df


def train_model(df, labels_df):
    X = df.to_numpy()
    y = labels_df[0].to_numpy()
    y = (y == 1).astype(int)  # Set -1 values to 0
    test_size = 0.2
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=350,
        class_weight="balanced",
        max_depth=12,
        max_features=0.5,
        min_samples_leaf=8,
        n_jobs=-1,
        random_state=42,
    )

    steps = [
        ("drop_nans", HighNaNColumnDropper(nan_threshold=0.5)),
        ("drop_constant", ConstantColumnDropper()),
        ("imputer", SimpleImputer(strategy="median")),
        ("corr_filter", CorrelationFilter(corr_threshold=0.9)),
        ("classifier", rf),
    ]
    pipeline = Pipeline(steps)
    pipeline.fit(X_full_train, y_full_train)
    return pipeline


def save_model(pipeline, filename="model.bin"):
    with open(filename, "wb") as f:
        pickle.dump(pipeline, f)

    print("Model training complete. Saved to model.bin")


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
    df, labels_df = load_data(url)
    pipeline = train_model(df, labels_df)
    save_model(pipeline)
