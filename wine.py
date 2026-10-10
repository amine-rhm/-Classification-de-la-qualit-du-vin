import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

def load_data(red_path="winequality-red.csv", white_path="winequality-white.csv"):
    red = pd.read_csv(red_path, sep=";"); red["type"] = "red"
    white = pd.read_csv(white_path, sep=";"); white["type"] = "white"
    return pd.concat([red, white], ignore_index=True)

def prepare(df):
    df = df.drop_duplicates().reset_index(drop=True).copy()
    df["type_white"] = (df["type"] == "white").astype(int)
    df["good"] = (df["quality"] >= 7).astype(int)
    X = df.drop(columns=["quality", "type", "good"])
    return X, df["good"]

def split(X, y, seed=42):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

def make_pipe(clf, k=12):
    return Pipeline([("scaler", StandardScaler()),
                     ("select", SelectKBest(f_classif, k=k)),
                     ("clf", clf)])