import re
import pandas as pd
from sklearn.utils import resample

def clean_text(text):
    text = text.lower()
    text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    return text

def preprocess_df(df, text_col="tweet"):
    df[text_col] = df[text_col].apply(clean_text)
    return df

def upsample(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    majority = df[df[label_col] == 0]
    minority = df[df[label_col] == 1]
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=123
    )
    return pd.concat(
        [pd.DataFrame(majority), pd.DataFrame(minority_upsampled)],
        axis=0
    ).reset_index(drop=True)
