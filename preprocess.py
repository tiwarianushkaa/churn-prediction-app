import pandas as pd

def preprocess_data(df):

    df.columns = df.columns.str.strip()

    # Detect target
    target_col = None
    for col in df.columns:
        if col.lower() in ['churn', 'exited', 'target']:
            target_col = col
            break

    if target_col is None:
        raise ValueError("No target column found")

    # Convert target
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    # Drop ID columns
    for col in df.columns:
        if 'id' in col.lower():
            df = df.drop(col, axis=1)

    # Handle missing
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encode
    df = pd.get_dummies(df, drop_first=True)

    return df