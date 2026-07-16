import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path, encoding="unicode_escape")
    
    rename_map = {}
    for col in df.columns:

        if col.startswith("Temperature"):
            rename_map[col] = "Temperature_C"
        elif col.startswith("Dew point temperature"):
            rename_map[col] = "Dew point temperature_C"
            
    df= df.rename(columns=rename_map)
    return df


def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "Rented Bike Count", "Hour", "Temperature_C", "Humidity(%)",
        "Wind speed (m/s)", "Visibility (10m)", "Dew point temperature_C",
        "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def drop_functional(df : pd.DataFrame) -> pd.DataFrame:
    df = df[df["Functioning Day"] == "Yes"].copy()
    df = df.drop(columns=["Functioning Day"])
    return df.reset_index(drop=True)



def split_data(df: pd.DataFrame, date_col: str = "Date",train_frac: float = 0.7, val_frac: float = 0.15):

    df = df.copy()
    test_frac = 1.0 - (train_frac + val_frac)

    train_val_df, test_df = train_test_split(
        df,
        test_size = test_frac,
        random_state=42,
        shuffle=True
    )
 
    relative_val_size = val_frac / (train_frac + val_frac)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=42,
        shuffle=True
    )
 
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def run_preprocessing(raw_csv_path: str):
    df = load_data(raw_csv_path)
    df = fix_datatypes(df)
    df = drop_functional(df)
    train_df, val_df, test_df = split_data(df)
    return train_df, val_df, test_df


