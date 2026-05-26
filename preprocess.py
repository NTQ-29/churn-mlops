import pandas as pd
import os

RAW_DATA_PATH = "data/raw/churn.csv"
PROCESSED_DATA_PATH = "data/processed/churn_clean.csv"

def main():
    print("🔄 Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # ---------------------------------------------------------
    # STEP 1 — Convert numeric-looking columns safely
    # ---------------------------------------------------------
    print("🔧 Converting numeric columns where possible...")
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            pass  # Leave column unchanged if conversion fails

    # ---------------------------------------------------------
    # STEP 2 — Handle missing values
    # ---------------------------------------------------------
    print("🧹 Handling missing values...")
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # ---------------------------------------------------------
    # STEP 3 — Encode categorical variables
    # ---------------------------------------------------------
    print("🔧 Encoding categorical variables...")
    df = pd.get_dummies(df, drop_first=True)

    # ---------------------------------------------------------
    # STEP 4 — Ensure output directory exists
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # ---------------------------------------------------------
    # STEP 5 — Save processed dataset
    # ---------------------------------------------------------
    print("💾 Saving processed dataset...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"✅ Preprocessing complete! Saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
