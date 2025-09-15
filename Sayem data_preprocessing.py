
"""
HIT140 Foundations of Data Science
Project: Bat vs. Rat - Data Preprocessing & Cleaning
Author: Person 1
Task: Load datasets, preprocess, merge, and prepare for analysis
"""

import pandas as pd

# -------------------------------
# 1. Load both datasets
# -------------------------------
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

# -------------------------------
# 2. Convert all time columns to datetime
# -------------------------------
time_cols_ds1 = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
time_cols_ds2 = ["time"]

for col in time_cols_ds1:
    dataset1[col] = pd.to_datetime(dataset1[col], errors="coerce")

for col in time_cols_ds2:
    dataset2[col] = pd.to_datetime(dataset2[col], errors="coerce")

# -------------------------------
# 3. Merge datasets by 30-minute intervals
# -------------------------------
dataset1["time_30min"] = dataset1["start_time"].dt.floor("30T")

merged = pd.merge(
    dataset1,
    dataset2,
    left_on="time_30min",
    right_on="time",
    how="left"
)

# -------------------------------
# 4. Create rat_present column
# -------------------------------
merged["rat_present"] = merged["rat_minutes"].apply(
    lambda x: 1 if pd.notnull(x) and x > 0 else 0
)

# -------------------------------
# 5. Handle missing values
# -------------------------------
# Fill numeric columns with 0 when missing means "no observation"
for col in ["rat_minutes", "bat_landing_number", "food_availability"]:
    if col in merged.columns:
        merged[col].fillna(0, inplace=True)

# Fill categorical columns with "Unknown" if missing
for col in ["season", "month"]:
    if col in merged.columns:
        merged[col].fillna("Unknown", inplace=True)

# Leave datetime NaN (NaT) as-is to preserve missing information

# -------------------------------
# 6. Summary of missing values
# -------------------------------
print("✅ Cleaned dataframe shape:", merged.shape)
print("\n✅ Missing values per column:")
print(merged.isnull().sum())

# -------------------------------
# 7. Save cleaned dataset
# -------------------------------
merged.to_csv("cleaned_merged_dataset.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_merged_dataset.csv'")
