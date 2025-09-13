import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics as st

# ------------------ Step 1: Load and Clean ------------------

# Read CSV files
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

# Keep only 200 rows
df1 = df1.head(200)
df2 = df2.head(200)

# Remove missing values
df1 = df1.dropna()
df2 = df2.dropna()

# Merge datasets on "month"
merged_data = pd.merge(df1, df2, on="month", how="inner").dropna()

# Split data by risk value
risk0 = merged_data[merged_data["risk"] == 0].dropna()
risk1 = merged_data[merged_data["risk"] == 1].dropna()

# Extract columns
bat0 = risk0["bat_landing_number"]
rat0 = risk0["rat_arrival_number"]
bat1 = risk1["bat_landing_number"]
rat1 = risk1["rat_arrival_number"]

# ------------------ Step 2: Charts ------------------

def plot_histograms():
    """Histogram plots for both risk levels"""

    plt.hist(bat0, bins=6, color="lightblue", edgecolor="black")
    plt.title("Distribution of Bat Activity (No Risk)")
    plt.xlabel("Bat Landings")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    plt.hist(bat1, bins=6, color="salmon", edgecolor="black")
    plt.title("Distribution of Bat Activity (Risk Present)")
    plt.xlabel("Bat Landings")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    plt.hist(rat0, bins=6, color="limegreen", edgecolor="black")
    plt.title("Rat Arrival Spread (No Risk)")
    plt.xlabel("Rat Arrivals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    plt.hist(rat1, bins=6, color="gold", edgecolor="black")
    plt.title("Rat Arrival Spread (Risk Present)")
    plt.xlabel("Rat Arrivals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# ------------------ Step 3: Descriptive Statistics ------------------

def show_statistics():
    """Show summary statistics of each group"""

    def describe(series, label):
        mean = np.mean(series)
        median = np.median(series)
        mode = st.mode(series)
        sd = np.std(series)
        q1 = np.percentile(series, 25)
        q2 = np.percentile(series, 50)
        q3 = np.percentile(series, 75)
        iqr = q3 - q1

        print(f"\nStatistics for {label}:")
        print(f"  Average: {mean}")
        print(f"  Median: {median}")
        print(f"  Mode: {mode}")
        print(f"  Std Dev: {sd}")
        print(f"  25% Quartile: {q1}")
        print(f"  50% Quartile: {q2}")
        print(f"  75% Quartile: {q3}")
        print(f"  IQR: {iqr}")
        print("------------------------------------------------")

    describe(bat0, "Bat Landings (No Risk)")
    describe(rat0, "Rat Arrivals (No Risk)")
    describe(bat1, "Bat Landings (Risk Present)")
    describe(rat1, "Rat Arrivals (Risk Present)")

# ------------------ Step 4: Regression ------------------

# Regression for risk=0
slope0, intercept0, r0, p0, se0 = stats.linregress(bat0, rat0)
pred0 = [slope0 * x + intercept0 for x in bat0]

# Regression for risk=1
slope1, intercept1, r1, p1, se1 = stats.linregress(bat1, rat1)
pred1 = [slope1 * x + intercept1 for x in bat1]

def plot_regression():
    """Scatter plots with fitted regression lines"""

    plt.scatter(bat0, rat0, color="blue", label="Observed")
    plt.plot(bat0, pred0, color="red", label="Regression")
    plt.title("Bat vs Rat Relationship (No Risk Condition)")
    plt.xlabel("Bat Landings")
    plt.ylabel("Rat Arrivals")
    plt.legend()
    plt.grid()
    plt.show()

    plt.scatter(bat1, rat1, color="green", label="Observed")
    plt.plot(bat1, pred1, color="orange", label="Regression")
    plt.title("Bat vs Rat Relationship (Risk Condition)")
    plt.xlabel("Bat Landings")
    plt.ylabel("Rat Arrivals")
    plt.legend()
    plt.grid()
    plt.show()

# ------------------ Step 5: Run ------------------

show_statistics()
plot_histograms()
plot_regression()
