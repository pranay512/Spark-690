import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline

import warnings
warnings.filterwarnings("ignore")

# Sample Data Creation
data = """date,price,type
2019 Q1,79.78,actual
2019 Q2,87.55,actual
2019 Q3,88.41,actual
2019 Q4,88.09,actual
2020 Q1,31.55,actual
2020 Q2,34.61,actual
2020 Q3,34.75,actual
2020 Q4,43.25,actual
2021 Q1,57.54,actual
2021 Q2,52.29,actual
2021 Q3,47.57,actual
2021 Q4,43.78,actual
2022 Q1,46.36,actual
2022 Q2,35.42,actual
2022 Q3,32.53,actual
2022 Q4,37.70,actual
2023 Q1,44.25,actual
2023 Q2,54.87,actual
2023 Q3,42.30,actual
2023 Q4,41.26,actual
2024 Q1,47.88,actual
2024 Q2,54.56,forecast
2024 Q3,55.01,forecast
2024 Q4,54.32,forecast
2025 Q1,52.99,forecast
"""

# Read the data into a DataFrame
df = pd.read_csv(StringIO(data))

# Filter actual and forecast data
start_date = "2019 Q1"
end_date_actual = "2024 Q1"
df_actual = df[
    (df['date'] >= start_date) & 
    (df['date'] <= end_date_actual) & 
    (df['type'] == 'actual')
]
df_forecast = df[df['type'] == 'forecast']

# Combine them for continuous plotting
df_combined = pd.concat([df_actual, df_forecast], ignore_index=True)
df_combined['x'] = np.arange(len(df_combined))

# Identify the index where forecast starts
forecast_start_idx = df_combined[df_combined['type'] == 'forecast'].index.min()

# Split the data into actual and forecast
x_all = df_combined['x'].values
y_all = df_combined['price'].values

# For spline smoothing, increase the number of points
# We'll apply spline to the entire dataset to maintain continuity
x_smooth = np.linspace(x_all.min(), x_all.max(), 300)
spline = make_interp_spline(x_all, y_all, k=3)
y_smooth = spline(x_smooth)

# Set figure size and DPI (3000x900 pixels at 200 DPI gives 15x4.5 inches)
dpi = 200
width = 3000 / dpi  # 15 inches
height = 900 / dpi  # 4.5 inches
fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

# Plot the smooth line
ax.plot(x_smooth, y_smooth, color='#225ea8', linewidth=2, zorder=2)

# Plot markers on actual data points
ax.scatter(x_all[:forecast_start_idx], y_all[:forecast_start_idx], color='#225ea8', zorder=3)

# Plot markers on forecast data points
ax.scatter(x_all[forecast_start_idx:], y_all[forecast_start_idx:], color='green', zorder=4)

# Annotate each data point with its price
for i, row in df_combined.iterrows():
    ax.annotate(
        f"{row['price']:.2f}", 
        (row['x'], row['price']),
        textcoords="offset points", 
        xytext=(0,5),
        ha='center', 
        fontsize=8, 
        color='black'
    )

# Add annotations for Actual and Forecast with enhanced readability
# Annotate 'Actual' near the last actual point
last_actual_idx = forecast_start_idx - 1
ax.annotate(
    'Actual',
    (df_combined.loc[last_actual_idx, 'x'], df_combined.loc[last_actual_idx, 'price']),
    textcoords="offset points",
    xytext=(-50, -30),
    ha='right',
    fontsize=12,
    fontweight='bold',
    color='#225ea8',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6),
    arrowprops=dict(arrowstyle="->", color='#225ea8', lw=1.5)
)

# Annotate 'Forecast' near the first forecast point
first_forecast_idx = forecast_start_idx
ax.annotate(
    'Forecast',
    (df_combined.loc[first_forecast_idx, 'x'], df_combined.loc[first_forecast_idx, 'price']),
    textcoords="offset points",
    xytext=(50, 30),
    ha='left',
    fontsize=12,
    fontweight='bold',
    color='green',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6),
    arrowprops=dict(arrowstyle="->", color='green', lw=1.5)
)

# Customize Axes
ax.set_xticks(df_combined['x'])
ax.set_xticklabels(df_combined['date'], rotation=45, ha='right')
ax.set_xlabel("Date", fontsize=14, labelpad=10)
ax.set_ylabel("Price", fontsize=14, labelpad=10)

# Remove Title

# Remove Grid and Spines
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Adjust Y-axis label padding to bring labels closer to the chart
ax.tick_params(axis='y', which='major', pad=5)  # Reduce padding

# Remove Legend
# ax.legend(loc='upper left')  # Removed as per request

# Enhance Layout
plt.tight_layout()

# Display the Plot
plt.show()