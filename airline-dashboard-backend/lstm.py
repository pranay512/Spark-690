# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Step 1: Load Data and Select Useful Features
# Loading the dataset
file_path = "stock_x_fin_op.csv"
data = pd.read_csv(file_path)

# Filter the dataset to include only rows corresponding to American Airlines
alaska_airlines_data = data[data['UNIQUE_CARRIER_NAME'] == 'Alaska Airlines Inc.']


# Step 2: Data preparation
# Selecting relevant features and the target
features = alaska_airlines_data.loc[:, "ASM":"PRASM"]
target = alaska_airlines_data["Close"]

# Dropping rows with missing values
features = features.dropna()
target = target.loc[features.index]

# Normalizing the features and target
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

features_scaled = scaler_x.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

# Creating sequences for LSTM
sequence_length = 20 # Increased sequence length for better pattern capture

def create_sequences(features, target, seq_length):
    x, y = [], []
    for i in range(len(features) - seq_length):
        x.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(x), np.array(y)

X, y = create_sequences(features_scaled, target_scaled, sequence_length)

# Splitting data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Build the LSTM model with tanh activation function
model = Sequential([
    LSTM(100, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(100, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(50, activation='relu'),  # You can keep ReLU for dense layer for non-linearity, or change to 'tanh'
    Dense(1)  # Output layer for regression
])
# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Step 4: Train the model with Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_unscaled, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_unscaled, y_pred)
r2 = r2_score(y_test_unscaled, y_pred)

# Print the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Step 6: Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs for Alaska Airlines')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 7: Plot Actual vs Predicted values with quarters
quarters = alaska_airlines_data["YEAR"].astype(str) + " Q" + alaska_airlines_data["QUARTER"].astype(str)
years_quarters = quarters.iloc[-len(y_test_unscaled):].values
plt.figure(figsize=(12, 6))
plt.plot(years_quarters, y_test_unscaled, label='Actual Values', marker='o')
plt.plot(years_quarters, y_pred, label='Predicted Values', marker='x')
plt.title('Actual vs Predicted Stock Prices for Alaska Airlines')
plt.xlabel('Year and Quarter')
plt.xticks(rotation=45)
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()

# Save the model
model.save('Alaska_Airlines_lstm_model.h5')
print("Model saved as 'Alaska_Airlines_lstm_model.h5'")

# Start forecasting using the last sequence
forecast_steps = 4  # Number of future quarters to predict
last_sequence = X[-1]
predictions_scaled = []

# Generate predictions iteratively
current_sequence = last_sequence.copy()
for _ in range(forecast_steps):
    pred_scaled = model.predict(current_sequence[np.newaxis, :, :])
    predictions_scaled.append(pred_scaled[0, 0])
    next_step = np.full((1, current_sequence.shape[1]), pred_scaled[0, 0])
    current_sequence = np.vstack([current_sequence[1:], next_step])

# Inverse transform predictions to get actual values
predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

# Prepare results for visualization
forecast_quarters = []
current_year = alaska_airlines_data["YEAR"].iloc[-1]
current_quarter = alaska_airlines_data["QUARTER"].iloc[-1]

for i in range(1, forecast_steps + 1):
    current_quarter += 1
    if current_quarter > 4:
        current_quarter = 1
        current_year += 1
    forecast_quarters.append(f"{current_year} Q{current_quarter}")

forecast_results = pd.DataFrame({
    "Quarter": forecast_quarters,
    "Predicted_Close": predictions.flatten()
})

# Combine actual and forecasted data
combined_df = pd.concat([
    pd.DataFrame({
        'date': alaska_airlines_data["YEAR"].astype(str) + " Q" + alaska_airlines_data["QUARTER"].astype(str),
        'price': target.values,
        'type': 'actual'
    }),
    pd.DataFrame({
        'date': forecast_results["Quarter"],
        'price': forecast_results["Predicted_Close"].values,
        'type': 'forecast'
    })
], ignore_index=True)

# Export to CSV for D3 visualization
combined_df.to_csv('united_airlines_forecast.csv', index=False)
print("Data exported to 'allegiant_forecast.csv'")
