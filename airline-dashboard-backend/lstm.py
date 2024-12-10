{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "04505be3-abf5-417a-9581-c6d34e298843",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
    "import matplotlib.pyplot as plt\n",
    "import plotly.graph_objects as go"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ae923e26-b4e7-4462-be2f-1266fdd1bd82",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Load Data and Select Useful Features\n",
    "# Loading the dataset\n",
    "file_path = \"C:/Users/Abhinav/Documents/sem4_capstone/dataset/t100segment_US_non_stop/stock_x_fin_op.csv\"\n",
    "data = pd.read_csv(file_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "e9e16ce1-d00c-401a-ae4e-b55db25301e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter the dataset to include only rows corresponding to American Airlines\n",
    "alaska_airlines_data = data[data['UNIQUE_CARRIER_NAME'] == 'United Air Lines Inc.']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "fe2818cd-778a-4bc5-9388-8e3599204007",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Step 2: Data preparation\n",
    "# Selecting relevant features and the target\n",
    "features = alaska_airlines_data.loc[:, \"ASM\":\"PRASM\"]\n",
    "target = alaska_airlines_data[\"Close\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "1fb435cc-097e-4f81-a2b1-0ae084a80d9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping rows with missing values\n",
    "features = features.dropna()\n",
    "target = target.loc[features.index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "1570bde4-8008-42cc-93a2-a59c23f91f30",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalizing the features and target\n",
    "scaler_x = MinMaxScaler()\n",
    "scaler_y = MinMaxScaler()\n",
    "\n",
    "features_scaled = scaler_x.fit_transform(features)\n",
    "target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "b7ab6b05-19bd-4161-abbb-759bae94cf31",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating sequences for LSTM\n",
    "sequence_length = 20 # Increased sequence length for better pattern capture\n",
    "\n",
    "def create_sequences(features, target, seq_length):\n",
    "    x, y = [], []\n",
    "    for i in range(len(features) - seq_length):\n",
    "        x.append(features[i:i + seq_length])\n",
    "        y.append(target[i + seq_length])\n",
    "    return np.array(x), np.array(y)\n",
    "\n",
    "X, y = create_sequences(features_scaled, target_scaled, sequence_length)\n",
    "\n",
    "# Splitting data into train, validation, and test sets\n",
    "X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "b89bcec2-d0bd-4327-a2a7-2d481add4963",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 3: Build the LSTM model with tanh activation function\n",
    "model = Sequential([\n",
    "    LSTM(100, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),\n",
    "    Dropout(0.3),\n",
    "    LSTM(100, activation='tanh', return_sequences=False),\n",
    "    Dropout(0.3),\n",
    "    Dense(50, activation='relu'),  # You can keep ReLU for dense layer for non-linearity, or change to 'tanh'\n",
    "    Dense(1)  # Output layer for regression\n",
    "])\n",
    "# Compiling the model\n",
    "model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "ab4e1837-a04b-49f6-8f66-f01f693b2c79",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 284ms/step - loss: 0.2352 - mae: 0.4641 - val_loss: 0.0177 - val_mae: 0.1214\n",
      "Epoch 2/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0692 - mae: 0.2366 - val_loss: 0.1545 - val_mae: 0.3746\n",
      "Epoch 3/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0973 - mae: 0.2683 - val_loss: 0.0327 - val_mae: 0.1621\n",
      "Epoch 4/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 34ms/step - loss: 0.0445 - mae: 0.1765 - val_loss: 0.0145 - val_mae: 0.0915\n",
      "Epoch 5/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 32ms/step - loss: 0.0697 - mae: 0.2189 - val_loss: 0.0266 - val_mae: 0.1374\n",
      "Epoch 6/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 33ms/step - loss: 0.0678 - mae: 0.2254 - val_loss: 0.0113 - val_mae: 0.0883\n",
      "Epoch 7/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 29ms/step - loss: 0.0517 - mae: 0.1926 - val_loss: 0.0146 - val_mae: 0.0899\n",
      "Epoch 8/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0435 - mae: 0.1607 - val_loss: 0.0312 - val_mae: 0.1525\n",
      "Epoch 9/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 31ms/step - loss: 0.0413 - mae: 0.1592 - val_loss: 0.0188 - val_mae: 0.1105\n",
      "Epoch 10/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 34ms/step - loss: 0.0449 - mae: 0.1560 - val_loss: 0.0060 - val_mae: 0.0708\n",
      "Epoch 11/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 29ms/step - loss: 0.0436 - mae: 0.1751 - val_loss: 0.0068 - val_mae: 0.0680\n",
      "Epoch 12/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0435 - mae: 0.1589 - val_loss: 0.0060 - val_mae: 0.0649\n",
      "Epoch 13/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 31ms/step - loss: 0.0392 - mae: 0.1575 - val_loss: 0.0046 - val_mae: 0.0610\n",
      "Epoch 14/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 29ms/step - loss: 0.0368 - mae: 0.1499 - val_loss: 0.0082 - val_mae: 0.0672\n",
      "Epoch 15/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 29ms/step - loss: 0.0370 - mae: 0.1401 - val_loss: 0.0121 - val_mae: 0.0921\n",
      "Epoch 16/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0331 - mae: 0.1437 - val_loss: 0.0091 - val_mae: 0.0762\n",
      "Epoch 17/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0363 - mae: 0.1489 - val_loss: 0.0050 - val_mae: 0.0530\n",
      "Epoch 18/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 38ms/step - loss: 0.0417 - mae: 0.1511 - val_loss: 0.0031 - val_mae: 0.0440\n",
      "Epoch 19/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0444 - mae: 0.1555 - val_loss: 0.0027 - val_mae: 0.0444\n",
      "Epoch 20/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 29ms/step - loss: 0.0331 - mae: 0.1494 - val_loss: 0.0031 - val_mae: 0.0404\n",
      "Epoch 21/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0360 - mae: 0.1371 - val_loss: 0.0063 - val_mae: 0.0556\n",
      "Epoch 22/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 29ms/step - loss: 0.0353 - mae: 0.1266 - val_loss: 0.0104 - val_mae: 0.0821\n",
      "Epoch 23/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0341 - mae: 0.1318 - val_loss: 0.0090 - val_mae: 0.0717\n",
      "Epoch 24/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 30ms/step - loss: 0.0300 - mae: 0.1430 - val_loss: 0.0056 - val_mae: 0.0431\n",
      "Epoch 25/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 32ms/step - loss: 0.0384 - mae: 0.1395 - val_loss: 0.0042 - val_mae: 0.0439\n",
      "Epoch 26/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 32ms/step - loss: 0.0311 - mae: 0.1335 - val_loss: 0.0037 - val_mae: 0.0437\n",
      "Epoch 27/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 31ms/step - loss: 0.0275 - mae: 0.1286 - val_loss: 0.0047 - val_mae: 0.0390\n",
      "Epoch 28/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 32ms/step - loss: 0.0327 - mae: 0.1390 - val_loss: 0.0116 - val_mae: 0.0950\n",
      "Epoch 29/100\n",
      "\u001b[1m2/2\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 34ms/step - loss: 0.0416 - mae: 0.1507 - val_loss: 0.0204 - val_mae: 0.1339\n"
     ]
    }
   ],
   "source": [
    "# Step 4: Train the model with Early Stopping\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "\n",
    "early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)\n",
    "\n",
    "history = model.fit(\n",
    "    X_train, y_train,\n",
    "    validation_data=(X_val, y_val),\n",
    "    epochs=100,\n",
    "    batch_size=32,\n",
    "    callbacks=[early_stopping],\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "983680da-9b0f-40dd-a17d-8d6f7016e706",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 193ms/step\n"
     ]
    }
   ],
   "source": [
    "# Step 5: Evaluate the model and compute metrics\n",
    "# Making predictions on the test set\n",
    "y_pred_scaled = model.predict(X_test)\n",
    "\n",
    "# Inverse transform the scaled predictions and test labels\n",
    "y_pred = scaler_y.inverse_transform(y_pred_scaled)\n",
    "y_test_unscaled = scaler_y.inverse_transform(y_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "3ceadb8a-e46e-4bde-9b12-0dc6403fc1f2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error (MSE): 128.71171371941472\n",
      "Root Mean Squared Error (RMSE): 11.345118497372106\n",
      "Mean Absolute Error (MAE): 8.648868176595053\n",
      "R-squared (R2): 0.5770505002927596\n"
     ]
    }
   ],
   "source": [
    "# Calculate evaluation metrics\n",
    "mse = mean_squared_error(y_test_unscaled, y_pred)\n",
    "rmse = np.sqrt(mse)\n",
    "mae = mean_absolute_error(y_test_unscaled, y_pred)\n",
    "r2 = r2_score(y_test_unscaled, y_pred)\n",
    "\n",
    "# Print the metrics\n",
    "print(f\"Mean Squared Error (MSE): {mse}\")\n",
    "print(f\"Root Mean Squared Error (RMSE): {rmse}\")\n",
    "print(f\"Mean Absolute Error (MAE): {mae}\")\n",
    "print(f\"R-squared (R2): {r2}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b58be694-449d-4010-bbfb-75a7ec07954b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA/IAAAIhCAYAAADtv4ENAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADOyElEQVR4nOzdd3hU1drG4d9MeodAGhBCJ6E3aUoTBAEpikdUBFFQsSOiolixYJfPAihSxAYqHuQoIiBVQKmhhk4IJQECJAFC6uzvjyEjIQGSkGSnPPd1zcWws2fvd5Jwjs+std5lMQzDQERERERERERKBavZBYiIiIiIiIhI3inIi4iIiIiIiJQiCvIiIiIiIiIipYiCvIiIiIiIiEgpoiAvIiIiIiIiUoooyIuIiIiIiIiUIgryIiIiIiIiIqWIgryIiIiIiIhIKaIgLyIiIiIiIlKKKMiLiAgzZszAYrGwfv16s0spE6Kjo7FYLJd9vPrqq2aXSI0aNbjllluK9B5//vknrVq1wsvLC4vFwty5c4v0flm2bt2KxWLBxcWF2NjYXM/p3LkznTt3LpL7Z/3833///SK5/qXi4+Nxc3O74r/hoUOHUqNGjTxf89Lf02XLlmGxWFi2bNm1FSsiIoXC2ewCREREyqrHH3+cu+++O8fxatWqmVBN8TIMgzvuuIN69eoxb948vLy8qF+/frHc+8svvwQgIyODmTNn8txzzxXLfc3y9ddfk5aWBsDUqVNp1apVjnNeeuklnnzyyQLfo0WLFqxZs4YGDRoU+BoiIlJ4FORFREQK4Pz587i7u2OxWC57TvXq1Wnbtm0xVlVyHD16lFOnTnHrrbfStWvXQrlmXr7nqampfPvttzRt2pT4+HimTZtW5oP8tGnTCAwMJCwsjO+//54PP/wQDw+PbOfUrl37qtcxDIOUlJQcrwXw9fUtt7/LIiIlkabWi4hInv3111907doVHx8fPD09ad++Pb/99lu2c5KTkxk9ejQ1a9bE3d0df39/WrVqxffff+84Z//+/dx5551UqVIFNzc3goKC6Nq1K5GRkVetYd68ebRr1w5PT098fHy46aabWLNmjePrc+fOxWKx8Oeff+Z47aRJk7BYLGzZssVxbP369fTt2xd/f3/c3d1p3rw5P/zwQ7bXZS09WLhwIffffz8BAQF4enqSmpqa12/dZXXu3JlGjRqxcuVK2rZti4eHB1WrVuWll14iMzMz27mnTp3ikUceoWrVqri6ulKrVi3Gjh2bow6bzcYnn3xCs2bN8PDwoEKFCrRt25Z58+bluP+CBQto0aIFHh4ehIeHM23atGxfz8vP81KvvvqqY9bBc889h8ViyTatOy+/RwX9ns+dO5eTJ08yfPhw7r33Xnbv3s1ff/11xddkee2112jTpg3+/v74+vrSokULpk6dimEY2c5bsmQJnTt3plKlSnh4eFC9enUGDBhAcnLyZa+dnp7Ovffei7e3N7/++itg/9278847qVGjBh4eHtSoUYO77rqLgwcP5qlegH/++Ydt27YxePBgHnjgARITE5kzZ06O83KbWm+xWHjssceYPHkyERERuLm58dVXX+V6n9ym1g8dOhRvb2/27t1Lr1698Pb2JjQ0lKeffjrHzyktLY033niD8PBw3NzcCAgI4L777uPEiRPZzivI91ZEpDzSiLyIiOTJ8uXLuemmm2jSpAlTp07Fzc2NiRMn0qdPH77//nsGDhwIwKhRo/j666954403aN68OefOnWPbtm2cPHnSca1evXqRmZnJu+++S/Xq1YmPj2f16tUkJCRcsYbvvvuOQYMG0b17d77//ntSU1N599136dy5M3/++Sc33HADt9xyC4GBgUyfPj3HSPCMGTNo0aIFTZo0AWDp0qXcfPPNtGnThsmTJ+Pn58esWbMYOHAgycnJDB06NNvr77//fnr37s3XX3/NuXPncHFxuWK9NpuNjIyMHMednbP/329cXBx33nknY8aMYdy4cfz222+88cYbnD59mk8//RSAlJQUunTpwr59+3jttddo0qQJK1euZPz48URGRmYLwkOHDuWbb75h2LBhjBs3DldXVzZu3Eh0dHS2+27evJmnn36aMWPGEBQUxJdffsmwYcOoU6cOHTt2BPL287zU8OHDadq0KbfddptjeYGbmxuQ99+jgn7Ps645aNAgTp06xfjx45k6dSo33HDDFV8H9rXtDz30ENWrVwfg77//5vHHH+fIkSO8/PLLjnN69+5Nhw4dmDZtGhUqVODIkSMsWLCAtLQ0PD09c1w3ISGB2267jaioKJYvX07Lli0d16pfvz533nkn/v7+xMbGMmnSJK677jp27NhB5cqVr1rz1KlTHd+n0NBQRo4cydSpU7nnnnuu+lqwf/CxcuVK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lStTqVIlFixYkOvXfXx8HM/z+r0VESnvNCIvIiJX5eXlRZs2bfj555+zjRTabDa++eYbqlWrRr169XK8LigoiKFDh3LXXXexa9euXDtP16tXjxdffJHGjRuzcePGy9ZQv359qlatynfffZeti/i5c+eYM2eOo5N9lrvuugt3d3dmzJjBjBkzqFq1qmPENet6devWZfPmzbmOmrdq1SpbwChKZ86cydFR/rvvvsNqtTqaznXt2pWzZ8/mCJYzZ850fB2gZ8+egL1Df2HLy8/zSgr6e5QXU6dOxWq1MnfuXJYuXZrt8fXXXwPk6Mh/MYvFgrOzM05OTo5j58+fd7w2N05OTrRp04bPPvsMIMfvr9Vq5fPPP+fJJ59k6NCh2X4mFosFwzAcTQCzfPnll3kaSc/6IGzKlCk53u/8+fNxcXG54vstTrfccgsnT54kMzMz139nl35gBlf/3oqIlHcakRcREYclS5bk6GwO9i7z48eP56abbqJLly6MHj0aV1dXJk6cyLZt2/j+++8dI3tt2rThlltuoUmTJlSsWJGoqCi+/vprR9DesmULjz32GP/5z3+oW7curq6uLFmyhC1btmQbmbyU1Wrl3XffZdCgQdxyyy089NBDpKam8t5775GQkMDbb7+d7fwKFSpw6623MmPGDBISEhg9enS2tfUAn3/+OT179qRHjx4MHTqUqlWrcurUKaKioti4cSM//vjjNX0/Y2Ji+Pvvv3McDwgIyLavd6VKlXj44YeJiYmhXr16zJ8/nylTpvDwww87RluHDBnCZ599xr333kt0dDSNGzfmr7/+4q233qJXr15069YNsI9oDh48mDfeeINjx45xyy234ObmxqZNm/D09OTxxx/P13u42s8zv/L6e5QfJ0+e5JdffqFHjx7069cv13M++ugjZs6cyfjx43PtfN+7d28+/PBD7r77bh588EFOnjzJ+++/nyNoT548mSVLltC7d2+qV69OSkqKIzBn/Qwu9cEHH+Dj48MjjzzC2bNneeaZZ/D19aVjx4689957VK5cmRo1arB8+XKmTp2a6zT8i2VkZDBz5kwiIiIYPnx4ruf06dOHefPmceLECQICAq54vaJ255138u2339KrVy+efPJJWrdujYuLC4cPH2bp0qX069ePW2+9tUDfWxGRcsvMTnsiIlIyZHUIv9zjwIEDhmEYxsqVK40bb7zR8PLyMjw8PIy2bdsa//vf/7Jda8yYMUarVq2MihUrGm5ubkatWrWMp556yoiPjzcMwzCOHTtmDB061AgPDze8vLwMb29vo0mTJsZHH31kZGRkXLXWuXPnGm3atDHc3d0NLy8vo2vXrsaqVatyPXfhwoWO97B79+5cz9m8ebNxxx13GIGBgYaLi4sRHBxs3HjjjcbkyZNzfH+u1NX/YlfrWj9o0CDHuZ06dTIaNmxoLFu2zGjVqpXh5uZmhISEGC+88EKObvonT540RowYYYSEhBjOzs5GWFiY8fzzzxspKSnZzsvMzDQ++ugjo1GjRoarq6vh5+dntGvXLtvPKiwszOjdu3eO2i/t5n61n+fVvgeXdq03jLz9HuXnez5hwgQDMObOnXvZc7K6/md1Ts+ta/20adOM+vXrO97n+PHjjalTp2b7N7BmzRrj1ltvNcLCwgw3NzejUqVKRqdOnYx58+Zd9b2/9957BmC8/PLLhmEYxuHDh40BAwYYFStWNHx8fIybb77Z2LZtmxEWFmbce++9l30vc+fOverODwsWLDAA44MPPjAM4/Jd6x999NFcX08eu9Z7eXnleO0rr7xiXPqfmOnp6cb7779vNG3a1HB3dze8vb2N8PBw46GHHjL27NljGEbevrciImJnMYyL5ieKiIhIsercuTPx8fFs27bN7FJERESklNAaeREREREREZFSREFeREREREREpBTR1HoRERERERGRUkQj8iIiIiIiIiKliIK8iIiIiIiISCmiIC8iIiIiIiJSijibXUBJZLPZOHr0KD4+PlgsFrPLERERERERkTLOMAzOnDlDlSpVsFqvPOauIJ+Lo0ePEhoaanYZIiIiIiIiUs4cOnSIatWqXfEcBflc+Pj4APZvoK+vr8nViIiIiIiISFmXlJREaGioI49eiYJ8LrKm0/v6+irIi4iIiIiISLHJy/JuNbsTERERERERKUUU5EVERERERERKEQV5ERERERERkVLE9DXyEydO5L333iM2NpaGDRsyYcIEOnTokOu5sbGxPP3002zYsIE9e/bwxBNPMGHChMtee9asWdx1113069ePuXPnFs0bEBERERGRMsUwDDIyMsjMzDS7FCljXFxccHJyuubrmBrkZ8+ezciRI5k4cSLXX389n3/+OT179mTHjh1Ur149x/mpqakEBAQwduxYPvrooyte++DBg4wePfqyHwqIiIiIiIhcKi0tjdjYWJKTk80uRcogi8VCtWrV8Pb2vrbrGIZhFFJN+damTRtatGjBpEmTHMciIiLo378/48ePv+JrO3fuTLNmzXIdkc/MzKRTp07cd999rFy5koSEhHyNyCclJeHn50diYqK61ouIiIiIlBM2m409e/bg5OREQEAArq6ueeogLpIXhmFw4sQJkpOTqVu3bo6R+fzkUNNG5NPS0tiwYQNjxozJdrx79+6sXr36mq49btw4AgICGDZsGCtXrrzq+ampqaSmpjr+npSUdE33FxERERGR0ictLQ2bzUZoaCienp5mlyNlUEBAANHR0aSnp1/TFHvTmt3Fx8eTmZlJUFBQtuNBQUHExcUV+LqrVq1i6tSpTJkyJc+vGT9+PH5+fo5HaGhoge8vIiIiIiKlm9WqnuBSNAprhofpv6GXvhHDMAr85s6cOcM999zDlClTqFy5cp5f9/zzz5OYmOh4HDp0qED3FxERERERESlqpk2tr1y5Mk5OTjlG348fP55jlD6v9u3bR3R0NH369HEcs9lsADg7O7Nr1y5q166d43Vubm64ubkV6J4iIiIiIiIixcm0EXlXV1datmzJokWLsh1ftGgR7du3L9A1w8PD2bp1K5GRkY5H37596dKlC5GRkZoyLyIiIiIikgedO3dm5MiReT4/Ojoai8VCZGRkkdUk/zJ1+7lRo0YxePBgWrVqRbt27fjiiy+IiYlhxIgRgH3K+5EjR5g5c6bjNVm/GGfPnuXEiRNERkbi6upKgwYNcHd3p1GjRtnuUaFCBYAcx0VEREREREq7qy1Lvvfee5kxY0a+r/vzzz/j4uKS5/NDQ0OJjY3N1xLngoiOjqZmzZps2rSJZs2aFem9SjJTg/zAgQM5efIk48aNIzY2lkaNGjF//nzCwsIAiI2NJSYmJttrmjdv7ni+YcMGvvvuO8LCwoiOji7O0kVEREREREwXGxvreD579mxefvlldu3a5Tjm4eGR7fz09PQ8BXR/f/981eHk5ERwcHC+XiMFZ3qzu0ceeYTo6GhSU1PZsGEDHTt2dHxtxowZLFu2LNv5hmHkeFwpxM+YMSNfe8iLiIiIiIiAPXskp2WY8jAMI081BgcHOx5+fn5YLBbH31NSUqhQoQI//PADnTt3xt3dnW+++YaTJ09y1113Ua1aNTw9PWncuDHff/99tuteOrW+Ro0avPXWW9x///34+PhQvXp1vvjiC8fXL51av2zZMiwWC3/++SetWrXC09OT9u3bZ/uQAeCNN94gMDAQHx8fhg8fzpgxY65ppD01NZUnnniCwMBA3N3dueGGG1i3bp3j66dPn2bQoEEEBATg4eFB3bp1mT59OmDffvCxxx4jJCQEd3d3atSowfjx4wtcS1EydUReRERERESkpDqfnkmDl/8w5d47xvXA07Vw4tpzzz3HBx98wPTp03FzcyMlJYWWLVvy3HPP4evry2+//cbgwYOpVasWbdq0uex1PvjgA15//XVeeOEFfvrpJx5++GE6duxIeHj4ZV8zduxYPvjgAwICAhgxYgT3338/q1atAuDbb7/lzTffZOLEiVx//fXMmjWLDz74gJo1axb4vT777LPMmTOHr776irCwMN5991169OjB3r178ff356WXXmLHjh38/vvvVK5cmb1793L+/HkAPv74Y+bNm8cPP/xA9erVOXToUInd0UxBXkREREREpAwbOXIkt912W7Zjo0ePdjx//PHHWbBgAT/++OMVg3yvXr145JFHAPuHAx999BHLli27YpB/88036dSpEwBjxoyhd+/epKSk4O7uzieffMKwYcO47777AHj55ZdZuHAhZ8+eLdD7PHfuHJMmTWLGjBn07NkTgClTprBo0SKmTp3KM888Q0xMDM2bN6dVq1aAfaZBlpiYGOrWrcsNN9yAxWJxLPkuiRTkS7G4xBQ2HDxNdX9PGlfzM7scEREREZEyxcPFiR3jeph278KSFVqzZGZm8vbbbzN79myOHDlCamoqqampeHl5XfE6TZo0cTzPmsJ//PjxPL8mJCQEsG85Xr16dXbt2uX4YCBL69atWbJkSZ7e16X27dtHeno6119/veOYi4sLrVu3JioqCoCHH36YAQMGsHHjRrp3707//v0du6YNHTqUm266ifr163PzzTdzyy230L179wLVUtQU5Euxz1fsY/qqaIa2r6EgLyIiIiJSyCwWS6FNbzfTpQH9gw8+4KOPPmLChAk0btwYLy8vRo4cSVpa2hWvc2mTPIvFgs1my/NrsjrsX/yaS7vu57U3QG6yXpvbNbOO9ezZk4MHD/Lbb7+xePFiunbtyqOPPsr7779PixYtOHDgAL///juLFy/mjjvuoFu3bvz0008FrqmomN7sTgquWWgFADYfTjC1DhERERERKT1WrlxJv379uOeee2jatCm1atViz549xV5H/fr1Wbt2bbZj69evL/D16tSpg6urK3/99ZfjWHp6OuvXryciIsJxLCAggKFDh/LNN98wYcKEbE37fH19GThwIFOmTGH27NnMmTOHU6dOFbimolL6P14qx5pWqwDA9qNJpGXYcHXW5zIiIiIiInJlderUYc6cOaxevZqKFSvy4YcfEhcXly3sFofHH3+cBx54gFatWtG+fXtmz57Nli1bqFWr1lVfe2n3e4AGDRrw8MMP88wzz+Dv70/16tV59913SU5OZtiwYYB9HX7Lli1p2LAhqamp/Prrr473/dFHHxESEkKzZs2wWq38+OOPBAcHU6FChUJ934VBQb4UC6vkiZ+HC4nn09kVd0bT60VERERE5KpeeuklDhw4QI8ePfD09OTBBx+kf//+JCYmFmsdgwYNYv/+/YwePZqUlBTuuOMOhg4dmmOUPjd33nlnjmMHDhzg7bffxmazMXjwYM6cOUOrVq34448/qFixIgCurq48//zzREdH4+HhQYcOHZg1axYA3t7evPPOO+zZswcnJyeuu+465s+fj9Va8gZMLca1LEIoo5KSkvDz8yMxMRFfX1+zy7miIdPWsmL3CV7v34jBbUtuV0URERERkZIuJSWFAwcOULNmTdzd3c0up1y66aabCA4O5uuvvza7lCJxpd+x/OTQkvfRguRLswuj8JsPJZhbiIiIiIiISD4kJyfz4Ycfsn37dnbu3Mkrr7zC4sWLuffee80urcTT1PpSrmlWwzsFeRERERERKUUsFgvz58/njTfeIDU1lfr16zNnzhy6detmdmklnoJ8KdfkQsO7vSfOciYlHR93lyu/QEREREREpATw8PBg8eLFZpdRKmlqfSkX4ONG1QoeGAZsPVK8zSlERERERESk+CnIlwFZ+8lHanq9iIiIiIhImacgXwY0DVXDOxERERERkfJCQb4MaHphnfzmQ5paLyIiIiIiUtYpyJcBjar6YbVAXFIKcYkpZpcjIiIiIiIiRUhBvgzwcnOmXpAPAJsPJ5hbjIiIiIiIiBQpBfky4t/p9Qmm1iEiIiIiIqVP586dGTlypOPvNWrUYMKECVd8jcViYe7cudd878K6TnmiIF9GNL3QuV4j8iIiIiIi5UefPn3o1q1brl9bs2YNFouFjRs35vu669at48EHH7zW8rJ59dVXadasWY7jsbGx9OzZs1DvdakZM2ZQoUKFIr1HcVKQLyOyOtdvOZSIzWaYXI2IiIiIiBSHYcOGsWTJEg4ePJjja9OmTaNZs2a0aNEi39cNCAjA09OzMEq8quDgYNzc3IrlXmWFgnwZUT/IB3cXK2dSM9gff87sckRERERESj/DgLRz5jyMvA3O3XLLLQQGBjJjxoxsx5OTk5k9ezbDhg3j5MmT3HXXXVSrVg1PT08aN27M999/f8XrXjq1fs+ePXTs2BF3d3caNGjAokWLcrzmueeeo169enh6elKrVi1eeukl0tPTAfuI+GuvvcbmzZuxWCxYLBZHzZdOrd+6dSs33ngjHh4eVKpUiQcffJCzZ886vj506FD69+/P+++/T0hICJUqVeLRRx913KsgYmJi6NevH97e3vj6+nLHHXdw7Ngxx9c3b95Mly5d8PHxwdfXl5YtW7J+/XoADh48SJ8+fahYsSJeXl40bNiQ+fPnF7iWvHAu0qtLsXF2stK4qh/rok+z+VACdQK9zS5JRERERKR0S0+Gt6qYc+8XjoKr11VPc3Z2ZsiQIcyYMYOXX34Zi8UCwI8//khaWhqDBg0iOTmZli1b8txzz+Hr68tvv/3G4MGDqVWrFm3atLnqPWw2G7fddhuVK1fm77//JikpKdt6+iw+Pj7MmDGDKlWqsHXrVh544AF8fHx49tlnGThwINu2bWPBggUsXrwYAD8/vxzXSE5O5uabb6Zt27asW7eO48ePM3z4cB577LFsH1YsXbqUkJAQli5dyt69exk4cCDNmjXjgQceuOr7uZRhGPTv3x8vLy+WL19ORkYGjzzyCAMHDmTZsmUADBo0iObNmzNp0iScnJyIjIzExcUFgEcffZS0tDRWrFiBl5cXO3bswNu7aPOYgnwZ0rRaBXuQP5zAgJbVzC5HRERERESKwf333897773HsmXL6NKlC2CfVn/bbbdRsWJFKlasyOjRox3nP/744yxYsIAff/wxT0F+8eLFREVFER0dTbVq9pzx1ltv5VjX/uKLLzqe16hRg6effprZs2fz7LPP4uHhgbe3N87OzgQHB1/2Xt9++y3nz59n5syZeHnZP8j49NNP6dOnD++88w5BQUEAVKxYkU8//RQnJyfCw8Pp3bs3f/75Z4GC/OLFi9myZQsHDhwgNDQUgK+//pqGDRuybt06rrvuOmJiYnjmmWcIDw8HoG7duo7Xx8TEMGDAABo3bgxArVq18l1DfinIlyGOhnfqXC8iIiIicu1cPO0j42bdO4/Cw8Np374906ZNo0uXLuzbt4+VK1eycOFCADIzM3n77beZPXs2R44cITU1ldTUVEdQvpqoqCiqV6/uCPEA7dq1y3HeTz/9xIQJE9i7dy9nz54lIyMDX1/fPL+PrHs1bdo0W23XX389NpuNXbt2OYJ8w4YNcXJycpwTEhLC1q1b83Wvi+8ZGhrqCPEADRo0oEKFCkRFRXHdddcxatQohg8fztdff023bt34z3/+Q+3atQF44oknePjhh1m4cCHdunVjwIABNGnSpEC15JXWyJchzS4E+R2xSaRmZJpbjIiIiIhIaWex2Ke3m/G4MEU+r4YNG8acOXNISkpi+vTphIWF0bVrVwA++OADPvroI5599lmWLFlCZGQkPXr0IC0tLU/XNnJZr2+5pL6///6bO++8k549e/Lrr7+yadMmxo4dm+d7XHyvS6+d2z2zprVf/DWbzZave13tnhcff/XVV9m+fTu9e/dmyZIlNGjQgP/+978ADB8+nP379zN48GC2bt1Kq1at+OSTTwpUS14pyJch1Sp64O/lSnqmQVTsGbPLERERERGRYnLHHXfg5OTEd999x1dffcV9993nCKErV66kX79+3HPPPTRt2pRatWqxZ8+ePF+7QYMGxMTEcPTov7MT1qxZk+2cVatWERYWxtixY2nVqhV169bN0Unf1dWVzMwrDzg2aNCAyMhIzp37t4H3qlWrsFqt1KtXL88150fW+zt06JDj2I4dO0hMTCQiIsJxrF69ejz11FMsXLiQ2267jenTpzu+FhoayogRI/j55595+umnmTJlSpHUmkVBvgyxWCw0rWZvGKHp9SIiIiIi5Ye3tzcDBw7khRde4OjRowwdOtTxtTp16rBo0SJWr15NVFQUDz30EHFxcXm+drdu3ahfvz5Dhgxh8+bNrFy5krFjx2Y7p06dOsTExDBr1iz27dvHxx9/7BixzlKjRg0OHDhAZGQk8fHxpKam5rjXoEGDcHd3595772Xbtm0sXbqUxx9/nMGDBzum1RdUZmYmkZGR2R47duygW7duNGnShEGDBrFx40bWrl3LkCFD6NSpE61ateL8+fM89thjLFu2jIMHD7Jq1SrWrVvnCPkjR47kjz/+4MCBA2zcuJElS5Zk+wCgKCjIlzFaJy8iIiIiUj4NGzaM06dP061bN6pXr+44/tJLL9GiRQt69OhB586dCQ4Opn///nm+rtVq5b///S+pqam0bt2a4cOH8+abb2Y7p1+/fjz11FM89thjNGvWjNWrV/PSSy9lO2fAgAHcfPPNdOnShYCAgFy3wPP09OSPP/7g1KlTXHfdddx+++107dqVTz/9NH/fjFycPXuW5s2bZ3v06tXLsf1dxYoV6dixI926daNWrVrMnj0bACcnJ06ePMmQIUOoV68ed9xxBz179uS1114D7B8QPProo0RERHDzzTdTv359Jk6ceM31XonFyG3BQzmXlJSEn58fiYmJ+W7OYLalu45z3/R11ArwYsnTnc0uR0RERESk1EhJSeHAgQPUrFkTd3d3s8uRMuhKv2P5yaEakS9jmlarAMD+E+dIPJ9ubjEiIiIiIiJS6BTkyxh/L1eq+9u3qth6ONHkakRERERERKSwKciXQY518ocTTK1DRERERERECp+CfBmU1bk+Ug3vREREREREyhwF+TKo2YUR+chDCaiXoYiIiIhI/ui/oaWoFNbvloJ8GdSwih9OVgsnzqQSl5RidjkiIiIiIqWCi4sLAMnJySZXImVVWloaYN/S7lo4F0YxUrJ4uDpRP8iHHbFJbD6UQIifh9kliYiIiIiUeE5OTlSoUIHjx48D9j3NLRaLyVVJWWGz2Thx4gSenp44O19bFFeQL6OahlZgR2wSkYcSublRiNnliIiIiIiUCsHBwQCOMC9SmKxWK9WrV7/mD4gU5MuoZqF+fL8WNqvhnYiIiIhInlksFkJCQggMDCQ9Pd3scqSMcXV1xWq99hXuCvJlVNYWdFsOJ5BpM3CyakqQiIiIiEheOTk5XfM6ZpGiomZ3ZVTdQB88XZ04l5bJvhNnzS5HREREREREComCfBnlZLXQqKr2kxcRERERESlrFOTLsKz95LVOXkREREREpOxQkC/DmlarAMDmwwmm1iEiIiIiIiKFR0G+DGsaap9avzP2DCnpmSZXIyIiIiIiIoVBQb4Mq1rBg8rermTYDLYfTTK7HBERERERESkECvJlmMVi+Xd6vdbJi4iIiIiIlAkK8mVc1n7yWicvIiIiIiJSNijIl3FN1bleRERERESkTFGQL+OaVrM3vIs+mUxCcprJ1YiIiIiIiMi1UpAv4yp4ulKjkicAmw8nmlyNiIiIiIiIXCsF+XJA0+tFRERERETKDgX5cqCZgryIiIiIiEiZYXqQnzhxIjVr1sTd3Z2WLVuycuXKy54bGxvL3XffTf369bFarYwcOTLHOVOmTKFDhw5UrFiRihUr0q1bN9auXVuE76Dku7hzvWEY5hYjIiIiIiIi18TUID979mxGjhzJ2LFj2bRpEx06dKBnz57ExMTken5qaioBAQGMHTuWpk2b5nrOsmXLuOuuu1i6dClr1qyhevXqdO/enSNHjhTlWynRGoT44my1EH82jSMJ580uR0RERERERK6BxTBxiLZNmza0aNGCSZMmOY5FRETQv39/xo8ff8XXdu7cmWbNmjFhwoQrnpeZmUnFihX59NNPGTJkSJ7qSkpKws/Pj8TERHx9ffP0mpKuzyd/sfVIIp/d3YLeTULMLkdEREREREQukp8catqIfFpaGhs2bKB79+7Zjnfv3p3Vq1cX2n2Sk5NJT0/H39//suekpqaSlJSU7VHWNA21b0O3+XCCuYWIiIiIiIjINTEtyMfHx5OZmUlQUFC240FBQcTFxRXafcaMGUPVqlXp1q3bZc8ZP348fn5+jkdoaGih3b+kaFqtAgCRangnIiIiIiJSqpne7M5isWT7u2EYOY4V1Lvvvsv333/Pzz//jLu7+2XPe/7550lMTHQ8Dh06VCj3L0myOtdvPZxIRqbN3GJERERERESkwJzNunHlypVxcnLKMfp+/PjxHKP0BfH+++/z1ltvsXjxYpo0aXLFc93c3HBzc7vme5ZktQK88XZz5mxqBntPnCU8uGys/RcRERERESlvTBuRd3V1pWXLlixatCjb8UWLFtG+fftruvZ7773H66+/zoIFC2jVqtU1XauscLJaaFz1wjp5Ta8XEREREREptUydWj9q1Ci+/PJLpk2bRlRUFE899RQxMTGMGDECsE95v7TTfGRkJJGRkZw9e5YTJ04QGRnJjh07HF9/9913efHFF5k2bRo1atQgLi6OuLg4zp49W6zvrSTK2k8+8lCiuYWIiIiIiIhIgZk2tR5g4MCBnDx5knHjxhEbG0ujRo2YP38+YWFhAMTGxubYU7558+aO5xs2bOC7774jLCyM6OhoACZOnEhaWhq33357tte98sorvPrq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AM6ePcuJEyeIjIzE1dWVBg0aAPDkk0/SsWNH3nnnHfr168cvv/zC4sWL+euvv4r9/ZV2NSp54uZs5Xx6JgdPJVOzspfZJYmIiIiIiJR7pgb5gQMHcvLkScaNG0dsbCyNGjVi/vz5hIWFARAbG5tjT/nmzZs7nm/YsIHvvvuOsLAwoqOjAWjfvj2zZs3ixRdf5KWXXqJ27drMnj2bNm3aFNv7KiucnazUC/Jh65FEdsYmKciLiIiIiIiUAKbuI19S5Wf/vrLumR838+OGwzzRtS6jbqpndjkiIiIiIiJlUqnYR15Kh/ALDe+0l7yIiIiIiEjJoCAvVxQRor3kRUREREREShIFebmirM71MaeSOZuaYXI1IiIiIiIioiAvV+Tv5UqQrxsAu+I0vV5ERERERMRsCvJyVVmj8lGxml4vIiIiIiJiNgV5uapwxzp5jciLiIiIiIiYTUFerqqBo3O9RuRFRERERETMpiAvV5U1tX5n3BkMwzC5GhERERERkfJNQV6uqlaAFy5OFs6mZnD49HmzyxERERERESnXFOTlqlycrNQJtK+Tj4rVOnkREREREREzKchLnkQEZzW80zp5ERERERERMynIS55EZDW8U+d6ERERERERUynIS544tqBT53oRERERERFTKchLnmR1rj9w8hzn0zJNrkZERERERKT8UpCXPAnwcaOytyuGAbuOaVReRERERETELArykmeO/eTVuV5ERERERMQ0CvKSZxEh6lwvIiIiIiJiNgV5ybOsEXntJS8iIiIiImIeBXnJs/CLRuQNwzC5GhERERERkfJJQV7yrE6gN05WC4nn04lNTDG7HBERERERkXJJQV7yzM3ZidoBXgDsjNP0ehERERERETMoyEu+/LtOXg3vREREREREzKAgL/kSEXJhCzp1rhcRERERETGFgrzki6PhnTrXi4iIiIiImEJBXvIl4sLU+v3x50hJzzS5GhERERERkfJHQV7yJcjXjQqeLmTaDPYeP2t2OSIiIiIiIuWOgrzki8ViITzYPr0+StPrRUREREREip2CvOSbGt6JiIiIiIiYR0Fe8i1rnbz2khcRERERESl+CvKSb1md66Niz2AYhsnViIiIiIiIlC8K8pJvdQN9sFrg1Lk0TpxJNbscERERERGRckVBXvLNw9WJGpW9AIjSOnkREREREZFipSAvBeJoeKfO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KaSZKjAcKteDzDTYvdDsahw0Ki8iIiIiIlI4ChTkx44dy6effsrbb7/Npk2b2LhxI2+99RaffPIJL730UmHXKPmVNSpfgqbXa1ReRERERESkcFgMwzDy+6IqVaowefJk+vbtm+34L7/8wiOPPOKYal9aJSUl4efnR2JiYunswn80Er7oBC6e8Mw+cPU0uyIAUtIz6fDuUk6cSeXt2xpzZ+vqZpckIiIiIiJSIuQnhxZoRP7UqVO5roUPDw/n1KlTBbmkFKaQplChOqQnw74/za7GQaPyIiIiIiIi165AQb5p06Z8+umnOY5/+umnNGnS5JqLkmtksVzUvf5/5tZyCa2VFxERERERuTbOBXnRu+++S+/evVm8eDHt2rXDYrGwevVqDh06xPz5JWP/8nIvog+s+RR2LYCMNHB2Nbsi4N9R+dd/3cGnS/dyW4tquDoXePMEERERERGRcqdACapTp07s3r2bW2+9lYSEBE6dOsVtt93G9u3bmT59emHXKAVRrTV4B0FqIhxYYXY12WhUXkREREREpOAK1OzucjZv3kyLFi3IzMwsrEuaotQ3u8vy6yhYPxVaDIG+n5hdTTZT/zrA67/uoFpFD5Y83Vmj8iIiIiIiUq4VebM7KSUaXFgnv/M3sJWsD1c0Ki8iIiIiIlIwCvJlWdj14FERkk/CwdVmV5ONOtiLiIiIiIgUjIJ8WebkAvV72Z+XsO71oFF5ERERERGRgshX1/rbbrvtil9PSEi4llqkKET0hchv7UH+5rfBWnI+u1EHexERERERkfzLV2ry8/O74iMsLIwhQ4YUVa1SELU6g6s3nDkKRzeaXU0OGpUXERERERHJn3yNyGtruVLIxR3q9YBtcyBqHlRrZXZF2WhUXkREREREJH+UmMqDiD72P3fMg8LbbbDQaFReREREREQk7xTky4M6N4GzO5w+AMe2m11NDupgLyIiIiIikncK8uWBmzfU7mp/HjXP3FouQ6PyIiIiIiIieaMgX1406Gv/swRuQwcalRcREREREckrBfnyol4PsDrD8R0Qv9fsanKlUXkREREREZGrU5AvLzwqQs2O9ucldHq9RuVFRERERESuTkG+PInIml5fMoM8aFReRERERETkahTky5Pw3oAFjm6ChENmV5MrjcqLiIiIiIhcmYJ8eeIdCGHt7c93/mpuLVegUXkREREREZHL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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Step 6: Plot training history\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(history.history['loss'], label='Training Loss')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "plt.title('Loss over Epochs for Alaska Airlines')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3bf13bbd-9e7d-4166-870a-a448d4b255f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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IWSmXAhRr129epXvvvuOG264gfbt2zNp0iTuv/9+fvrpJyZMmGA57vz589xwww0YDAZefvllOnbsSGpqKsuXL+fy5ct07dqVBQsWcN999/HKK68watQoAJo0aVLlMZ06dYqHHnqIZs2aAWog+sQTT3D+/HleffXVKp1r0qRJjBkzhjVr1jB48GDL9sOHD7Njxw4++ugjQE0Rvv3227n99tuZPn06rq6unD59mjVr1lR5/AAGg4HTp08TGBhotf2jjz6iVatWvPfee3h7e9OyZUvWrl3L8OHD6dmzJ59//jk+Pj58//333H777eTk5Fj+UY6PjycuLo7w8HAWLlyIu7s7n376KYsXL77meAoKChgxYgQbN25kypQpxMbGUlBQwLZt2zhz5gx9+vRh69atxMbGEhMTY0lj9fb2BtSAvEePHmi1Wl599VWioqLYunUrM2fO5NSpUyxYsABQv58GDx7MhQsXeOutt2jVqhW///47t99+e6Vet969e/P7779z11138dBDD9GjR48yZ04//fRTHnzwQU6cOMHSpUut9uXm5hITE8OJEyeYMWMGHTt2ZOPGjbz11lvs3buX33//3XLsxIkT+eabb5g8eTKvv/46zs7O7Nmzp9y1yIqi8Pzzz/PRRx/x73//u9prcE+cOIGTkxONGjWybPv555/ZuHEjr776KiEhIQQFBZV532v9LAYHB5OTk8PAgQM5d+6c5ZiDBw/y6quvsn//flatWoVGo6nW931oaChbt27l0UcfJT09nW+//RZQszOq8tpX5TmbLVmyhMuXLzNp0iRatmxJv379+OGHH5g7d265b+QUf8179+7N/fffj4+PD6dOneKDDz6gX79+7N+/35KxcM8997Bnzx7efPNNWrVqxZUrV9izZw+pqanlnnvOnDk8//zzTJ8+nVdeeQWA7Oxs0tLSeO6552jcuDH5+fmsWrWKsWPHsmDBAu69994KxwuwZcsWDh06xPPPP4+/vz+33HIL3377LQkJCURERFzz/vn5+dx000089NBDvPTSS6XeQKuMl19+mb59+/Lvf/+bjIwMXnzxRW688UYOHTqETqcD4JtvvuHee+9lzJgxLFq0CL1ezxdffMGwYcNYvny55c2A6ry2QggBgCKEEPXYV199pQDK559/riiKomRmZiqenp5K//79rY6bNGmSotfrlfj4+HLP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585V+7FcXV0tM9XFlSyy9P3336PX6/ntt99wdXW1bL+eNl1t27alR48eLFiwgIceeogFCxYQFhZmCf7NxowZw5gxY8jLy2Pbtm289dZbjB8/nvDwcHr37l3hY/j4+NC9e/drjqVksBIQEADA1KlTGTt2bJn3ad26NaD+c1zWP6+V+Yc2MDCQTZs2YTKZqhWYBwQE0LFjR958880y94eFhVnGuGPHjmqNsTwajYann36a119/nQMHDlzzeH9/f7Zv346iKFavd3JyMgUFBZbXPDAwEKPRSFJS0jWDw969e3PrrbcyefJkAD777LNKv46urq7V+t4oT2V+FgMCAsotCmbeb3Y93/clVfa1N6vsczaZTJYe9+X9nMyfP7/ctmwHDhxg3759LFy40KpWxvHjx0sdGxAQwNy5c5k7dy5nzpzhl19+4aWXXiI5OZlly5ZZjmvWrBkffPAB//jHPxg7diw//fST1e+sb775hoiICH744Qer51my4F15zAU4p0yZwpQpU8rcP2zYsArPUdnX93qYr+nHH39cbgV48xt1lX1thRCiJJkpF0LUS0ajkUWLFhEVFcXatWtLfTz77LMkJiZaUnxHjBjB2rVrK0wfLjmDUlx4eDhHjx61+oc0NTXVUvnbTKPR4OTkZCkgZD7f119/fV3P97777mP79u1s2rSJX3/9lQkTJlg9RsnnMXDgQN555x0AS0/k2tC6dWtatmzJvn376N69e5kf5tZgMTExrF692qrCttFo5Icffrjm44wYMYLc3FxLYFMeFxeXMq/f6NGjLS2wyhqjOSiPiYkhMzOzVPXyyhSjA0hMTCxz+4ULF8jIyLA8TkVjjYuLIysrq9QbOV999ZVlP2BJva9sL/kJEybw/fffW4p0GY3GSt2vplXmZ3H06NGcOHECf3//Mq9XyY4HUDPf95V97atq+fLlnDt3jscee6zM31ft2rXjq6++KreQmTk4Ldl+8IsvvqjwcZs1a8bjjz/OkCFD2LNnT6n9Q4cOZfny5WzYsIHRo0dbVbbXaDQ4OztbBcZJSUmVqr5+6NAhtm7dyi233FLm842Li+N///ufXRRI69u3L76+vsTHx5f7O6xkxhBc+7UVQojiZKZcCFEv/fnnn1y4cIF33nmnzFZl7du355NPPmHevHmMHj2a119/nT///JMBAwbw8ssv06FDB65cucKyZct45plnaNOmDVFRUbi5ufHtt98SHR2Np6cnYWFhhIWFcc899/DFF19w991388ADD5Camsrs2bNLpVaOGjWKDz74gPHjx/Pggw+SmprKe++9d929vO+8806eeeYZ7rzzTvLy8kpVzX711Vc5d+4ccXFxNGnShCtXrvDhhx+i1+sZOHDgdT32tXzxxReMGDGCYcOGMXHiRBo3bkxaWhqHDh1iz549/PTTTwC88sor/PLLL8TGxvLqq6/i7u7O//3f/5VqcVWWO++8kwULFvDwww9z5MgRYmJiMJlMbN++nejoaEtNgQ4dOrBu3Tp+/fVXQkND8fLyonXr1rz++uusXLmSPn368OSTT9K6dWtyc3M5deoUf/zxB59//jlNmjTh3nvvZc6cOdx77728+eabtGzZkj/++IPly5dX6rV48MEHuXLlCrfccgvt27dHp9Nx+PBh5syZg1ar5cUXX7Qc26FDB5YsWcJnn31Gt27d0Gq1dO/enXvvvZf/+7//Y8KECZw6dYoOHTqwadMmZs2axciRIy1V+Pv3788999zDzJkzuXjxIqNHj8bFxYW//voLd3d3nnjiiVLjGzduHO7u7owbN87SuaCsgKM2VeZnccqUKfz3v/9lwIABPP3003Ts2BGTycSZM2dYsWIFzz77LD179qzx7/vKvvZVNW/ePJycnHj55Zet3pgxe+ihh3jyySf5/fffGTNmTKn95t9PL730Eoqi4Ofnx6+//loqBTw9PZ2YmBjGjx9PmzZt8PLyYufOnSxbtqzcGfp+/fqxevVqhg8fztChQ/njjz/w8fFh9OjRLFmyhEcffZRx48Zx9uxZ3njjDUJDQ63qdZT3fAFeeOEFevToUWp/ZmYmq1ev5ptvvuGpp56q8Fy1zdPTk48//pgJEyaQlpbGuHHjCAoKIiUlhX379pGSksJnn31WrddWCCEsbFdjTgghas/NN9+sODs7l6poXdwdd9yhODk5Wap9nz17Vpk0aZISEhKi6PV6JSwsTLntttuUixcvWu7z3XffKW3atFH0en2pqtGLFi1SoqOjFVdXV6Vt27bKDz/8UGYV4Pnz5yutW7dWXFxclMjISOWtt95S5s2bd80K49cyfvx4BVD69u1bat9vv/2mjBgxQmncuLHi7OysBAUFKSNHjlQ2btx4zfOaq3NXxFx9/d133y1z/759+5TbbrtN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GUnPnj2rNF6znJwcYmJi6N+/Pxs2bMDJyYmZM2cyfPhw/v77b7RaLTfffDMPPPAA3333Hfn5+ezYscMyPiGEEEIIUQ+tfQu0Ohj4Qul962eDyQgxU+t+XKJBkaDcQc2bN4+7774bgOHDh5OVlcXq1ast67XffPNN7rjjDmbMmGG5T6dOnQDw8/NDp9Ph5eVFSEhIlR43NjbW6usvvviCRo0asX79ekaPHl2pc0yaNIl3332XdevWERMTA6ip62PHjqVRo0Y0atSI5557znL8E088wbJly/jpp5+qHZR///33aLVa/v3vf1sC7QULFuDr68u6devo3r076enpjB49mqioKACio6Or9VhCCCGEEMJBaHWw9k31dp+ni7avn61uj5lmm3GJBkWC8hLc9DriXx92zeN2JKQxccHOax638L4b6BHhd83jXHQaMnOveRgAR44cYceOHSxZsgRQZ59vv/125s+fbwnK9+7dW+VZ8MpITk7m1VdfZc2aNVy8eBGj0UhOTk6V1rS3adOGPn36MH/+fGJiYjhx4gQbN25kxQo1q8BoNPL222/zww8/cP78efLy8sjLy8PDw6Pa4969ezfHjx8vNZufm5vLiRMnGDp0KBMnTmTYsGEMGTKEwYMHc9tttxEaGlrtxxRCCCGEEHbOPEO+9k20WcmEXdaj/W0Z7FsMA54vewZdiBomQXkJGo0Gd+drvyz9WwYS6uNKUnpumevKNUCIjyv9Wwai0147BdpkMlV6jPPmzaOgoIDGjRtbtimKgl6v5/LlyzRq1Ag3N7dKn89Mq9WWWttuMBisvp44cSIpKSnMnTuX5s2b4+LiQu/evcnPz6/SY02ePJnHH3+c//u//2PBggU0b96cuLg4QE2JnzNnDnPnzqVDhw54eHgwZcqUCh9Do9FUOHaTyUS3bt349ttvS903MDAQUGfOn3zySZYtW8YPP/zAK6+8wsqVK+nVq1eVnpsQQgghhHAgA18AkxHd+re5ofj2De/C9n+BV0jhR2g5n0PAycVWoxf1gATl1aTTanjtxrY88s0eNGAVmJtD8NdubFupgLwqCgoK+Oqrr3j//fcZOnSo1b5bbrmFb7/9lscff5yOHTuyevVq7rvvvjLP4+zsjNFotNoWGBhIUlISiqJYUrz37t1rdczGjRv59NNPGTlyJABnz57l0qVLVX4et912G089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",
      "text/plain": [
       "<Figure size 1200x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Step 7: Plot Actual vs Predicted values with quarters\n",
    "quarters = alaska_airlines_data[\"YEAR\"].astype(str) + \" Q\" + alaska_airlines_data[\"QUARTER\"].astype(str)\n",
    "years_quarters = quarters.iloc[-len(y_test_unscaled):].values\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(years_quarters, y_test_unscaled, label='Actual Values', marker='o')\n",
    "plt.plot(years_quarters, y_pred, label='Predicted Values', marker='x')\n",
    "plt.title('Actual vs Predicted Stock Prices for Alaska Airlines')\n",
    "plt.xlabel('Year and Quarter')\n",
    "plt.xticks(rotation=45)\n",
    "plt.ylabel('Stock Price')\n",
    "plt.legend()\n",
    "plt.grid()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d37f9e0c-c367-4aa4-bdf1-5dc01423fefd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved as 'Alaska_Airlines_lstm_model.h5'\n"
     ]
    }
   ],
   "source": [
    "# Save the model\n",
    "model.save('Alaska_Airlines_lstm_model.h5')\n",
    "print(\"Model saved as 'Alaska_Airlines_lstm_model.h5'\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "62952fcc-25fc-41b5-bcf3-c592f9558100",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 188ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 15ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 15ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 16ms/step\n",
      "Data exported to 'allegiant_forecast.csv'\n"
     ]
    }
   ],
   "source": [
    "# Start forecasting using the last sequence\n",
    "forecast_steps = 4  # Number of future quarters to predict\n",
    "last_sequence = X[-1]\n",
    "predictions_scaled = []\n",
    "\n",
    "# Generate predictions iteratively\n",
    "current_sequence = last_sequence.copy()\n",
    "for _ in range(forecast_steps):\n",
    "    pred_scaled = model.predict(current_sequence[np.newaxis, :, :])\n",
    "    predictions_scaled.append(pred_scaled[0, 0])\n",
    "    next_step = np.full((1, current_sequence.shape[1]), pred_scaled[0, 0])\n",
    "    current_sequence = np.vstack([current_sequence[1:], next_step])\n",
    "\n",
    "# Inverse transform predictions to get actual values\n",
    "predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))\n",
    "\n",
    "# Prepare results for visualization\n",
    "forecast_quarters = []\n",
    "current_year = alaska_airlines_data[\"YEAR\"].iloc[-1]\n",
    "current_quarter = alaska_airlines_data[\"QUARTER\"].iloc[-1]\n",
    "\n",
    "for i in range(1, forecast_steps + 1):\n",
    "    current_quarter += 1\n",
    "    if current_quarter > 4:\n",
    "        current_quarter = 1\n",
    "        current_year += 1\n",
    "    forecast_quarters.append(f\"{current_year} Q{current_quarter}\")\n",
    "\n",
    "forecast_results = pd.DataFrame({\n",
    "    \"Quarter\": forecast_quarters,\n",
    "    \"Predicted_Close\": predictions.flatten()\n",
    "})\n",
    "\n",
    "# Combine actual and forecasted data\n",
    "combined_df = pd.concat([\n",
    "    pd.DataFrame({\n",
    "        'date': alaska_airlines_data[\"YEAR\"].astype(str) + \" Q\" + alaska_airlines_data[\"QUARTER\"].astype(str),\n",
    "        'price': target.values,\n",
    "        'type': 'actual'\n",
    "    }),\n",
    "    pd.DataFrame({\n",
    "        'date': forecast_results[\"Quarter\"],\n",
    "        'price': forecast_results[\"Predicted_Close\"].values,\n",
    "        'type': 'forecast'\n",
    "    })\n",
    "], ignore_index=True)\n",
    "\n",
    "# Export to CSV for D3 visualization\n",
    "combined_df.to_csv('united_airlines_forecast.csv', index=False)\n",
    "print(\"Data exported to 'allegiant_forecast.csv'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "c2783e58-43e1-4e6c-9ba8-06b263c5543a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "fill": "tozeroy",
         "fillcolor": "rgba(34, 94, 168, 0.8)",
         "line": {
          "color": "rgba(34, 94, 168, 1)"
         },
         "mode": "lines",
         "name": "Actual Close Prices",
         "type": "scatter",
         "x": [
          "1995 Q1",
          "1995 Q2",
          "1995 Q3",
          "1995 Q4",
          "1996 Q1",
          "1996 Q2",
          "1996 Q3",
          "1996 Q4",
          "1997 Q1",
          "1997 Q2",
          "1997 Q3",
          "1997 Q4",
          "1998 Q1",
          "1998 Q2",
          "1998 Q3",
          "1998 Q4",
          "1999 Q1",
          "1999 Q2",
          "1999 Q3",
          "1999 Q4",
          "2000 Q1",
          "2000 Q2",
          "2000 Q3",
          "2000 Q4",
          "2001 Q1",
          "2001 Q2",
          "2001 Q3",
          "2001 Q4",
          "2002 Q1",
          "2002 Q2",
          "2002 Q3",
          "2002 Q4",
          "2003 Q1",
          "2003 Q2",
          "2003 Q3",
          "2003 Q4",
          "2004 Q1",
          "2004 Q2",
          "2004 Q3",
          "2004 Q4",
          "2005 Q1",
          "2005 Q2",
          "2005 Q3",
          "2005 Q4",
          "2006 Q1",
          "2006 Q2",
          "2006 Q3",
          "2006 Q4",
          "2007 Q1",
          "2007 Q2",
          "2007 Q3",
          "2007 Q4",
          "2008 Q1",
          "2008 Q2",
          "2008 Q3",
          "2008 Q4",
          "2009 Q1",
          "2009 Q2",
          "2009 Q3",
          "2009 Q4",
          "2010 Q1",
          "2010 Q2",
          "2010 Q3",
          "2010 Q4",
          "2011 Q1",
          "2011 Q2",
          "2011 Q3",
          "2011 Q4",
          "2012 Q1",
          "2012 Q2",
          "2012 Q3",
          "2012 Q4",
          "2013 Q1",
          "2013 Q2",
          "2013 Q3",
          "2013 Q4",
          "2014 Q1",
          "2014 Q2",
          "2014 Q3",
          "2014 Q4",
          "2015 Q1",
          "2015 Q2",
          "2015 Q3",
          "2015 Q4",
          "2016 Q1",
          "2016 Q2",
          "2016 Q3",
          "2016 Q4",
          "2017 Q1",
          "2017 Q2",
          "2017 Q3",
          "2017 Q4",
          "2018 Q1",
          "2018 Q2",
          "2018 Q3",
          "2018 Q4",
          "2019 Q1",
          "2019 Q2",
          "2019 Q3",
          "2019 Q4",
          "2020 Q1",
          "2020 Q2",
          "2020 Q3",
          "2020 Q4",
          "2021 Q1",
          "2021 Q2",
          "2021 Q3",
          "2021 Q4",
          "2022 Q1",
          "2022 Q2",
          "2022 Q3",
          "2022 Q4",
          "2023 Q1",
          "2023 Q2",
          "2023 Q3",
          "2023 Q4",
          "2024 Q1"
         ],
         "y": [
          3.96875,
          4.5625,
          3.9375,
          4.0625,
          6.6875,
          6.84375,
          5.34375,
          5.25,
          6.40625,
          6.40625,
          8.21875,
          9.6875,
          13.546875,
          13.640625,
          8.515625,
          11.0625,
          11.875,
          10.4375,
          10.171875,
          8.78125,
          7.515625,
          6.78125,
          6,
          7.4375,
          6.425,
          7.225,
          4.9925,
          7.275,
          8.3225,
          6.525,
          4.425,
          5.4125,
          3.915,
          5.3625,
          6.955,
          6.8225,
          6.165,
          5.9675,
          6.195,
          8.3725,
          7.36,
          7.4375,
          7.265,
          8.93,
          8.8625,
          9.855,
          9.51,
          9.875,
          9.525,
          6.965,
          5.7725,
          6.2525,
          4.905,
          3.835,
          5.0975,
          7.3125,
          4.3925,
          4.565,
          6.6975,
          8.64,
          10.3075,
          11.2375,
          12.7575,
          14.1725,
          15.855,
          17.115,
          14.0725,
          18.772499,
          17.91,
          17.950001,
          17.530001,
          21.545,
          31.98,
          26,
          31.309999,
          36.685001,
          46.654999,
          47.525002,
          43.540001,
          59.759998,
          66.18,
          64.43,
          79.449997,
          80.510002,
          82.019997,
          58.290001,
          65.860001,
          88.730003,
          92.220001,
          89.760002,
          76.269997,
          73.510002,
          61.959999,
          60.389999,
          68.860001,
          60.849998,
          56.119999,
          63.91,
          64.910004,
          67.75,
          28.469999,
          36.259998,
          36.630001,
          52,
          69.209999,
          60.310001,
          58.599998,
          52.099998,
          58.009998,
          40.049999,
          39.150002,
          42.939999,
          41.959999,
          53.18,
          37.080002,
          39.07,
          42.990002
         ]
        },
        {
         "fill": "tozeroy",
         "fillcolor": "rgba(65, 182, 196, 0.8)",
         "line": {
          "color": "rgba(65, 182, 196, 1)"
         },
         "mode": "lines+markers",
         "name": "Forecasted Close Prices",
         "type": "scatter",
         "x": [
          "2024 Q2",
          "2024 Q3",
          "2024 Q4",
          "2025 Q1"
         ],
         "y": [
          49.79650115966797,
          50.062286376953125,
          48.975948333740234,
          47.03936004638672
         ]
        }
       ],
       "layout": {
        "autosize": true,
        "legend": {
         "title": {
          "text": "Legend"
         }
        },
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "white",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "white",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "#C8D4E3",
             "linecolor": "#C8D4E3",
             "minorgridcolor": "#C8D4E3",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "#C8D4E3",
             "linecolor": "#C8D4E3",
             "minorgridcolor": "#C8D4E3",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "heatmapgl": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmapgl"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "white",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "#C8D4E3"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "white",
          "polar": {
           "angularaxis": {
            "gridcolor": "#EBF0F8",
            "linecolor": "#EBF0F8",
            "ticks": ""
           },
           "bgcolor": "white",
           "radialaxis": {
            "gridcolor": "#EBF0F8",
            "linecolor": "#EBF0F8",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "white",
            "gridcolor": "#DFE8F3",
            "gridwidth": 2,
            "linecolor": "#EBF0F8",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "#EBF0F8"
           },
           "yaxis": {
            "backgroundcolor": "white",
            "gridcolor": "#DFE8F3",
            "gridwidth": 2,
            "linecolor": "#EBF0F8",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "#EBF0F8"
           },
           "zaxis": {
            "backgroundcolor": "white",
            "gridcolor": "#DFE8F3",
            "gridwidth": 2,
            "linecolor": "#EBF0F8",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "#EBF0F8"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "#DFE8F3",
            "linecolor": "#A2B1C6",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "#DFE8F3",
            "linecolor": "#A2B1C6",
            "ticks": ""
           },
           "bgcolor": "white",
           "caxis": {
            "gridcolor": "#DFE8F3",
            "linecolor": "#A2B1C6",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "#EBF0F8",
           "linecolor": "#EBF0F8",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "#EBF0F8",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "#EBF0F8",
           "linecolor": "#EBF0F8",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "#EBF0F8",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Alaska Airlines Stock Price Forecast for 4 Quarters"
        },
        "xaxis": {
         "autorange": true,
         "range": [
          0,
          120
         ],
         "tickangle": 45,
         "title": {
          "text": "Year and Quarter"
         },
         "type": "category"
        },
        "yaxis": {
         "autorange": true,
         "range": [
          0,
          97.0736852631579
         ],
         "title": {
          "text": "Stock Price (Close)"
         },
         "type": "linear"
        }
       }
      },
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABE0AAAFoCAYAAACixgUDAAAAAXNSR0IArs4c6QAAIABJREFUeF7snQd8FVXa/3/pvUAg9GahCNgr2LCgYkHddV1W2aLYUbErusqqG8UVbOhad3VZ/65d7GURy2JXVEBAeg8hgdSbnvzfZ+KEuZO5d8qZm9zk/ubz7ueV3HPOnPM9z5Tzm+c8T1xzc3MzeJAACZAACZAACZAACZAACZAACZAACZAACQQRiKNoQosgARIgARIgARIgARIgARIgARIgARIggbYEKJrQKkiABEiABEiABEiABEiABEiABEiABEjAggBFE5oFCZAACZAACZAACZAACZAACZAACZAACVA0oQ2QAAmQAAmQAAmQAAmQAAmQAAmQAAmQgDMC9DRxxomlSIAESIAESIAESIAESIAESIAESIAEYowARZMYm3AOlwRIgARIgARIgARIgARIgARIgARIwBkBiibOOLEUCZAACZAACZAACZAACZAACZAACZBAjBGgaBJjE87hkgAJkAAJkAAJkAAJkAAJkAAJkAAJOCNA0cQZJ5YiARIgARIgARIgARIgARIgARIgARKIMQIUTWJswjlcEiABEiABEiABEiABEiABEiABEiABZwQomjjjxFIkQAIkQAIkQAIkQAIkQAIkQAIkQAIxRoCiSYxNOIdLAiRAAiRAAiRAAiRAAiRAAiRAAiTgjABFE2ecWIoESIAESIAESIAESIAESIAESIAESCDGCFA0ibEJ53BJgARIgARIgARIgARIgARIgARIgAScEaBo4owTS5EACZAACZAACZAACZAACZAACZAACcQYAYomMTbhHC4JkAAJkAAJkAAJkAAJkAAJkAAJkIAzAhRNnHFiKRIgARIgARIgARIgARIgARIgARIggRgjQNEkxiacwyUBEiABEiABEiABEiABEiABEiABEnBGgKKJM04sRQIkQAIkQAIkQAIkQAIkQAIkQAIkEGMEKJrE2IRzuCRAAiRAAiRAAiRAAiRAAiRAAiRAAs4IUDRxxomlSIAESIAESIAESIAESIAESIAESIAEYowARZMYm3AOlwRIgARIgARIgARIgARIgARIgARIwBkBiibOOLEUCZAACZAACZAACZAACZAACZAACZBAjBGgaBJjE87hkgAJkAAJkAAJkAAJkAAJkAAJkAAJOCNA0cQZ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KqHOHW5Bb95CoLdhzv5jtUg0L5KtxBk3wonThZzb+dSvHeN2CvNXf6vYKTJXMvfmjFNWtysnookV23DZc8yeTeYFu1W2FGNf7cY9atiQNmKRcQugvmXHyv6denmFu7XbBbVdsHBRq+daKLt0uz0nVH+c2p6xfijPMP36NG9Ps2JmxduqnHFLjXHe7Z4Ldl56obwD9fuU20CwTq4Dfx73bIUESIAESIAESIAESMALgS4tmpgXt1aAzC/+Vi/MVnEnpC0J4GlMC6svCPTzmAWOUAseY/wRffFvdomXl34JHmuVZcY4rlALJfPYjeMsKi5tE8vB7SJbZXuO3jer+Bxmt387xuEuglDbDMwLaav+WMWi0FP4SvlQqYytYssYBR4r0STUYs+J/XpZ4IZa5FnFAzIvakPFDHISm8XNYtE8725SxepMnPbVbtzm38VG9UC+xiCyVnNvZWtmuwwnAIYTTWSc5mvIyi47UjSRPlpxCbe9xylvc9wRuWb1bZBOY5pI/8ztmBna3aeciDLGoOJOhHAvD3fWIQESIAESIAESIAES8IdAlxZN/EHEVkiABEiABCJFwC8RJ1L9Y7skQAIkQAIkQAIkQAKxTYCiSWzPP0dPAiRAAh1KwI3HUYd2lCcnARIgARIgARIgARKISQIUTWJy2jloEiABEogeAuYtSk62d0VP79kTEiABEiABEiABEiCBrkyAoklXnl2OjQRIgARIgARIgARIgARIgARIgARIwDMBiiae0bEiCZAACZAACZAACZAACZAACZAACZBAVyZA0aQrzy7HRgIkQAIkQAIkQAIkQAIkQAIkQAIk4JkARRPP6FiRBEiABEiABEiABEiABEiABEiABEigKxOgaNKVZ5djIwESIAESIAESIAESIAESIAESIAES8EyAoolndKxIAiRAAiRAAiRAAiRAAiRAAiRAAiTQlQlQNOnKs8uxkQAJkAAJkAAJkAAJkAAJkAAJkAAJeCZA0cQzOlYkARIgARIgARIgARIgARIgARIgARLoygQomnTl2eXYSIAESIAESIAESIAESIAESIAESIAEPBOgaOIZHSuSAAmQAAmQAAmQAAmQAAmQAAmQAAl0ZQIUTbry7HJsJEACJEACJEACJEACJEACJEACJEACnglQNPGMjhVJgARIgARIgARIgARIgARIgARIgAS6MgGKJl15djk2EiABEiABEiABEiABEiABEiABEiABzwQomnhGx4okQAIkQAIkQAIkQAIkQAIkQAIkQAJdmQBFk648uxwbCZAACZAACZAACZAACZAACZAACZCAZwIUTTyjY0USIAESIAESIAESIAESIAESIAESIIGuTICiSVeeXY6NBEiABEiABEiABEiABEiABEiABEjAMwGKJp7RsSIJkAAJkAAJkAAJkAAJkAAJkAAJkEBXJkDRpCvPLsdGAiRAAiRAAiRAAiRAAiRAAiRAAiTgmQBFE8/oWJEESIAESIAESIAESIAESIAESIAESKArE6Bo0pVnl2MjARIgARIgARIgARIgARIgARIgARLwTICiiWd0rEgCJEACJEACJEACJEACJEACJEACJNCVCVA06cqzy7GRAAmQAAmQAAmQAAmQAAmQAAmQAAl4JkDRxDM6ViQBEiABEiABEiABEiABEiABEiABEujKBCiadOXZ5dhIgARIgARIgARIgARIgARIgARIgAQ8E6Bo4hkdK5IACZAACZAACZAACZAACZAACZAACXRlAhRNuvLscmwkQAIkQAIkQAIkQAIkQAIkQAIkQAKeCVA08YyOFUmABEiABEiABEiABEiABEiABEiABLoyAYomXXl2OTYSIAESIAESIAESIAESIAESIAESIAHPBCiaeEbHiiRAAiRAAiRAAiRAAiRAAiRAAiRAAl2ZAEWTrjy7HBsJkAAJkAAJkAAJkAAJkAAJkAAJkIBnAhRNPKNjRRIgARIgARIgARIgARIgARIgARIgga5MgKJJV55djo0ESIAESIAESIAESIAESIAESIAESMAzAYomntGxIgmQAAmQAAmQAAmQAAmQAAmQAAmQQFcmQNGkK88ux0YCJEACJEACJEACJEACJEACJEACJOCZAEUTz+hYkQRIgARIgARIgARIgARIgARIgARIoCsToGjSlWeXYyMBEiABEiABEiABEiABEiABEiABEvBMgKKJZ3SsSAIkQAIkQAIkQAIkQAIkQAIkQAIk0JUJUDTpyrPLsZEACZAACZAACZAACZAACZAACZAACXgmQNHEMzpWJAESIAESIAESIAESIAESIAESIAES6MoEKJp05dnl2EiABEiABEiABEiABEiABEiABEiABDwToGjiGR0rkgAJkAAJkAAJkAAJkAAJkAAJkAAJdGUCFE268uxybCRA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",
      "text/html": [
       "<div>                            <div id=\"daa68f9e-e556-4884-b858-8daba8cc5351\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                require([\"plotly\"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"daa68f9e-e556-4884-b858-8daba8cc5351\")) {                    Plotly.newPlot(                        \"daa68f9e-e556-4884-b858-8daba8cc5351\",                        [{\"fill\":\"tozeroy\",\"fillcolor\":\"rgba(34, 94, 168, 0.8)\",\"line\":{\"color\":\"rgba(34, 94, 168, 1)\"},\"mode\":\"lines\",\"name\":\"Actual Close Prices\",\"x\":[\"1995 Q1\",\"1995 Q2\",\"1995 Q3\",\"1995 Q4\",\"1996 Q1\",\"1996 Q2\",\"1996 Q3\",\"1996 Q4\",\"1997 Q1\",\"1997 Q2\",\"1997 Q3\",\"1997 Q4\",\"1998 Q1\",\"1998 Q2\",\"1998 Q3\",\"1998 Q4\",\"1999 Q1\",\"1999 Q2\",\"1999 Q3\",\"1999 Q4\",\"2000 Q1\",\"2000 Q2\",\"2000 Q3\",\"2000 Q4\",\"2001 Q1\",\"2001 Q2\",\"2001 Q3\",\"2001 Q4\",\"2002 Q1\",\"2002 Q2\",\"2002 Q3\",\"2002 Q4\",\"2003 Q1\",\"2003 Q2\",\"2003 Q3\",\"2003 Q4\",\"2004 Q1\",\"2004 Q2\",\"2004 Q3\",\"2004 Q4\",\"2005 Q1\",\"2005 Q2\",\"2005 Q3\",\"2005 Q4\",\"2006 Q1\",\"2006 Q2\",\"2006 Q3\",\"2006 Q4\",\"2007 Q1\",\"2007 Q2\",\"2007 Q3\",\"2007 Q4\",\"2008 Q1\",\"2008 Q2\",\"2008 Q3\",\"2008 Q4\",\"2009 Q1\",\"2009 Q2\",\"2009 Q3\",\"2009 Q4\",\"2010 Q1\",\"2010 Q2\",\"2010 Q3\",\"2010 Q4\",\"2011 Q1\",\"2011 Q2\",\"2011 Q3\",\"2011 Q4\",\"2012 Q1\",\"2012 Q2\",\"2012 Q3\",\"2012 Q4\",\"2013 Q1\",\"2013 Q2\",\"2013 Q3\",\"2013 Q4\",\"2014 Q1\",\"2014 Q2\",\"2014 Q3\",\"2014 Q4\",\"2015 Q1\",\"2015 Q2\",\"2015 Q3\",\"2015 Q4\",\"2016 Q1\",\"2016 Q2\",\"2016 Q3\",\"2016 Q4\",\"2017 Q1\",\"2017 Q2\",\"2017 Q3\",\"2017 Q4\",\"2018 Q1\",\"2018 Q2\",\"2018 Q3\",\"2018 Q4\",\"2019 Q1\",\"2019 Q2\",\"2019 Q3\",\"2019 Q4\",\"2020 Q1\",\"2020 Q2\",\"2020 Q3\",\"2020 Q4\",\"2021 Q1\",\"2021 Q2\",\"2021 Q3\",\"2021 Q4\",\"2022 Q1\",\"2022 Q2\",\"2022 Q3\",\"2022 Q4\",\"2023 Q1\",\"2023 Q2\",\"2023 Q3\",\"2023 Q4\",\"2024 Q1\"],\"y\":[3.96875,4.5625,3.9375,4.0625,6.6875,6.84375,5.34375,5.25,6.40625,6.40625,8.21875,9.6875,13.546875,13.640625,8.515625,11.0625,11.875,10.4375,10.171875,8.78125,7.515625,6.78125,6.0,7.4375,6.425,7.225,4.9925,7.275,8.3225,6.525,4.425,5.4125,3.915,5.3625,6.955,6.8225,6.165,5.9675,6.195,8.3725,7.36,7.4375,7.265,8.93,8.8625,9.855,9.51,9.875,9.525,6.965,5.7725,6.2525,4.905,3.835,5.0975,7.3125,4.3925,4.565,6.6975,8.64,10.3075,11.2375,12.7575,14.1725,15.855,17.115,14.0725,18.772499,17.91,17.950001,17.530001,21.545,31.98,26.0,31.309999,36.685001,46.654999,47.525002,43.540001,59.759998,66.18,64.43,79.449997,80.510002,82.019997,58.290001,65.860001,88.730003,92.220001,89.760002,76.269997,73.510002,61.959999,60.389999,68.860001,60.849998,56.119999,63.91,64.910004,67.75,28.469999,36.259998,36.630001,52.0,69.209999,60.310001,58.599998,52.099998,58.009998,40.049999,39.150002,42.939999,41.959999,53.18,37.080002,39.07,42.990002],\"type\":\"scatter\"},{\"fill\":\"tozeroy\",\"fillcolor\":\"rgba(65, 182, 196, 0.8)\",\"line\":{\"color\":\"rgba(65, 182, 196, 1)\"},\"mode\":\"lines+markers\",\"name\":\"Forecasted Close Prices\",\"x\":[\"2024 Q2\",\"2024 Q3\",\"2024 Q4\",\"2025 Q1\"],\"y\":[49.79650115966797,50.062286376953125,48.975948333740234,47.03936004638672],\"type\":\"scatter\"}],                        {\"template\":{\"data\":{\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"white\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"#C8D4E3\",\"linecolor\":\"#C8D4E3\",\"minorgridcolor\":\"#C8D4E3\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"choropleth\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"choropleth\"}],\"contourcarpet\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"contourcarpet\"}],\"contour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"contour\"}],\"heatmapgl\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmapgl\"}],\"heatmap\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"heatmap\"}],\"histogram2dcontour\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2dcontour\"}],\"histogram2d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"histogram2d\"}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"mesh3d\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"type\":\"mesh3d\"}],\"parcoords\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"parcoords\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}],\"scatter3d\":[{\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatter3d\"}],\"scattercarpet\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattercarpet\"}],\"scattergeo\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergeo\"}],\"scattergl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattergl\"}],\"scattermapbox\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scattermapbox\"}],\"scatterpolargl\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolargl\"}],\"scatterpolar\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterpolar\"}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"scatterternary\":[{\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"type\":\"scatterternary\"}],\"surface\":[{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"type\":\"surface\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}]},\"layout\":{\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"autotypenumbers\":\"strict\",\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]],\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]},\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"geo\":{\"bgcolor\":\"white\",\"lakecolor\":\"white\",\"landcolor\":\"white\",\"showlakes\":true,\"showland\":true,\"subunitcolor\":\"#C8D4E3\"},\"hoverlabel\":{\"align\":\"left\"},\"hovermode\":\"closest\",\"mapbox\":{\"style\":\"light\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"white\",\"polar\":{\"angularaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"radialaxis\":{\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\"}},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"yaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"},\"zaxis\":{\"backgroundcolor\":\"white\",\"gridcolor\":\"#DFE8F3\",\"gridwidth\":2,\"linecolor\":\"#EBF0F8\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"#EBF0F8\"}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"ternary\":{\"aaxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"},\"bgcolor\":\"white\",\"caxis\":{\"gridcolor\":\"#DFE8F3\",\"linecolor\":\"#A2B1C6\",\"ticks\":\"\"}},\"title\":{\"x\":0.05},\"xaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2},\"yaxis\":{\"automargin\":true,\"gridcolor\":\"#EBF0F8\",\"linecolor\":\"#EBF0F8\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"#EBF0F8\",\"zerolinewidth\":2}}},\"xaxis\":{\"title\":{\"text\":\"Year and Quarter\"},\"tickangle\":45},\"title\":{\"text\":\"Alaska Airlines Stock Price Forecast for 4 Quarters\"},\"yaxis\":{\"title\":{\"text\":\"Stock Price (Close)\"}},\"legend\":{\"title\":{\"text\":\"Legend\"}}},                        {\"responsive\": true}                    ).then(function(){\n",
       "                            \n",
       "var gd = document.getElementById('daa68f9e-e556-4884-b858-8daba8cc5351');\n",
       "var x = new MutationObserver(function (mutations, observer) {{\n",
       "        var display = window.getComputedStyle(gd).display;\n",
       "        if (!display || display === 'none') {{\n",
       "            console.log([gd, 'removed!']);\n",
       "            Plotly.purge(gd);\n",
       "            observer.disconnect();\n",
       "        }}\n",
       "}});\n",
       "\n",
       "// Listen for the removal of the full notebook cells\n",
       "var notebookContainer = gd.closest('#notebook-container');\n",
       "if (notebookContainer) {{\n",
       "    x.observe(notebookContainer, {childList: true});\n",
       "}}\n",
       "\n",
       "// Listen for the clearing of the current output cell\n",
       "var outputEl = gd.closest('.output');\n",
       "if (outputEl) {{\n",
       "    x.observe(outputEl, {childList: true});\n",
       "}}\n",
       "\n",
       "                        })                };                });            </script>        </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import plotly.graph_objects as go\n",
    "\n",
    "# Visualization using Plotly\n",
    "fig = go.Figure()\n",
    "\n",
    "# Add actual values as an area chart with gradient\n",
    "fig.add_trace(go.Scatter(\n",
    "    x=alaska_airlines_data[\"YEAR\"].astype(str) + \" Q\" + alaska_airlines_data[\"QUARTER\"].astype(str),\n",
    "    y=target.values,\n",
    "    mode='lines',\n",
    "    fill='tozeroy',  # Fill the area below the line\n",
    "    line=dict(color='rgba(34, 94, 168, 1)'),  # Dark blue line\n",
    "    name='Actual Close Prices',\n",
    "    fillcolor='rgba(34, 94, 168, 0.8)'  # Gradient start color (darker blue)\n",
    "))\n",
    "\n",
    "# Add forecasted values as an area chart with gradient\n",
    "fig.add_trace(go.Scatter(\n",
    "    x=forecast_results[\"Quarter\"],\n",
    "    y=forecast_results[\"Predicted_Close\"],\n",
    "    mode='lines+markers',\n",
    "    fill='tozeroy',  # Fill the area below the line\n",
    "    line=dict(color='rgba(65, 182, 196, 1)'),  # Light blue line\n",
    "    name='Forecasted Close Prices',\n",
    "    fillcolor='rgba(65, 182, 196, 0.8)'  # Gradient start color (lighter blue)\n",
    "))\n",
    "\n",
    "# Update layout\n",
    "fig.update_layout(\n",
    "    title=\"Alaska Airlines Stock Price Forecast for 4 Quarters\",\n",
    "    xaxis_title=\"Year and Quarter\",\n",
    "    yaxis_title=\"Stock Price (Close)\",\n",
    "    legend_title=\"Legend\",\n",
    "    template=\"plotly_white\",\n",
    "    xaxis=dict(tickangle=45),  # Rotate x-axis labels for better readability\n",
    ")\n",
    "\n",
    "fig.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b360e3ec-f8ea-4310-b3cf-c11996d2c8e0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
