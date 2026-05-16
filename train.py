import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- STEP 1: LOAD AND PREPARE DATA ---
print("Reading the dataset...")

# The CSV file uses semicolons as separators and commas for decimals, so we have to specify that.
df = pd.read_csv('dataset/Gercek_Zamanli_Uretim-01012025-01042025(in).csv', sep=';', decimal=',', encoding='utf-8')

print("Extracting features from datetime...")

# First, convert the string date into a proper pandas datetime object
df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')

# The 'Saat' column looks like '14:00'. We just want the '14' as an integer.
df['Saat_Num'] = df['Saat'].str.split(':').str[0].astype(int)

# Create new features for the model to learn from
df['Day'] = df['Tarih'].dt.day
df['Month'] = df['Tarih'].dt.month
df['DayOfWeek'] = df['Tarih'].dt.dayofweek
# If it's Saturday (5) or Sunday (6), set Is_Weekend to 1, else 0
df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
df['Hour'] = df['Saat_Num']

# We are trying to predict the total energy production (Toplam)
target = 'Toplam'
features = ['Month', 'Day', 'DayOfWeek', 'Is_Weekend', 'Hour']

X = df[features].copy()
y = df[target].copy()

# Check if there are any missing values in our target column.
# If there are, fill them with the median to avoid dropping rows and breaking the timeline.
missing_count = y.isnull().sum()
if missing_count > 0:
    print(f"Found {missing_count} missing targets. Filling with median...")
    y.fillna(y.median(), inplace=True)


# --- STEP 2: TRAIN/TEST SPLIT AND SCALING ---

# Standard 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features so the models don't get biased by larger numbers
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- STEP 3: MODEL TOURNAMENT ---

# Model 1: Baseline Linear Regression
print("\n[1] Training Linear Regression baseline...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions and calculate errors
lr_preds = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print(f"Linear Regression -> RMSE: {lr_rmse:.2f} | MAE: {lr_mae:.2f} | R2: {lr_r2:.4f}")


# Model 2: AdaBoost with Hyperparameter Tuning
print("\n[2] Running Grid Search for AdaBoost. This might take a bit...")
ada_base = AdaBoostRegressor(random_state=42)

# We want to test different combinations of these parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

# 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=ada_base, 
    param_grid=param_grid, 
    cv=5, 
    n_jobs=-1, 
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train_scaled, y_train)

# Get the best model found by the grid search
best_ada = grid_search.best_estimator_
ada_preds = best_ada.predict(X_test_scaled)

ada_rmse = np.sqrt(mean_squared_error(y_test, ada_preds))
ada_mae = mean_absolute_error(y_test, ada_preds)
ada_r2 = r2_score(y_test, ada_preds)

print(f"AdaBoost Best Params: {grid_search.best_params_}")
print(f"AdaBoost -> RMSE: {ada_rmse:.2f} | MAE: {ada_mae:.2f} | R2: {ada_r2:.4f}")


# --- STEP 4: PICK THE WINNER AND EXPORT ---

# Compare R2 scores to decide the winner
if ada_r2 > lr_r2:
    winner_model = best_ada
    winner_name = "AdaBoost Regressor"
    winner_r2 = ada_r2
    winner_rmse = ada_rmse
else:
    winner_model = lr_model
    winner_name = "Linear Regression"
    winner_r2 = lr_r2
    winner_rmse = lr_rmse

print(f"\n>>> WINNER: {winner_name} (R2: {winner_r2:.4f}) <<<")

# Extract feature importances if the model supports it (AdaBoost does)
importances = {}
if hasattr(winner_model, 'feature_importances_'):
    importances = dict(zip(features, winner_model.feature_importances_.tolist()))
elif hasattr(winner_model, 'coef_'):
    importances = dict(zip(features, np.abs(winner_model.coef_).tolist()))

# Bundle everything we need into a dictionary
export_package = {
    'model': winner_model,
    'model_name': winner_name,
    'scaler': scaler,
    'feature_names': features,
    'feature_importance': importances
}

# Save it as a pickle file for app.py to use
with open('model_package.pkl', 'wb') as file:
    pickle.dump(export_package, file)

print("\nSaved the model package to 'model_package.pkl'. Ready for the Flask app!")
