# Energy Production Forecast System

This project is an end-to-end machine learning application that predicts future hourly **Total Energy Production** using a real-time energy production dataset. The project is designed as a **Regression** problem in accordance with the course requirements.

## How It Works

### 1. Data Preparation and Feature Engineering
The system uses real-time data from the `dataset/Gercek_Zamanli_Uretim-01012025-01042025(in).csv` file.
- **Target Variable:** `Toplam` (Total Energy Production). Since this is continuous numerical data, it is a regression prediction.
- **Feature Engineering:** New columns that algorithms can understand are derived from the `Tarih` (Date) and `Saat` (Hour) data for prediction: `Month`, `Day`, `DayOfWeek`, `Is_Weekend`, and `Hour`.
- **Transformation and Scaling:** All these created time features are scaled using `StandardScaler` to ensure the machine learning algorithms operate at optimum speed and accuracy.

### 2. Model Training and Selection (Machine Learning)
Two different models were implemented and compared:
- **Linear Regression:** The baseline algorithm that attempts to capture linear relationships between variables.
- **AdaBoost Regressor:** An ensemble algorithm that maximizes accuracy by sequentially training basic learners (usually Decision Trees). The `n_estimators` and `learning_rate` hyperparameters were optimized using `GridSearchCV` (cross-validation) on the model.
- **Evaluation Metrics:** Since a regression problem is solved rather than a classification problem, the models compete based on **RMSE**, **MAE**, and **R²** metrics. The model with the highest R² score becomes the winning model.
- **Packaging:** The winning model and the data scaler are saved to the project as `model_package.pkl`.

### 3. API and Integration (Flask)
The Flask server acts as a bridge for prediction operations:
- It receives Date and Time information (ISO 8601 format) from the user.
- It parses this string expression and converts it into the column structure required by the machine learning model.
- It performs the prediction operation with the saved model.
- It returns the total production (MWh) and feature importance order as JSON back to the UI.

### 4. User Interface (Frontend)
- **Design:** A modern, clean, and task-oriented design.
- **Analysis Screen:** When the user selects an hour and clicks, the predicted energy is instantly reflected on the screen. The Chart.js-based graph below visualizes how much weight the model gave to which time labels (e.g., Hour or whether it's a Weekend) when making this prediction.

## 🚀 How to Run?

1. **Install Dependencies:** 
   ```bash
   pip install pandas numpy scikit-learn flask flask-cors
   ```
2. **Train the Model:** 
   You can train the models by running the code below in the terminal. At the end of the training, performance metrics (RMSE, R²) are listed in the terminal.
   ```bash
   python train.py
   ```
3. **Start the API:** 
   ```bash
   python app.py
   ```
4. **Browser:** 
   Open your browser and navigate to `http://localhost:8000`. You can get your prediction by selecting any date!
