import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify, render_template

# Load the dataset
data = pd.read_csv("btc_2015_2024.csv")

# Preprocessing
# Drop irrelevant columns if any
data.drop(columns=['date'], inplace=True)

# Split features and target variable
X = data.drop(columns=['next_day_close'])  # Features
y = data['next_day_close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model and train it
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame(data, index=[0])
    # Preprocess input data if required (e.g., convert date format)
    # Make predictions
    prediction = model.predict(input_data)
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)