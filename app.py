import numpy as np
from flask import Flask, render_template, request
import pickle  # For loading the pre-trained model
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (assuming you have saved it as a .pkl file)
model = pickle.load(open('random_forest_model.pkl','rb'))  # Replace with your actual model path
sc=pickle.load(open('StandardScaler.pkl','rb'))
crops=json.load(open('crops.json','rb'))


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Collect the form inputs from the user
        nitrogen_ratio = float(request.form['nitrogen_ratio'])
        phosphorus_ratio = float(request.form['phosphorus_ratio'])
        potassium_ratio = float(request.form['potassium_ratio'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph_value = float(request.form['ph_value'])
        rainfall = float(request.form['rainfall'])
        
        # Create the feature array
        input_features = np.array([[nitrogen_ratio, phosphorus_ratio, potassium_ratio,
                                    temperature, humidity, ph_value, rainfall]])
        input_features=sc.transform(input_features)
        
        # Make the prediction using the model
        recommended_crop = model.predict(input_features)[0]
        recommended_crop=crops[str(int(recommended_crop))]
        
        # Return the prediction result
        return render_template('result.html', prediction=recommended_crop)

if __name__ == "__main__":
    app.run(debug=True)
