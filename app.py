from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('house_price_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            bhk = int(request.form['bhk'])
            bathrooms = int(request.form['bathrooms'])
            furnishing = request.form['furnishing']
            parking = request.form['parking']
            age = int(request.form['age'])
            proximity = request.form['proximity']
            
            furnishing_encoded = label_encoders['Furnishing Status'].transform([furnishing])[0]
            parking_encoded = label_encoders['Parking'].transform([parking])[0]
            proximity_encoded = label_encoders['Proximity'].transform([proximity])[0]
            
            input_data = np.array([[area, bhk, bathrooms, furnishing_encoded, parking_encoded, age, proximity_encoded]])
            
            result = model.predict(input_data)[0]
            prediction = round(result, 2)
        
        except Exception as e:
            prediction = f'Error: {e}'

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
