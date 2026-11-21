
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Extract relevant features from the input data
    sample = {
        'Product_Id': data['Product_Id_char'] + "6114",
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_Type': "Frozen Foods" if data['Product_Type_Category'] == "Non Perishables" else "Dairy",
        'Product_MRP': data['Product_MRP'],
        'Store_Id': "OUT004",
        'Store_Establishment_Year': 2025 - int(data['Store_Age_Years']),
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type']
    }

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])
    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': prediction})

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']
    input_data = pd.read_csv(file)

    # Map engineered columns to raw columns expected by the saved model
    if 'Product_Id' not in input_data.columns and 'Product_Id_char' in input_data.columns:
        input_data['Product_Id'] = input_data['Product_Id_char'] + "6114"
    if 'Product_Type' not in input_data.columns and 'Product_Type_Category' in input_data.columns:
        input_data['Product_Type'] = input_data['Product_Type_Category'].apply(
            lambda x: "Frozen Foods" if x == "Non Perishables" else "Dairy"
        )
    if 'Store_Id' not in input_data.columns:
        input_data['Store_Id'] = "OUT004"
    if 'Store_Establishment_Year' not in input_data.columns and 'Store_Age_Years' in input_data.columns:
        input_data['Store_Establishment_Year'] = 2025 - input_data['Store_Age_Years'].astype(int)

    # Drop target column if present in the uploaded CSV
    input_data = input_data.drop(columns=['Product_Store_Sales_Total'], errors='ignore')

    # Generate predictions
    predictions = model.predict(input_data).tolist()
    return jsonify({'predictions': predictions})
    
    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output_dict


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
