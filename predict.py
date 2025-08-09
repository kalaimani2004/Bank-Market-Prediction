import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and feature columns
model_path = r'D:\Bank Marketing predictions\model\random_forest_model.pkl'
model = joblib.load(model_path)
feature_columns_path = r'D:\Bank Marketing predictions\model\feature_columns.pkl'
feature_columns = joblib.load(feature_columns_path)

# Sample input data for a new customer prediction
new_customer_yes = pd.DataFrame([{
    'age': 35, 'job': 'manager', 'marital': 'single', 'education': 'tertiary',
    'default': 'no', 'balance': 1500, 'housing': 'yes', 'loan': 'no', 'contact': 'cellular',
    'day': 15, 'month': 'may', 'day_of_week': 'mon', 'duration': 350, 'campaign': 1,
    'pdays': 10, 'previous': 2, 'poutcome': 'success',
    'emp.var.rate': 1.4, 'cons.price.idx': 93.2, 'cons.conf.idx': -30.0, 'euribor3m': 4.857, 'nr.employed': 5191.0
}])
new_customer_no = pd.DataFrame([{
    'age': 35, 'job': 'admin.', 'marital': 'married', 'education': 'tertiary',
    'default': 'no', 'balance': 12000, 'housing': 'yes', 'loan': 'no', 'contact': 'telephone',
    'day': 15, 'month': 'jun', 'day_of_week': 'mon', 'duration': 600, 'campaign': 6,
    'pdays': 10, 'previous': 2, 'poutcome': 'success',
    'emp.var.rate': 1.2, 'cons.price.idx': 95.0, 'cons.conf.idx': -5.0, 'euribor3m': 5.0, 'nr.employed': 5200.0
}])

# Encode the new customer data
label_enc = LabelEncoder()
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']:
    new_customer_yes[col] = label_enc.fit_transform(new_customer_yes[col])

# Select only the columns used during training
new_customer = new_customer_yes[feature_columns]

# Predict
prediction = model.predict(new_customer)
print("Prediction:", "Yes" if prediction[0] == 1 else "No")


