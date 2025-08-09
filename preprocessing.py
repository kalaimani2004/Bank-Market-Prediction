import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully!")
        return data
    except FileNotFoundError:
        print("File not found, please check the file path.")
        return None

scaler = StandardScaler()
le = LabelEncoder()


def preprocess_data(data):
    # Handle categorical columns
    categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])

    # Define numerical columns
    numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous']
    
    # Scale numerical features
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data


if __name__ == "__main__":
    filepath = r'D:\Bank Marketing predictions\data\bank-marketing.csv'  # Use raw string for filepath
    data = load_data(filepath)
    if data is not None:
        preprocessed_data = preprocess_data(data)
        if preprocessed_data is not None:
            preprocessed_data.to_csv(r'D:\Bank Marketing predictions\data\preprocessed_data.csv', index=False)
            print("Preprocessing complete and data saved!")
