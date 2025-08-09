import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load preprocessed data
data = pd.read_csv(r'D:\Bank Marketing predictions\data\preprocessed_data.csv')

# Encode the target column 'y'
label_enc = LabelEncoder()
data['y'] = label_enc.fit_transform(data['y'])  # 'no' becomes 0, 'yes' becomes 1

# Encode other categorical columns if necessary (like 'default', 'housing', 'loan', etc.)
categorical_columns = ['default', 'housing', 'loan', 'job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
for col in categorical_columns:
    data[col] = label_enc.fit_transform(data[col])

# Split into features and target variable
X = data.drop(columns=['y'])  # Drop the target column 'y'
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Feature Importance Analysis
importances = rf_clf.feature_importances_
features = X.columns

# Create a DataFrame for visualization
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.show()
