# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess the data
# The dataset can be found at the following kaggle link: 
# "https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric" 
file_path = '/path/to/file/METABRIC_RNA_Mutation.csv'
df = pd.read_csv(file_path, low_memory=False)

# Impute missing values for numerical data
imputer = SimpleImputer(strategy='mean')
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = imputer.fit_transform(df[num_cols])

# Encode categorical variables
label_encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Prepare features and target variable
X = df.drop('chemotherapy', axis=1)
y = df['chemotherapy']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initial Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Initial Model Performance:")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Selection with RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
selector = SelectFromModel(rf, prefit=True)
X_important_train = selector.transform(X_train)
X_important_test = selector.transform(X_test)

# Hyperparameter Tuning with Logistic Regression
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), log_reg_params, cv=5)
grid_search.fit(X_important_train, y_train)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_important_test)

# Evaluation of Optimized Model
print("Optimized Model Performance:")
print("Classification Report:\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# Cross-Validation of the Best Model
scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("Cross-Validation Accuracy Scores: ", scores)
print("Average Cross-Validation Score: ", scores.mean())
