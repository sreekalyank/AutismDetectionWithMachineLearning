import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, cohen_kappa_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("C:\\Users\\91939\\Downloads\\csv_result-Autism-Adult-Data.csv")
df.drop(df.columns[0], axis=1, inplace=True)

# Separate features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Select categorical features
category_features = features.iloc[:, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

# Drop categorical columns
features.drop(features.columns[-10:], axis=1, inplace=True)

# Fill missing values with mean
features = features.fillna(features.mean())

# One Hot Encoding without changing column names
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
category_transformed = enc.fit_transform(category_features)
category_encoded_columns = enc.get_feature_names_out(category_features.columns)
transformed_df = pd.DataFrame(category_transformed, columns=category_encoded_columns)

# Concatenate encoded features with numerical features
result_df = pd.concat([features, transformed_df], axis=1)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(result_df, labels, test_size=0.3, random_state=42)

# Define Decision Tree classifier
decision_tree = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=3, random_state=39)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('normalizer', Normalizer())  # Using Normalizer
        ]), ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), category_encoded_columns)
    ],
    remainder='passthrough'
)

# Create pipeline with preprocessing and Decision Tree classifier for training and testing
pipeline_cv_dt = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', decision_tree)
])

# Define 10-fold cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=35)

# Perform cross-validation on training set
cv_scores_dt = cross_val_score(pipeline_cv_dt, X_train, y_train, cv=cv, scoring='accuracy')

# Print cross-validation scores
print("Cross-validation scores (Decision Tree):", cv_scores_dt)
print("Mean CV accuracy (Decision Tree):", np.mean(cv_scores_dt))

# Train the model on the entire training set using pipeline_cv_dt
pipeline_cv_dt.fit(X_train, y_train)

# Test the model on the separate testing set
y_pred_test_dt = pipeline_cv_dt.predict(X_test)
test_accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
print("Testing Set Accuracy with cross-validation (Decision Tree):", test_accuracy_test_dt)

# Additional evaluation metrics
print("Precision:", precision_score(y_test, y_pred_test_dt, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_test_dt, average='macro'))
print("Recall:", recall_score(y_test, y_pred_test_dt, average='macro'))
print("Cohen's Kappa Score:", cohen_kappa_score(y_test, y_pred_test_dt))
print("Log Loss:", log_loss(y_test, pipeline_cv_dt.predict_proba(X_test)))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred_test_dt))
print("ROC AUC:", roc_auc_score(y_test, pipeline_cv_dt.predict_proba(X_test)[:, 1]))
