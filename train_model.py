import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_excel("credit_rating.xls")

# Separate features (X) and target variable (y)
X = dataset.drop(columns=["Credit classification"])
y = dataset["Credit classification"]

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing steps for categorical and numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = MinMaxScaler()

# Get column indices for categorical features
categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

# Get column indices for numerical features
numerical_indices = [i for i in range(len(X.columns)) if i not in categorical_indices]

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_indices),
        ('num', numerical_transformer, numerical_indices)
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classification model using Random Forest Classifier
classification_model = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', RandomForestClassifier())])

# Define the hyperparameters grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20], # Maximum depth of the trees
    'classifier__min_samples_split': [2, 5, 10] # Minimum number of samples required to split a node
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(classification_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Save the best estimator (model)
joblib.dump(best_estimator, "trained_model_rf_best.pkl")

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Predict the target variable on the test set using the best estimator
y_pred = best_estimator.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')

# Print F1 score
print("F1 Score:", f1)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)



