# ==============================================================================
# PROJECT: CREDIT CARD FRAUD DETECTION
# AUTHOR: [Your Name Here]
# GITHUB: [Your GitHub Link Here]
# DESCRIPTION: An end-to-end machine learning project to build a model that
#              can identify fraudulent credit card transactions. This project
#              focuses on handling extreme class imbalance, a common and
#              critical challenge in fraud detection.
# ==============================================================================


# === Step 1: Import Essential Libraries ===
# Foundational libraries for data manipulation, analysis, and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn modules for preprocessing, model training, and evaluation.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler  # Scaler of choice for data with outliers.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Specialty library for handling imbalanced datasets.
from imblearn.over_sampling import SMOTE

# Advanced visualization library for creating interactive charts.
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# === Step 2: Load and Inspect Data ===
# Load the dataset and perform the most critical initial check for this problem:
# understanding the level of class imbalance.
train_data = pd.read_csv('creditcard.csv')
train_data_original = train_data.copy() # Create a backup for reference.

# --- Visualize the Core Problem: Extreme Class Imbalance ---
# This plot is the foundation of our entire strategy. It shows that fraudulent
# transactions are incredibly rare, making accuracy a useless metric.
sns.countplot(x='Class', data=train_data)
plt.title('Class Distribution (0: Non-Fraudulent || 1: Fraudulent)')
# plt.show() # Uncomment to display the plot during execution.


# === Step 3: Data Preparation ===
# Separate the data into features (X) and the target (y). The features are the
# information we use to make predictions, and the target is what we are trying to predict.
X = train_data.drop('Class', axis=1)
y = train_data['Class']

# --- Split Data into Training and Validation Sets ---
# This is a critical step to prevent data leakage. The model will only learn from
# the training set, and its performance will be honestly evaluated on the unseen
# validation set. `stratify=y` ensures that the tiny percentage of fraud cases
# is distributed proportionally in both sets.
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,      # Hold out 20% of the data for validation.
    random_state=42,    # Ensures reproducibility of the split.
    stratify=y          # Essential for imbalanced datasets.
)

# --- Feature Scaling ---
# The 'Time' and 'Amount' columns are on a different scale than the PCA-transformed
# V-columns. We use RobustScaler because it is less sensitive to extreme outliers,
# which are common in financial transaction data.
# NOTE: The scaler is fitted ONLY on the training data to avoid data leakage.
scaler = RobustScaler()
columns_to_scale = ['Time', 'Amount']

# Fit the scaler's rules on the training data.
scaler.fit(X_train[columns_to_scale])

# Apply the learned rules to transform both the training and validation sets.
X_train[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
print("Data successfully split and scaled without leakage.")


# === Step 4: Baseline Model Training ===
# We start with a simple Logistic Regression model to establish a baseline.
# This benchmark helps us understand the effectiveness of more advanced techniques.
baseline_model = LogisticRegression()
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_val)


# === Step 5: Baseline Model Evaluation ===
# Evaluate the baseline model's performance. The key metric here is Recall, which
# tells us what percentage of actual fraudulent transactions we successfully caught.
print("\n--- Baseline Model Classification Report ---")
print(classification_report(y_val, baseline_predictions))

# --- Visualize Performance with an Interactive Dashboard ---
report_baseline = classification_report(y_val, baseline_predictions, output_dict=True)
recall_baseline = report_baseline['1']['recall'] * 100
precision_baseline = report_baseline['1']['precision'] * 100
f1_baseline = report_baseline['1']['f1-score'] * 100

fig_baseline = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}]*3])
fig_baseline.add_trace(go.Indicator(mode="gauge+number", value=recall_baseline, title={'text': "<b>Recall</b>"}), row=1, col=1)
fig_baseline.add_trace(go.Indicator(mode="gauge+number", value=precision_baseline, title={'text': "<b>Precision</b>"}), row=1, col=2)
fig_baseline.add_trace(go.Indicator(mode="gauge+number", value=f1_baseline, title={'text': "<b>F1-Score</b>"}), row=1, col=3)
fig_baseline.update_layout(title_text="Baseline Model: Fraud Detection Performance", title_x=0.5)
fig_baseline.write_html("baseline_full_performance.html")
print("\nBaseline performance dashboard saved to 'baseline_full_performance.html'.")


# === Step 6: Handling Extreme Class Imbalance with SMOTE ===
# Our baseline model struggles because it hasn't seen enough fraud examples.
# We use SMOTE (Synthetic Minority Over-sampling Technique) to create new,
# synthetic fraud examples in our training data, giving the model more to learn from.
# IMPORTANT: SMOTE is applied ONLY to the training data.
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("\nSMOTE applied. New balanced training set shape:", X_train_balanced.shape)


# === Step 7: Train a Model on the Balanced Data ===
# Now, we train a new Logistic Regression model on the balanced (resampled) data.
# We expect this model to be much more sensitive to fraudulent transactions.
print("\nTraining a new model on the balanced data...")
balanced_model = LogisticRegression()
balanced_model.fit(X_train_balanced, y_train_balanced)
balanced_predictions = balanced_model.predict(X_val)


# === Step 8: Evaluate the SMOTE Model ===
# We evaluate the new model to see if our recall score has improved. This will
# show the direct impact of using SMOTE to handle the class imbalance.
print("\n--- SMOTE Model Classification Report ---")
print(classification_report(y_val, balanced_predictions))

# --- Visualize SMOTE Model Performance ---
report_balanced = classification_report(y_val, balanced_predictions, output_dict=True)
recall_balanced = report_balanced['1']['recall'] * 100
precision_balanced = report_balanced['1']['precision'] * 100
f1_balanced = report_balanced['1']['f1-score'] * 100

fig_balanced = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}]*3])
fig_balanced.add_trace(go.Indicator(mode="gauge+number", value=recall_balanced, title={'text': "<b>Recall</b>"}), row=1, col=1)
fig_balanced.add_trace(go.Indicator(mode="gauge+number", value=precision_balanced, title={'text': "<b>Precision</b>"}), row=1, col=2)
fig_balanced.add_trace(go.Indicator(mode="gauge+number", value=f1_balanced, title={'text': "<b>F1-Score</b>"}), row=1, col=3)
fig_balanced.update_layout(title_text="SMOTE Model: Fraud Detection Performance", title_x=0.5)
fig_balanced.write_html("balanced_full_performance.html")
print("\nSMOTE model performance dashboard saved to 'balanced_full_performance.html'.")


# === Step 9: Generate Final Submission File ===
# The final output is a CSV file containing the probability of fraud for each
# transaction in the validation set. This provides an actionable, prioritized
# list for a fraud investigation team. We use our best model (the SMOTE model)
# for this task.
print("\nGenerating final submission file with fraud probabilities...")

# Use `predict_proba` to get the probability scores instead of a binary 0/1.
final_probabilities = balanced_model.predict_proba(X_val)[:, 1]

# Create a clear DataFrame for the results.
submission_df = pd.DataFrame({
    'Transaction_Index': X_val.index,
    'Fraud_Probability': final_probabilities
})

# Sort by probability to bring the highest-risk transactions to the top.
submission_df = submission_df.sort_values(by='Fraud_Probability', ascending=False)

# Save the final results.
submission_df.to_csv('fraud_predictions.csv', index=False)
print("\nSubmission file 'fraud_predictions.csv' created successfully!")
print("\nProject Complete.")