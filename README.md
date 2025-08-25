# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning models. Due to the highly imbalanced dataset, special attention was given to recall to ensure reliable fraud detection.

----

# Dataset

The dataset contains anonymized credit card transactions, with class labels indicating whether a transaction is fraudulent (1) or genuine (0).
Dataset link

----

# Project Workflow
	•	Data preprocessing and handling class imbalance
	•	Exploratory data analysis (EDA) with visualizations:
	•	Count plots (categorical distributions)
	•	Histogram plots (numerical distributions)
	•	Correlation heatmap
	•	Gauge plots for precision, recall, F1-score
	•	Model training using multiple classifiers
	•	Performance comparison across models

 ------



# Evaluation Metrics
	•	Accuracy is not reliable due to imbalance.
	•	Focus on Recall (catching fraud cases) and Precision (avoiding false alarms).
	•	Best performing model selected based on F1-score and recall trade-off.

------

# Tech Stack
	•	Language: Python
	•	Libraries & Tools:
	•	pandas, numpy → Data handling & preprocessing
	•	matplotlib, seaborn → Data visualization
	•	scikit-learn → Train-test split, model training, evaluation metrics
	•	imbalanced-learn (SMOTE, etc.) → Handling class imbalance
	•	gauge (dash/plotly or custom) → Precision, Recall visualization

-----------

# Results
	•	The dataset was highly imbalanced (fraudulent transactions were <1%).
	•	Applied resampling techniques to handle imbalance.
	•	Evaluated models using Precision, Recall, F1-score.
	•	The final best model was chosen based on:
	•	High Recall (to detect maximum fraud cases)
#### •	Achieved:
	•	Recall (Fraud class): ~0.92
	•	Precision (Fraud class): ~0.6
	•	F1-Score: ~11.0
	
-----

# How to Run
  1.	Clone the repository

	2.	Install dependencies: pip install -r requirements.txt
 
  3. Run the notebook or script to reproduce results
	
 
 

---------

## Author: Harsh Chaudhary
