🏠 California Real Estate Price Prediction using Machine Learning

Predicting housing prices using multiple Regression algorithms and identifying the best performing model based on performance metrics.

📌 Project Overview

This project predicts median house values using the California Housing dataset by applying various Machine Learning regression models. You will learn:

✔ Data preprocessing & cleaning
✔ Model building and evaluation
✔ Comparison of regression algorithms
✔ Feature importance analysis
✔ Insights from visualizations

📂 Dataset

The dataset contains housing information such as:

median income

total rooms & bedrooms

population

households

housing age

ocean proximity (encoded)

Target variable: median_house_value

💡 If the dataset is >25MB and can’t be uploaded directly, you can download it from:
🔗 https://www.kaggle.com/datasets/camnugent/california-housing-prices

Save the CSV file in a folder named data/ inside this repo.

🚀 Tech Stack

Python

Pandas

NumPy

Scikit-Learn

Matplotlib

Seaborn

Jupyter Notebook

🧠 Models Trained

We trained and evaluated the following regression models:

Model	Description
Linear Regression	Baseline linear model
Decision Tree Regressor	Single tree regression
KNN Regressor	Distance based regression
Random Forest Regressor ⭐	Ensemble of decision trees
Gradient Boosting Regressor	Boosted trees
📊 Performance Comparison
Model	RMSE (Lower is better)	MAE
Linear Regression	77,005	56,671
Decision Tree	70,865	49,570
KNN Regressor	64,779	45,741
Random Forest Regressor	63,236	43,633
Gradient Boosting Regressor	(fix code run separately)	(run evaluation)

✅ The Random Forest Regressor gave the best performance.

🏆 Best Performing Model

🎯 Random Forest Regressor
RMSE ≈ 63,000

This shows it can capture non-linear relationships in housing features and generalize better than other models.

📈 Visualizations Included

Correlation Heatmap — Shows feature relationships

Actual vs Predicted Prices — Model prediction quality

Feature Importance Plot — Most influential features

These graphs provide deeper insights beyond numbers.

📉 Key Insights

✔ median_income is the most influential predictor
✔ Tree-based ensembles perform strongly
✔ Distance-based (KNN) also gives competitive results
✔ Simple linear model performs the worst

📁 Repo Structure
├── data/
│   └── housing.csv
├── notebooks/
│   └── California_housing_regression.ipynb
├── plots/
│   ├── feature_importance.png
│   ├── corr_heatmap.png
│   └── actual_vs_pred.png
├── requirements.txt
├── README.md

▶️ How to Run

Clone the repo

git clone https://github.com/bhupender5/California-realstate-price-predicion-using-ml-


Install dependencies

pip install -r requirements.txt


Load and explore the notebook
Open:

notebooks/California_housing_regression.ipynb

🧰 Requirements

Example requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn

🔮 Future Enhancements

Hyperparameter tuning (GridSearchCV / RandomizedSearch)

XGBoost model for further improvement

Deployment with Streamlit / Flask

Exporting model as API

👤 Author

Bhupender Singh
📊 Aspiring Data Scientist / ML Engineer

🔗 https://github.com/bhupender5

🔗 https://www.linkedin.com/in/bhupinder-singh-bba271187

⭐ Enjoyed this project?

👉 Give it a ⭐ on GitHub!
