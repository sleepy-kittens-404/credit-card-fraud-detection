<div align="center">

# 💳 Credit Card Fraud Detection

[![Live Demo](https://img.shields.io/badge/Live%20Demo-AWS%20Elastic%20Beanstalk-orange?style=for-the-badge&logo=amazon-aws)](http://credit-card-fraud-detection-env.eba-yyzs69mt.eu-north-1.elasticbeanstalk.com/)
[![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end machine learning pipeline that detects fraudulent credit card transactions in real time. Trained on 284K anonymized transactions with a 0.17% fraud rate, the system handles severe class imbalance using SMOTE and achieves an **F1 score of 0.85** with a Random Forest classifier.

</div>

---

## 🖥️ App Demo

| Home Screen | Fraud Detected | Legitimate Transaction |
|:-----------:|:--------------:|:---------------------:|
| ![Home](screenshots/home.png) | ![Fraud](screenshots/fraud.png) | ![Legit](screenshots/legit.png) |

> The app loads a real transaction from the held-out test set, runs it through the trained model, and displays the fraud probability alongside the ground truth label — so you can see whether the model got it right.

---

## 🧠 How It Works

```
Raw CSV Data
     │
     ▼
Data Ingestion ──► Deduplication ──► Train/Test Split (80/20)
     │
     ▼
Data Transformation
     ├── RobustScaler  ──► Amount
     ├── StandardScaler ──► Time
     └── SMOTE  ──► Oversample minority (fraud) class on train set
     │
     ▼
Model Training ──► GridSearchCV across 8 models ──► Best model saved
     │
     ▼
Prediction Pipeline ──► Load preprocessor + model ──► Predict + Probability
     │
     ▼
Flask App ──► Random sample from test set ──► Display result
```

---

## 📊 Model Comparison

8 classifiers were trained with GridSearchCV and evaluated on F1 score (chosen over accuracy due to severe class imbalance):

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Fast, interpretable |
| **Random Forest ✅** | **Best performer — selected as final model** |
| Gradient Boosting | Strong but slow to train |
| AdaBoost | Sensitive to noise in imbalanced data |
| Naive Bayes | Fast but assumes feature independence |
| XGBoost | Close second to Random Forest |
| CatBoost | Strong, handles categoricals natively |

**Final Model: RandomForestClassifier**
```
n_estimators : 300
max_depth    : None
resampling   : SMOTE
scaler       : RobustScaler (Amount) + StandardScaler (Time)
F1 Score     : 0.85
```

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
├── artifacts/                  # Saved model, preprocessor, datasets
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── test_small.csv
├── notebooks/                  # EDA and model training experiments
│   ├── EDA.ipynb
│   └── Model_training.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Load, deduplicate, split data
│   │   ├── data_transformation.py  # Scale + SMOTE pipeline
│   │   └── model_training.py       # GridSearchCV across 8 models
│   ├── pipeline/
│   │   ├── predict_pipeline.py     # Inference logic
│   │   └── train_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/                  # Flask HTML templates
├── .ebextensions/              # AWS Elastic Beanstalk config
├── application.py              # Flask app entry point
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## 🚀 Run Locally

### Option 1 — Docker (recommended)

```bash
docker pull sami1001/credit-card-fraud-detection
docker run -p 5000:5000 sami1001/credit-card-fraud-detection
```

Then open [http://localhost:5000](http://localhost:5000)

### Option 2 — Python

```bash
git clone https://github.com/sleepy-kittens-404/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
python application.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost, CatBoost |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Web Framework | Flask |
| Containerization | Docker |
| CI/CD | AWS CodePipeline |
| Deployment | AWS Elastic Beanstalk (EC2) |
| Serialization | dill |

---

## 📁 Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 2 days
- 492 fraud cases (0.17% of all transactions)
- Features V1–V28 are PCA-transformed for anonymization
- Only `Time` and `Amount` are original features

---

## 👤 Author

**Muhammad Sami**
[![GitHub](https://img.shields.io/badge/GitHub-sleepy--kittens--404-181717?style=flat&logo=github)](https://github.com/sleepy-kittens-404)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-muhammad--sami-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/muhammad-sami-7ba422395)