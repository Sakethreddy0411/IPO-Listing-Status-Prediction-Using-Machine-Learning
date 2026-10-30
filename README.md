# IPO Listing Status Prediction Using Machine Learning

## 1. Project Overview

The IPO Listing Status Prediction project focuses on analyzing Initial Public Offering (IPO) data from 2019 to 2026 and predicting IPO listing status using Machine Learning classification algorithms.

This project explores historical IPO trends across different countries, business sectors, years, and IPO categories. It applies Data Science techniques to identify patterns in IPO records and compares multiple Machine Learning models to determine their performance in predicting listing status.

The project was developed using Python in the Spyder IDE (Anaconda).

## 2. Project Objectives

- Analyze historical IPO data from 2019 to 2026.
- Identify IPO trends across different countries and sectors.
- Perform Exploratory Data Analysis (EDA).
- Apply data preprocessing and feature engineering techniques.
- Implement multiple Machine Learning classification algorithms.
- Compare model performance using evaluation metrics.
- Predict IPO listing status using historical IPO information.

## 3. Dataset Description

The dataset contains historical IPO information, including the following attributes:

| Feature | Description |
|---|---|
| ipo_type | Type of IPO |
| year | IPO record year |
| news_reports | Number of news reports |
| amount_status | IPO amount-related status |
| country | Country associated with the IPO |
| sector | Business sector |
| first_report_date | Date of the first IPO report |
| listing_status | IPO listing status (Target Variable) |

### Target Variable

The project predicts three listing status categories:

- Announced / Filed
- Completed
- Withdrawn / Postponed

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Spyder (Anaconda) | Development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning algorithms |

## 5. Project Workflow

### 5.1 Data Loading

- Imported the dataset using Pandas.
- Examined dataset dimensions and structure.
- Checked column names and data types.
- Analyzed missing values and duplicate records.

### 5.2 Data Preprocessing

The following preprocessing techniques were applied:

- Converted first_report_date into datetime format.
- Extracted report month and report quarter.
- Handled missing numerical values using median imputation.
- Handled missing categorical values using most frequent value imputation.
- Applied One-Hot Encoding to categorical variables.
- Applied StandardScaler to numerical variables.
- Used a ColumnTransformer and Pipeline for preprocessing and model training.

### 5.3 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution and trends in the IPO dataset.

The following visualizations were generated:

1. IPO Listing Status Distribution
2. IPO Records by Year
3. Top 10 Countries by IPO Records
4. Top 10 IPO Sectors

### 5.4 Machine Learning Models

Four classification algorithms were implemented and evaluated.

| Model | Description |
|---|---|
| Logistic Regression | Linear classification algorithm |
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble learning algorithm |

### 5.5 Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Macro averaging was used for Precision, Recall, and F1-Score to evaluate performance across all listing status classes.

## 6. Model Performance Comparison

The following table presents the approximate model evaluation results.

| Machine Learning Model | Accuracy | F1-Score |
|---|---:|---:|
| Logistic Regression | 64% | 48% |
| KNN | 80% | 49% |
| Decision Tree | 78% | 51% |
| Random Forest | 85% | 54% |

**Best Performing Model: Random Forest Classifier**

Random Forest achieved the highest accuracy and macro F1-score among the four implemented classification models.

Note: The values are rounded estimates from the evaluation charts. Refer to IPO_Model_Results_Final.csv for exact performance values.

## 7. Results and Findings

The Exploratory Data Analysis and Machine Learning implementation produced the following observations:

- Retail and Consumer Goods recorded the highest number of IPO records among the analyzed sectors.
- The United States recorded the highest number of IPO records, followed by India.
- The year 2021 recorded the highest number of IPO records in the dataset.
- Announced / Filed was the most frequently occurring listing status.
- Random Forest achieved the best accuracy and F1-score among the four implemented models.
- The confusion matrix helped analyze correct and incorrect predictions for each listing status category.

## 8. Visualizations

The project includes the following visualizations:

- IPO Listing Status Distribution
- IPO Records by Year
- Top 10 Countries by IPO Records
- Top 10 IPO Sectors
- Accuracy Comparison
- F1-Score Comparison
- Confusion Matrix

## 9. Project Structure

```text
IPO-Listing-Status-Prediction/
│
├── dataset/
│   └── ipo_data_rows.csv
│
├── visualizations/
│   ├── ipo_listing_status_distribution.png
│   ├── ipo_records_by_year.png
│   ├── top_10_countries.png
│   ├── top_10_sectors.png
│   ├── accuracy_comparison.png
│   ├── f1_score_comparison.png
│   └── confusion_matrix.png
│
├── ipo_prediction.py
├── IPO_Model_Results_Final.csv
└── README.md
```

## 10. Installation and Execution

### Step 1: Install Anaconda

Download and install Anaconda:

https://www.anaconda.com/download

### Step 2: Open Spyder

Launch Spyder IDE through Anaconda Navigator.

### Step 3: Install Required Libraries

Open Anaconda Prompt and execute:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 4: Download or Clone the Repository

```bash
git clone https://github.com/your-username/IPO-Listing-Status-Prediction.git
```

### Step 5: Open the Python File

Open the `ipo_prediction.py` file in Spyder.

### Step 6: Configure Dataset Path

Update the CSV file path in the Python script according to your local dataset location.

### Step 7: Execute the Project

Press F5 or click the Run button in Spyder.

### Step 8: View the Results

After execution, the program displays:

- Data analysis and visualizations.
- Classification reports for all four models.
- Model performance comparison.
- Confusion matrix for the best-performing model.
- Final model performance CSV file.

## 11. Future Enhancements

- Improve model performance through hyperparameter tuning.
- Explore advanced ensemble learning algorithms.
- Include additional financial and market-related features.
- Develop an interactive IPO analysis dashboard.
- Deploy the prediction model using Flask or Streamlit.
- Expand the dataset with more historical IPO records.

## 12. Conclusion

This project demonstrates the application of Data Science and Machine Learning techniques to historical IPO data.

Through data preprocessing, exploratory data analysis, feature engineering, and classification algorithms, the project identifies important IPO trends and predicts IPO listing status.

The comparison of four Machine Learning models demonstrates that Random Forest achieved the best overall performance among the implemented algorithms.

The project provided practical experience in Python programming, data analysis, visualization, Machine Learning implementation, and model evaluation.

## 13. Author

**Saketh Reddy Ambati**

B.Tech – Computer Science and Engineering

Lovely Professional University

**Project Domain:** Data Science and Machine Learning

**Development Environment:** Spyder IDE (Anaconda)
