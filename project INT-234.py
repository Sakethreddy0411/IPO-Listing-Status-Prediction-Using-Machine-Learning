import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("C:/Users/ambati saketh reddy/Downloads/ipo_data_rows.csv")

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

df["first_report_date"] = pd.to_datetime(df["first_report_date"], errors="coerce")
df["report_month"] = df["first_report_date"].dt.month.fillna(0).astype(int)
df["report_quarter"] = df["first_report_date"].dt.quarter.fillna(0).astype(int)

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="listing_status")
plt.title("IPO Listing Status Distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df["year"].value_counts().sort_index().plot(kind="bar")
plt.title("IPO Records by Year")
plt.xlabel("Year")
plt.ylabel("Number of IPOs")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df["country"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Countries by IPO Records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
df["sector"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 IPO Sectors")
plt.tight_layout()
plt.show()

features = [
    "ipo_type",
    "year",
    "news_reports",
    "amount_status",
    "country",
    "sector",
    "report_month",
    "report_quarter"
]

X = df[features]
y = df["listing_status"]

categorical_features = [
    "ipo_type",
    "amount_status",
    "country",
    "sector"
]

numerical_features = [
    "year",
    "news_reports",
    "report_month",
    "report_quarter"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = pipeline

    print("\n", name)
    print(classification_report(y_test, predictions, zero_division=0))

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(4))

best_model_name = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]
best_predictions = best_model.predict(X_test)

print("\nBest Model:", best_model_name)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=best_model.classes_
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.tight_layout()
plt.show()

for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Model", y=metric)
    plt.title(metric + " Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

results_df.to_csv(
    r"C:\Users\ambati saketh reddy\Downloads\IPO_Model_Results_Final.csv",
    index=False
)

print("CSV file saved successfully!")
