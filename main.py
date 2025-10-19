import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# loading data
data = pd.read_csv("task_data.csv")
# formating df
data.columns = data.columns.str.strip().str.replace(' ', '_')
wrong_type_columns = ["CTR_-_Cardiothoracic_Ratio","Inscribed_circle_radius","Heart_perimeter"]
for col in wrong_type_columns:
    data[col] = data[col].astype(str).str.replace(',','.').astype(float)
print(data)
# data separation
X = data[[
    "Heart_width","Lung_width", "CTR_-_Cardiothoracic_Ratio",
    "xx","yy","xy","normalized_diff","Inscribed_circle_radius",
    "Polygon_Area_Ratio","Heart_perimeter","Heart_area", "Lung_area"
]]
y = data["Cardiomegaly"]
# splitting dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(y_train.value_counts())

#SVM
pipe_svc = Pipeline([
    ('scaler', StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=3,
        gamma="scale",
        class_weight="balanced", # In cardiomegaly 1 occurs much more frequently than 0
    ))
])

cv_score = np.round(cross_val_score(pipe_svc, X_train, y_train), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {cv_score.mean():.3f}")
print(f"Standard deviation of CV score: {cv_score.std():.3f}")
