import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict, RepeatedStratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
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
# not splitting dataset because of its size
X_train = X
y_train = y
print(y_train.value_counts())

#SVM
pipe_svc = Pipeline([
    ('generator',ADASYN(n_neighbors=4,random_state=42)),
    ('scaler',StandardScaler()),
    ("model", SVC(
        kernel="poly",
        C=10,
        gamma="auto",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
cv_score = np.round(cross_val_score(pipe_svc, X_train, y_train,cv=cv,scoring="recall_macro"), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {cv_score.mean():.3f}")
print(f"Standard deviation of CV score: {cv_score.std():.3f}")

# Classification report from svc prediction
y_pred = cross_val_predict(pipe_svc, X_train, y_train)
print(classification_report(y_train, y_pred, digits=3))

param_grid = {
    'generator__n_neighbors': [2, 3,4],
    'model__C': [0.5, 1, 3, 10],
    'model__gamma': ['scale', 'auto'],
    'model__kernel': ['rbf', 'poly']

}

grid = GridSearchCV(pipe_svc, param_grid, cv=5, scoring="recall_macro", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best recall_macro:", grid.best_score_)


