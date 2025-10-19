import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# normalizing numeric data
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
