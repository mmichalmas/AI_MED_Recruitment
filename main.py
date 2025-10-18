import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# loading data
data = pd.read_csv("task_data.csv")
# data separation
X = data[[
    "Heart width","Lung width", "CTR - Cardiothoracic Ratio",
    "xx","yy","xy","normalized_diff","Inscribed circle radius",
    "Polygon Area Ratio","Heart perimeter","Heart area", "Lung area"
]]
y = data["Cardiomegaly"]
# splitting dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalizing numeric data
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
