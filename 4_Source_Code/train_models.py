# AI Smart Car - Model Training
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "1_Data", "AI_Car_Recommendation_Safety_5000.csv")
MODEL_DIR = os.path.join(BASE, "3_Models")
OUT = os.path.join(BASE, "6_Outputs")
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
print("Dataset shape:", df.shape)
print(df.head())

def make_preprocessor(X):
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])

# 1. Recommendation classification
rec_features = ["Price_Lakh_INR","Fuel_Type","Transmission","Engine_CC",
                "Mileage_or_Range","Safety_Rating","Airbags","ABS","ESP",
                "Rear_Camera","Driver_Behavior"]
X = df[rec_features]; y = df["Recommendation"]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
rec_model = Pipeline([("prep",make_preprocessor(X)),
                       ("model",RandomForestClassifier(n_estimators=200,random_state=42))])
rec_model.fit(Xtr,ytr)
print("Recommendation accuracy:", accuracy_score(yte,rec_model.predict(Xte)))
joblib.dump(rec_model, os.path.join(MODEL_DIR,"recommendation_model.pkl"))

# 2. Overall safety score regression
safety_features = ["Price_Lakh_INR","Fuel_Type","Transmission","Engine_CC",
                   "Mileage_or_Range","Safety_Rating","Airbags","ABS","ESP",
                   "Rear_Camera","Driver_Behavior","Road_Risk","Pothole_Risk",
                   "Accident_Risk_Score"]
X = df[safety_features]; y = df["Overall_Safety_Score"]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
safety_model = Pipeline([("prep",make_preprocessor(X)),
                          ("model",RandomForestRegressor(n_estimators=200,random_state=42))])
safety_model.fit(Xtr,ytr)
pred=safety_model.predict(Xte)
print("Safety MAE:", mean_absolute_error(yte,pred), "R2:", r2_score(yte,pred))
joblib.dump(safety_model, os.path.join(MODEL_DIR,"safety_model.pkl"))

# 3. Maintenance risk classification
maint_features = ["Engine_Temperature_C","Battery_Voltage_V","Brake_Condition",
                  "Tire_Pressure_PSI","Oil_Health_Percent"]
X=df[maint_features]; y=df["Maintenance_Risk"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
maint_model=Pipeline([("prep",make_preprocessor(X)),
                      ("model",RandomForestClassifier(n_estimators=200,random_state=42))])
maint_model.fit(Xtr,ytr)
print("Maintenance accuracy:",accuracy_score(yte,maint_model.predict(Xte)))
joblib.dump(maint_model,os.path.join(MODEL_DIR,"maintenance_model.pkl"))

# 4. Driver behavior classification
driver_features=["Driver_Behavior","Road_Risk","Pothole_Risk","Accident_Risk"]
# For a real deployment, replace this with camera/image features; this tabular model is only a demo.
X=df[driver_features]; y=df["Driver_Behavior"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
driver_model=Pipeline([("prep",make_preprocessor(X)),
                       ("model",RandomForestClassifier(n_estimators=200,random_state=42))])
# Avoid target leakage: train a simple road-risk proxy instead of predicting Driver_Behavior from itself
driver_features=["Road_Risk","Pothole_Risk","Accident_Risk"]
X=df[driver_features]; y=df["Driver_Behavior"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
driver_model=Pipeline([("prep",make_preprocessor(X)),
                       ("model",RandomForestClassifier(n_estimators=200,random_state=42))])
driver_model.fit(Xtr,ytr)
print("Driver behavior proxy accuracy:",accuracy_score(yte,driver_model.predict(Xte)))
joblib.dump(driver_model,os.path.join(MODEL_DIR,"driver_model.pkl"))

print("Training complete. Models saved in 3_Models/")
