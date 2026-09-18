import os
import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "1_Data", "AI_Car_Recommendation_Safety_5000.csv")
MODELS = os.path.join(ROOT, "3_Models")

st.set_page_config(page_title="AI Smart Car Recommendation & Safety", page_icon="🚗", layout="wide")
st.title("🚗 AI Based Smart Car Recommendation and Safety System")
st.write("Enter vehicle/user details and get AI-based predictions.")

@st.cache_resource
def load_models():
    return (joblib.load(os.path.join(MODELS,"recommendation_model.pkl")),
            joblib.load(os.path.join(MODELS,"safety_model.pkl")),
            joblib.load(os.path.join(MODELS,"maintenance_model.pkl")),
            joblib.load(os.path.join(MODELS,"driver_model.pkl")))

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

try:
    rec_model, safety_model, maint_model, driver_model = load_models()
    df = load_data()
except Exception:
    st.error("Models not found. First run 4_Source_Code/train_models.py")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🚘 Car Recommendation","🛡️ Safety Prediction","🔧 Maintenance Risk"])

with tab1:
    st.subheader("User / Car Inputs")
    c1,c2,c3=st.columns(3)
    with c1:
        price=st.number_input("Budget / Price (₹ Lakhs)", min_value=1.0, max_value=200.0, value=10.0)
        fuel=st.selectbox("Fuel Type", sorted(df["Fuel_Type"].dropna().unique()))
        trans=st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
        engine=st.number_input("Engine CC", 500, 6000, 1200)
    with c2:
        mileage=st.number_input("Mileage / Range", 1.0, 1000.0, 18.0)
        safety_rating=st.number_input("Safety Rating", 0.0, 5.0, 4.0)
        airbags=st.number_input("Airbags", 0, 12, 6)
        abs=st.selectbox("ABS", sorted(df["ABS"].dropna().unique()))
    with c3:
        esp=st.selectbox("ESP", sorted(df["ESP"].dropna().unique()))
        rear=st.selectbox("Rear Camera", sorted(df["Rear_Camera"].dropna().unique()))
        driver=st.selectbox("Driver Behavior", sorted(df["Driver_Behavior"].dropna().unique()))
    if st.button("🔮 Predict Recommendation", type="primary"):
        x=pd.DataFrame([{"Price_Lakh_INR":price,"Fuel_Type":fuel,"Transmission":trans,
                         "Engine_CC":engine,"Mileage_or_Range":mileage,"Safety_Rating":safety_rating,
                         "Airbags":airbags,"ABS":abs,"ESP":esp,"Rear_Camera":rear,
                         "Driver_Behavior":driver}])
        pred=rec_model.predict(x)[0]
        st.success(f"Recommendation: {pred}")
        # Show closest cars matching the user budget/preferences
        view=df.copy()
        view["distance"]=(view["Price_Lakh_INR"]-price).abs()
        view=view[(view["Fuel_Type"]==fuel) & (view["Transmission"]==trans)]
        if len(view):
            top=view.sort_values(["distance","Recommendation_Score"],ascending=[True,False]).head(5)
            st.subheader("Suggested Cars from Dataset")
            st.dataframe(top[["Brand","Model","Year","Price_Lakh_INR","Fuel_Type","Transmission",
                              "Mileage_or_Range","Safety_Rating","Overall_Safety_Score",
                              "Recommendation_Score","Recommendation"]],use_container_width=True)

with tab2:
    st.subheader("Safety Inputs")
    road=st.selectbox("Road Risk", sorted(df["Road_Risk"].dropna().unique()))
    pothole=st.selectbox("Pothole Risk", sorted(df["Pothole_Risk"].dropna().unique()))
    accident=st.number_input("Accident Risk Score",0.0,100.0,20.0)
    if st.button("🛡️ Predict Safety Score"):
        x=pd.DataFrame([{"Price_Lakh_INR":price if 'price' in locals() else 10.0,
                         "Fuel_Type":fuel if 'fuel' in locals() else df["Fuel_Type"].mode()[0],
                         "Transmission":trans if 'trans' in locals() else df["Transmission"].mode()[0],
                         "Engine_CC":engine if 'engine' in locals() else 1200,
                         "Mileage_or_Range":mileage if 'mileage' in locals() else 18.0,
                         "Safety_Rating":safety_rating if 'safety_rating' in locals() else 4.0,
                         "Airbags":airbags if 'airbags' in locals() else 6,
                         "ABS":abs if 'abs' in locals() else df["ABS"].mode()[0],
                         "ESP":esp if 'esp' in locals() else df["ESP"].mode()[0],
                         "Rear_Camera":rear if 'rear' in locals() else df["Rear_Camera"].mode()[0],
                         "Driver_Behavior":driver if 'driver' in locals() else df["Driver_Behavior"].mode()[0],
                         "Road_Risk":road,"Pothole_Risk":pothole,"Accident_Risk_Score":accident}])
        score=float(safety_model.predict(x)[0])
        st.metric("Predicted Overall Safety Score",f"{score:.1f}/100")
        st.progress(max(0,min(100,int(score))))

with tab3:
    st.subheader("Vehicle Health Inputs")
    temp=st.number_input("Engine Temperature (°C)",50.0,150.0,90.0)
    batt=st.number_input("Battery Voltage (V)",8.0,16.0,12.5)
    brake=st.number_input("Brake Condition (%)",0.0,100.0,80.0)
    tire=st.number_input("Tire Pressure (PSI)",10.0,60.0,32.0)
    oil=st.number_input("Oil Health (%)",0.0,100.0,80.0)
    if st.button("🔧 Predict Maintenance Risk"):
        x=pd.DataFrame([{"Engine_Temperature_C":temp,"Battery_Voltage_V":batt,
                         "Brake_Condition":brake,"Tire_Pressure_PSI":tire,
                         "Oil_Health_Percent":oil}])
        pred=maint_model.predict(x)[0]
        st.warning(f"Maintenance Risk: {pred}")
