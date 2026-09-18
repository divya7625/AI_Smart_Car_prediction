AI BASED SMART CAR RECOMMENDATION AND SAFETY SYSTEM

Folder order:
1_Data -> datasets
2_Notebook -> Jupyter notebook / experiments
3_Models -> trained .pkl files
4_Source_Code -> model training and helper code
5_Streamlit -> user interface
6_Outputs -> graphs and prediction outputs

RUN:
1. Open 4_Source_Code/train_models.py in VS Code or Jupyter and run it once.
2. Confirm 3_Models contains:
   recommendation_model.pkl
   safety_model.pkl
   maintenance_model.pkl
   driver_model.pkl
3. Open terminal in this project root.
4. Run:
   streamlit run 5_Streamlit/app.py

IMPORTANT:
The supplied 29-column CSV is tabular. It can support tabular ML demos.
Actual YOLOv8 pothole detection needs an image dataset.
Actual webcam drowsiness detection needs a face/eye/video dataset and a separate vision model.
