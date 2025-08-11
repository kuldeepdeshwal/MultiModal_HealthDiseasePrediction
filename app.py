# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# import datetime
# import os
# import matplotlib.pyplot as plt
#
# # -------------------- PAGE SETUP --------------------
# st.set_page_config(page_title="🩺 Health Prediction App", layout="centered")
#
# # -------------------- SESSION STATE INIT --------------------
# for key, val in {
#     "app_mode": None,
#     "predicted": False,
#     "predicted_disease": "",
#     "selected_weights": [],
#     "user_symptoms": [],
#     "show_history": False,
#     "show_desc": False,
#     "show_adv": False
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = val
#
# # ✅ Reset app_mode cleanly if it's an empty string
# if "app_mode" in st.session_state and st.session_state.app_mode == "":
#     del st.session_state["app_mode"]
#
# # -------------------- SIDEBAR HISTORY --------------------
# st.sidebar.markdown("## 📂 My Prediction History")
# if st.sidebar.button("🔎 View History"):
#     st.session_state.show_history = True
#
# if st.session_state.show_history:
#     st.title("🗂️ View Prediction History")
#
#     if st.session_state.app_mode is None:
#         history_choice = st.radio(
#             "Select which history to view:",
#             ["Heart Disease Prediction", "Symptom-Based Disease Prediction"],
#             index=None,
#             key="history_choice"
#         )
#
#         if history_choice == "Heart Disease Prediction":
#             history_file = "heart_history.csv"
#         elif history_choice == "Symptom-Based Disease Prediction":
#             history_file = "user_history.csv"
#         else:
#             history_file = None
#     else:
#         history_file = "heart_history.csv" if st.session_state.app_mode == "Heart" else "user_history.csv"
#
#     if history_file:
#         if os.path.exists(history_file):
#             df = pd.read_csv(history_file, header=None, names=["DateTime", "Inputs", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found.")
#     else:
#         st.info("Please select a module to view its history.")
#
#     if st.button("❌ Close History"):
#         st.session_state.show_history = False
#         st.rerun()
#
#     st.stop()
#
# # -------------------- MODULE SELECTION --------------------
# if not st.session_state.app_mode:
#     st.title("🩺 Unified Health Prediction Platform")
#     st.markdown("Choose the type of prediction you want to perform:")
#
#     if st.button("💓 Heart Disease Prediction"):
#         st.session_state.app_mode = "Heart"
#         st.rerun()
#
#     if st.button("🦠 Symptom-Based Disease Prediction"):
#         st.session_state.app_mode = "Symptom"
#         st.rerun()
#
# # -------------------- BACK BUTTON --------------------
# if st.session_state.app_mode:
#     if st.button("🔙 Back to Module Selection"):
#         st.session_state.app_mode = None
#         st.session_state.predicted = False
#         st.session_state.predicted_disease = ""
#         st.session_state.user_symptoms = []
#         st.session_state.selected_weights = []
#         st.session_state.show_desc = False
#         st.session_state.show_adv = False
#         st.rerun()
#
# # -------------------- HEART DISEASE MODULE --------------------
# if st.session_state.app_mode == "Heart":
#     st.header("💓 Heart Disease Prediction using KNN")
#     model = joblib.load("models/knn_heart_disease_model.pkl")
#
#     age = st.number_input("Age", 20, 100, 50)
#     sex = st.selectbox("Sex", ["Male", "Female"])
#     trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#     chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
#     fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
#     thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
#     exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
#     oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
#     cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
#     restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
#
#     cp_dict = {
#         "Typical angina": [1, 0, 0],
#         "Atypical angina": [0, 1, 0],
#         "Non-anginal": [0, 0, 1],
#         "Asymptomatic": [0, 0, 0]
#     }
#     restecg_dict = {
#         "Normal": [1, 0],
#         "ST-T abnormality": [0, 1],
#         "Left ventricular hypertrophy": [0, 0]
#     }
#
#     features = [
#         age, 1 if sex == "Male" else 0, trestbps, chol, 1 if fbs == "Yes" else 0,
#         thalch, 1 if exang == "Yes" else 0, oldpeak, *cp_dict[cp], *restecg_dict[restecg]
#     ]
#
#     if st.button("🧠 Predict Heart Disease"):
#         prediction = model.predict([features])[0]
#         result = "No Heart Disease" if prediction == 0 else "Heart Disease Detected"
#         st.success(f"🩺 Prediction: **{result}**")
#
#         now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         inputs = f"age={age}, sex={sex}, bp={trestbps}, chol={chol}, fbs={fbs}, thalch={thalch}, exang={exang}, oldpeak={oldpeak}"
#         with open("heart_history.csv", "a") as f:
#             f.write(f"{now},\"{inputs}\",{result}\n")
#
#         st.session_state.predicted = True
#
#     if st.session_state.predicted and st.button("🔄 Clear & Predict Again"):
#         st.session_state.predicted = False
#         st.rerun()
#
# # -------------------- SYMPTOM-BASED MODULE --------------------
# elif st.session_state.app_mode == "Symptom":
#     st.header("🦠 Symptom-Based Disease Prediction")
#
#     model = joblib.load("models/final_model.pkl")
#     label_encoder = joblib.load("models/label_encoder.pkl")
#     all_features = joblib.load("models/all_symptoms.pkl")
#     try:
#         disease_info = joblib.load("models/disease_info.pkl")
#     except:
#         disease_info = {}
#
#     df_sev = pd.read_csv("data/Symptom-severity.csv")
#     df_sev.columns = ['Symptom', 'weight']
#     df_sev['Symptom'] = df_sev['Symptom'].str.lower().str.strip()
#     severity_dict = dict(zip(df_sev['Symptom'], df_sev['weight']))
#
#     symptom_features = [feat for feat in all_features if feat not in ['total_severity_score', 'num_reported_symptoms', 'avg_severity_score']]
#
#     # ✅ Proper multiselect using session state
#     st.multiselect(
#         "Select Symptoms:",
#         options=sorted(symptom_features),
#         key="user_symptoms"
#     )
#
#     if st.button("🔍 Predict Disease"):
#         if not st.session_state.user_symptoms:
#             st.warning("Please select at least one symptom.")
#         else:
#             symptom_vector = [1 if s in st.session_state.user_symptoms else 0 for s in symptom_features]
#             selected_weights = [severity_dict.get(s.lower().strip(), 0) for s in st.session_state.user_symptoms]
#             total_severity = sum(selected_weights)
#             num_symptoms = len(st.session_state.user_symptoms)
#             avg_severity = total_severity / num_symptoms if num_symptoms else 0
#             input_vector = np.array(symptom_vector + [total_severity, num_symptoms, avg_severity]).reshape(1, -1)
#
#             pred_encoded = model.predict(input_vector)[0]
#             predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
#             st.success(f"🧬 Predicted Disease: **{predicted_disease.title()}**")
#
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             symptom_str = ", ".join(st.session_state.user_symptoms)
#             with open("user_history.csv", "a") as f:
#                 f.write(f"{now},\"{symptom_str}\",{predicted_disease}\n")
#
#             st.session_state.predicted = True
#             st.session_state.predicted_disease = predicted_disease
#             st.session_state.selected_weights = selected_weights
#             st.session_state.show_desc = False
#             st.session_state.show_adv = False
#
#     if st.session_state.predicted:
#         with st.expander("🧠 More Analysis", expanded=True):
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 if st.button("📋 Show Description & Precautions", key="desc_btn"):
#                     st.session_state.show_desc = True
#
#             with col2:
#                 if st.button("📊 Advanced Analysis", key="adv_btn"):
#                     st.session_state.show_adv = True
#
#             if st.session_state.show_desc:
#                 key = st.session_state.predicted_disease.lower().strip()
#                 if key in disease_info:
#                     st.markdown("### 📝 Description:")
#                     st.info(disease_info[key]["desc"])
#                     st.markdown("### 🛡️ Precautions:")
#                     for step in disease_info[key]["precautions"]:
#                         st.markdown(f"- {step}")
#                 else:
#                     st.info("ℹ️ No info available for this disease.")
#
#             if st.session_state.show_adv:
#                 if hasattr(model, "feature_importances_"):
#                     importances = model.feature_importances_
#                     feature_df = pd.DataFrame({
#                         "Feature": all_features,
#                         "Importance": importances
#                     }).sort_values(by="Importance", ascending=False)
#                     st.markdown("### 🧠 Top Contributing Features")
#                     st.dataframe(feature_df.head(10))
#
#                 st.markdown("### 🔥 Symptom Severity Chart")
#                 fig, ax = plt.subplots()
#                 ax.barh(st.session_state.user_symptoms, st.session_state.selected_weights, color="orange")
#                 ax.set_xlabel("Severity Weight")
#                 ax.set_title("Symptom Severity")
#                 st.pyplot(fig)
#
#         if st.button("🔄 Clear & Predict Again"):
#             st.session_state.predicted = False
#             st.session_state.predicted_disease = ""
#             st.session_state.selected_weights = []
#             st.session_state.show_desc = False
#             st.session_state.show_adv = False
#             # ❌ DO NOT manually reset st.session_state.user_symptoms
#             st.rerun()
#
# # -------------------- FOOTER --------------------
# if not st.session_state.app_mode:
#     st.markdown("---")
#     st.markdown("Made with ❤️ by **Kuldeep Deshwal**")









# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import pandas as pd
# import joblib
# import datetime
# import os
# import sqlite3
# import uuid
# import matplotlib.pyplot as plt
#
# # -------------------- PAGE SETUP --------------------
# st.set_page_config(page_title="Unified Health Prediction", layout="centered")
#
#
# # -------------------- SESSION STATE INITIALIZATION --------------------
# # Initialize keys for all modules in one place
# def init_session_state():
#     # General app state
#     if "app_mode" not in st.session_state:
#         st.session_state.app_mode = None
#     if "show_history" not in st.session_state:
#         st.session_state.show_history = False
#
#     # State for Symptom-Based module
#     if "predicted_disease" not in st.session_state:
#         st.session_state.predicted_disease = ""
#     if "user_symptoms" not in st.session_state:
#         st.session_state.user_symptoms = []
#
#     # State for Pneumonia module
#     if "key_suffix" not in st.session_state:
#         st.session_state.key_suffix = str(uuid.uuid4())
#     if "uploaded_image" not in st.session_state:
#         st.session_state.uploaded_image = None
#     if "selected_image" not in st.session_state:
#         st.session_state.selected_image = None
#
#
# init_session_state()
#
#
# # -------------------- HELPER FUNCTIONS & MODEL LOADING --------------------
#
# # --- Pneumonia Module Functions ---
# @st.cache_resource
# def load_pneumonia_model():
#     """Loads the pre-trained ResNet18 model for pneumonia detection."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = models.resnet18(weights=None)
#     model.fc = torch.nn.Linear(512, 2)
#     # Ensure the model file is in the root directory or provide the correct path
#     model.load_state_dict(torch.load("pneumonia_resnet18_model.pth", map_location=device))
#     model.to(device)
#     model.eval()
#     return model, device
#
#
# # --- Database Functions for Pneumonia History ---
# DB_NAME = "predictions.db"
#
#
# def create_pneumonia_db():
#     """Creates the SQLite database and table if they don't exist."""
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS prediction_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             filename TEXT,
#             prediction TEXT,
#             timestamp TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()
#
#
# def insert_prediction(filename, prediction):
#     """Inserts a new prediction record into the database."""
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     c.execute("INSERT INTO prediction_log (filename, prediction, timestamp) VALUES (?, ?, ?)",
#               (filename, prediction, timestamp))
#     conn.commit()
#     conn.close()
#
#
# def get_all_predictions():
#     """Retrieves all prediction records from the database."""
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("SELECT timestamp, filename, prediction FROM prediction_log ORDER BY timestamp DESC")
#     rows = c.fetchall()
#     conn.close()
#     return rows
#
#
# create_pneumonia_db()
#
#
# # --- Functions for Clearing State ---
# def reset_all_states():
#     """Resets the session state for all modules when going back to the main menu."""
#     st.session_state.app_mode = None
#     st.session_state.show_history = False
#
#     # Reset symptom module state
#     st.session_state.predicted_disease = ""
#     st.session_state.user_symptoms = []
#
#     # Reset pneumonia module state
#     st.session_state.uploaded_image = None
#     st.session_state.selected_image = None
#     st.session_state.key_suffix = str(uuid.uuid4())  # Refresh file uploaders
#
#
# # -------------------- SIDEBAR --------------------
# st.sidebar.title("Dashboard")
#
# # --- Conditional Sample Selector for Pneumonia Module ---
# if st.session_state.app_mode == "Pneumonia":
#     st.sidebar.header("🔍 Choose a Sample X-ray")
#     sample_dir = "sample_images"
#     if os.path.exists(sample_dir):
#         sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
#
#         selected_sample = st.sidebar.selectbox(
#             "Pick a sample image",
#             [""] + sample_images,
#             key="sample_" + st.session_state.key_suffix
#         )
#         if selected_sample:
#             st.session_state.selected_image = selected_sample
#             st.session_state.uploaded_image = None  # Clear any uploaded file
#     else:
#         st.sidebar.warning("`sample_images` directory not found.")
#
# # --- Unified History Viewer ---
# st.sidebar.markdown("---")
# st.sidebar.header("📂 Prediction History")
#
# if st.sidebar.button("🔎 View Full History"):
#     st.session_state.show_history = True
# else:
#     # This logic allows the history view to persist until the user closes it
#     pass
#
# if st.session_state.show_history:
#     st.title("🗂️ View Prediction History")
#
#     history_choice = st.radio(
#         "Select which history to view:",
#         ["Heart Disease", "Symptom-Based Disease", "Pneumonia Detection"],
#         index=None,
#         key="history_choice"
#     )
#
#     if history_choice == "Heart Disease":
#         history_file = "heart_history.csv"
#         if os.path.exists(history_file):
#             df = pd.read_csv(history_file, header=None, names=["DateTime", "Inputs", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found for Heart Disease.")
#
#     elif history_choice == "Symptom-Based Disease":
#         history_file = "user_history.csv"
#         if os.path.exists(history_file):
#             df = pd.read_csv(history_file, header=None, names=["DateTime", "Symptoms", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found for Symptom-Based Disease.")
#
#     elif history_choice == "Pneumonia Detection":
#         history = get_all_predictions()
#         if history:
#             df = pd.DataFrame(history, columns=["Timestamp", "Filename", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found for Pneumonia Detection.")
#
#     if st.button("❌ Close History"):
#         st.session_state.show_history = False
#         st.rerun()
#
#     # Stop the rest of the app from running while history is displayed
#     st.stop()
#
# # -------------------- MAIN APP LOGIC --------------------
#
# # --- Module Selection Screen (Main Page) ---
# if not st.session_state.app_mode:
#     st.title("🩺 Unified Health Prediction Platform")
#     st.markdown("Choose the type of prediction you want to perform from the options below.")
#     st.markdown("---")
#
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         if st.button("💓 Heart Disease"):
#             st.session_state.app_mode = "Heart"
#             st.rerun()
#     with col2:
#         if st.button("🦠 Symptom-Based"):
#             st.session_state.app_mode = "Symptom"
#             st.rerun()
#     with col3:
#         if st.button("🫁 Pneumonia X-ray"):
#             st.session_state.app_mode = "Pneumonia"
#             st.rerun()
#
#     st.markdown("---")
#     st.info("You can view the prediction history for any module using the sidebar.")
#
# # --- Back Button for all modules ---
# if st.session_state.app_mode:
#     if st.button("🔙 Back to Main Menu"):
#         reset_all_states()
#         st.rerun()
#
# # -------------------- MODULE 1: HEART DISEASE --------------------
# if st.session_state.app_mode == "Heart":
#     st.header("💓 Heart Disease Prediction using KNN")
#     try:
#         model = joblib.load("models/knn_heart_disease_model.pkl")
#
#         age = st.number_input("Age", 20, 100, 50)
#         sex = st.selectbox("Sex", ["Male", "Female"])
#         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#         chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
#         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
#         thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
#         exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
#         oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
#         cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
#         restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
#
#         cp_dict = {"Typical angina": [1, 0, 0], "Atypical angina": [0, 1, 0], "Non-anginal": [0, 0, 1],
#                    "Asymptomatic": [0, 0, 0]}
#         restecg_dict = {"Normal": [1, 0], "ST-T abnormality": [0, 1], "Left ventricular hypertrophy": [0, 0]}
#
#         if st.button("🧠 Predict Heart Disease"):
#             features = [age, 1 if sex == "Male" else 0, trestbps, chol, 1 if fbs == "Yes" else 0, thalch,
#                         1 if exang == "Yes" else 0, oldpeak, *cp_dict[cp], *restecg_dict[restecg]]
#             prediction = model.predict([features])[0]
#             result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
#             st.success(f"🩺 Prediction: **{result}**")
#
#             # Log to CSV
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             inputs = f"age={age}, sex={sex}, bp={trestbps}, chol={chol}, fbs={fbs}, thalch={thalch}, exang={exang}, oldpeak={oldpeak}"
#             with open("heart_history.csv", "a") as f:
#                 f.write(f"{now},\"{inputs}\",{result}\n")
#
#     except FileNotFoundError:
#         st.error(
#             "Model file not found. Please ensure `models/knn_heart_disease_model.pkl` is in the correct directory.")
#
# # -------------------- MODULE 2: SYMPTOM-BASED DISEASE --------------------
# elif st.session_state.app_mode == "Symptom":
#     st.header("🦠 Symptom-Based Disease Prediction")
#     try:
#         model = joblib.load("models/final_model.pkl")
#         label_encoder = joblib.load("models/label_encoder.pkl")
#         all_features = joblib.load("models/all_symptoms.pkl")
#         df_sev = pd.read_csv("data/Symptom-severity.csv")
#
#         df_sev.columns = ['Symptom', 'weight']
#         df_sev['Symptom'] = df_sev['Symptom'].str.lower().str.strip().str.replace('_', ' ')
#         severity_dict = dict(zip(df_sev['Symptom'], df_sev['weight']))
#         symptom_features = [feat for feat in all_features if
#                             feat not in ['total_severity_score', 'num_reported_symptoms', 'avg_severity_score']]
#
#         st.multiselect(
#             "Select Your Symptoms:",
#             options=sorted(symptom_features),
#             key="user_symptoms"
#         )
#
#         if st.button("🔍 Predict Disease"):
#             if not st.session_state.user_symptoms:
#                 st.warning("Please select at least one symptom.")
#             else:
#                 symptom_vector = [1 if s in st.session_state.user_symptoms else 0 for s in symptom_features]
#                 selected_weights = [severity_dict.get(s.lower().strip(), 0) for s in st.session_state.user_symptoms]
#                 total_severity = sum(selected_weights)
#                 num_symptoms = len(st.session_state.user_symptoms)
#                 avg_severity = total_severity / num_symptoms if num_symptoms else 0
#                 input_vector = np.array(symptom_vector + [total_severity, num_symptoms, avg_severity]).reshape(1, -1)
#
#                 pred_encoded = model.predict(input_vector)[0]
#                 predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
#                 st.success(f"🧬 Predicted Disease: **{predicted_disease.title()}**")
#                 st.session_state.predicted_disease = predicted_disease
#
#                 # Log to CSV
#                 now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 symptom_str = ", ".join(st.session_state.user_symptoms)
#                 with open("user_history.csv", "a") as f:
#                     f.write(f"{now},\"{symptom_str}\",{predicted_disease}\n")
#
#     except FileNotFoundError as e:
#         st.error(f"A required file was not found: {e.filename}. Please check your `models` and `data` directories.")
#
#
# # -------------------- MODULE 3: PNEUMONIA DETECTION --------------------
# elif st.session_state.app_mode == "Pneumonia":
#     st.header("🫁 Pneumonia Detection from Chest X-rays")
#     st.write("Upload a chest X-ray image or choose a sample from the sidebar.")
#
#     try:
#         model, device = load_pneumonia_model()
#         class_names = ['NORMAL', 'PNEUMONIA']
#         transform = transforms.Compose([
#             transforms.Resize((128, 128)),
#             transforms.ToTensor(),
#         ])
#
#         image = None
#         filename = None
#
#         # We use a unique key for the file_uploader that changes when cleared
#         uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],
#                                          key="file_" + st.session_state.key_suffix)
#
#         # Logic to handle image loading from either upload or sample selection
#         if uploaded_file:
#             image = Image.open(uploaded_file).convert("RGB")
#             st.session_state.uploaded_image = image
#             st.session_state.selected_image = None
#             filename = uploaded_file.name
#         elif st.session_state.selected_image:
#             image_path = os.path.join("sample_images", st.session_state.selected_image)
#             image = Image.open(image_path).convert("RGB")
#             st.session_state.uploaded_image = image
#             filename = st.session_state.selected_image
#
#         # Display the image and prediction buttons
#         if image:
#             st.image(image, caption=filename, use_container_width=True)
#
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("🔬 Predict"):
#                     input_tensor = transform(image).unsqueeze(0).to(device)
#                     with torch.no_grad():
#                         outputs = model(input_tensor)
#                         _, predicted = torch.max(outputs, 1)
#                         prediction = class_names[predicted.item()]
#                         st.success(f"Prediction: **{prediction}**")
#                         # Log to SQLite DB
#                         if filename:
#                             insert_prediction(filename, prediction)
#
#             with col2:
#                 if st.button("🔄 Clear Image"):
#                     st.session_state.uploaded_image = None
#                     st.session_state.selected_image = None
#                     st.session_state.key_suffix = str(uuid.uuid4())  # Force reload of uploader
#                     st.rerun()
#
#     except FileNotFoundError:
#         st.error("Model file not found. Please ensure `pneumonia_resnet18_model.pth` is present.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")








# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import pandas as pd
# import joblib
# import datetime
# import os
# import sqlite3
# import uuid
# import matplotlib.pyplot as plt
#
# # -------------------- PAGE SETUP --------------------
# st.set_page_config(page_title="Unified Health Prediction", layout="centered")
#
#
# # -------------------- SESSION STATE INITIALIZATION --------------------
# def init_session_state():
#     # General app state
#     if "app_mode" not in st.session_state:
#         st.session_state.app_mode = None
#     if "show_history" not in st.session_state:
#         st.session_state.show_history = False
#
#     # State for Symptom-Based module
#     if "predicted" not in st.session_state:
#         st.session_state.predicted = False
#     if "predicted_disease" not in st.session_state:
#         st.session_state.predicted_disease = ""
#     if "user_symptoms" not in st.session_state:
#         st.session_state.user_symptoms = []
#     if "selected_weights" not in st.session_state:
#         st.session_state.selected_weights = []
#     if "show_desc" not in st.session_state:
#         st.session_state.show_desc = False
#     if "show_adv" not in st.session_state:
#         st.session_state.show_adv = False
#
#     # State for Pneumonia module
#     if "key_suffix" not in st.session_state:
#         st.session_state.key_suffix = str(uuid.uuid4())
#     if "uploaded_image" not in st.session_state:
#         st.session_state.uploaded_image = None
#     if "selected_image" not in st.session_state:
#         st.session_state.selected_image = None
#
#
# init_session_state()
#
#
# # -------------------- HELPER FUNCTIONS & MODEL LOADING --------------------
#
# # --- Pneumonia Module Functions ---
# @st.cache_resource
# def load_pneumonia_model():
#     """Loads the pre-trained ResNet18 model for pneumonia detection."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = models.resnet18(weights=None)
#     model.fc = torch.nn.Linear(512, 2)
#     model.load_state_dict(torch.load("pneumonia_resnet18_model.pth", map_location=device))
#     model.to(device)
#     model.eval()
#     return model, device
#
#
# # --- Database Functions for Pneumonia History ---
# DB_NAME = "predictions.db"
#
#
# def create_pneumonia_db():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("""
#         CREATE TABLE IF NOT EXISTS prediction_log (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             filename TEXT,
#             prediction TEXT,
#             timestamp TEXT
#         )
#     """)
#     conn.commit()
#     conn.close()
#
#
# def insert_prediction(filename, prediction):
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     c.execute("INSERT INTO prediction_log (filename, prediction, timestamp) VALUES (?, ?, ?)",
#               (filename, prediction, timestamp))
#     conn.commit()
#     conn.close()
#
#
# def get_all_predictions():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("SELECT timestamp, filename, prediction FROM prediction_log ORDER BY timestamp DESC")
#     rows = c.fetchall()
#     conn.close()
#     return rows
#
#
# create_pneumonia_db()
#
#
# # --- Functions for Clearing State ---
# def reset_all_states():
#     """Resets the session state for all modules when going back to the main menu."""
#     st.session_state.app_mode = None
#     st.session_state.show_history = False
#
#     # Reset symptom module state
#     st.session_state.predicted = False
#     st.session_state.predicted_disease = ""
#     st.session_state.user_symptoms = []
#     st.session_state.selected_weights = []
#     st.session_state.show_desc = False
#     st.session_state.show_adv = False
#
#     # Reset pneumonia module state
#     st.session_state.uploaded_image = None
#     st.session_state.selected_image = None
#     st.session_state.key_suffix = str(uuid.uuid4())
#
#
# # -------------------- SIDEBAR --------------------
# st.sidebar.title("Dashboard")
#
# if st.session_state.app_mode == "Pneumonia":
#     st.sidebar.header("🔍 Choose a Sample X-ray")
#     sample_dir = "sample_images"
#     if os.path.exists(sample_dir):
#         sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
#         selected_sample = st.sidebar.selectbox(
#             "Pick a sample image", [""] + sample_images, key="sample_" + st.session_state.key_suffix
#         )
#         if selected_sample:
#             st.session_state.selected_image = selected_sample
#             st.session_state.uploaded_image = None
#     else:
#         st.sidebar.warning("`sample_images` directory not found.")
#
# st.sidebar.markdown("---")
# st.sidebar.header("📂 Prediction History")
#
# if st.sidebar.button("🔎 View Full History"):
#     st.session_state.show_history = True
#
# if st.session_state.show_history:
#     st.title("🗂️ View Prediction History")
#     history_choice = st.radio(
#         "Select which history to view:",
#         ["Heart Disease", "Symptom-Based Disease", "Pneumonia Detection"],
#         index=None, key="history_choice"
#     )
#
#     if history_choice == "Heart Disease":
#         history_file = "heart_history.csv"
#         if os.path.exists(history_file):
#             df = pd.read_csv(history_file, header=None, names=["DateTime", "Inputs", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found for Heart Disease.")
#
#     elif history_choice == "Symptom-Based Disease":
#         history_file = "user_history.csv"
#         if os.path.exists(history_file):
#             df = pd.read_csv(history_file, header=None, names=["DateTime", "Symptoms", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found for Symptom-Based Disease.")
#
#     elif history_choice == "Pneumonia Detection":
#         history = get_all_predictions()
#         if history:
#             df = pd.DataFrame(history, columns=["Timestamp", "Filename", "Prediction"])
#             st.dataframe(df)
#         else:
#             st.info("No prediction history found for Pneumonia Detection.")
#
#     if st.button("❌ Close History"):
#         st.session_state.show_history = False
#         st.rerun()
#
#     st.stop()
#
# # -------------------- MAIN APP LOGIC --------------------
#
# if not st.session_state.app_mode:
#     st.title("🩺 Unified Health Prediction Platform")
#     st.markdown("Choose the type of prediction you want to perform from the options below.")
#     st.markdown("---")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         if st.button("💓 Heart Disease"):
#             st.session_state.app_mode = "Heart"
#             st.rerun()
#     with col2:
#         if st.button("🦠 Symptom-Based"):
#             st.session_state.app_mode = "Symptom"
#             st.rerun()
#     with col3:
#         if st.button("🫁 Pneumonia X-ray"):
#             st.session_state.app_mode = "Pneumonia"
#             st.rerun()
#     st.markdown("---")
#     st.info("You can view the prediction history for any module using the sidebar.")
#
# if st.session_state.app_mode:
#     if st.button("🔙 Back to Main Menu"):
#         reset_all_states()
#         st.rerun()
#
# # -------------------- MODULE 1: HEART DISEASE --------------------
# if st.session_state.app_mode == "Heart":
#     st.header("💓 Heart Disease Prediction using KNN")
#     try:
#         model = joblib.load("models/knn_heart_disease_model.pkl")
#         age = st.number_input("Age", 20, 100, 50)
#         sex = st.selectbox("Sex", ["Male", "Female"])
#         # ... (rest of the heart disease inputs)
#         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#         chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
#         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
#         thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
#         exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
#         oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
#         cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
#         restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
#
#         cp_dict = {"Typical angina": [1, 0, 0], "Atypical angina": [0, 1, 0], "Non-anginal": [0, 0, 1],
#                    "Asymptomatic": [0, 0, 0]}
#         restecg_dict = {"Normal": [1, 0], "ST-T abnormality": [0, 1], "Left ventricular hypertrophy": [0, 0]}
#
#         if st.button("🧠 Predict Heart Disease"):
#             features = [age, 1 if sex == "Male" else 0, trestbps, chol, 1 if fbs == "Yes" else 0, thalch,
#                         1 if exang == "Yes" else 0, oldpeak, *cp_dict[cp], *restecg_dict[restecg]]
#             prediction = model.predict([features])[0]
#             result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
#             st.success(f"🩺 Prediction: **{result}**")
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             inputs = f"age={age}, sex={sex}, bp={trestbps}, chol={chol}, fbs={fbs}, thalch={thalch}, exang={exang}, oldpeak={oldpeak}"
#             with open("heart_history.csv", "a") as f:
#                 f.write(f"{now},\"{inputs}\",{result}\n")
#     except FileNotFoundError:
#         st.error(
#             "Model file not found. Please ensure `models/knn_heart_disease_model.pkl` is in the correct directory.")
#
# # -------------------- MODULE 2: SYMPTOM-BASED DISEASE --------------------
# elif st.session_state.app_mode == "Symptom":
#     st.header("🦠 Symptom-Based Disease Prediction")
#     try:
#         model = joblib.load("models/final_model.pkl")
#         label_encoder = joblib.load("models/label_encoder.pkl")
#         all_features = joblib.load("models/all_symptoms.pkl")
#         disease_info = joblib.load("models/disease_info.pkl")
#         df_sev = pd.read_csv("data/Symptom-severity.csv")
#
#         df_sev.columns = ['Symptom', 'weight']
#         df_sev['Symptom'] = df_sev['Symptom'].str.lower().str.strip().str.replace('_', ' ')
#         severity_dict = dict(zip(df_sev['Symptom'], df_sev['weight']))
#         symptom_features = [feat for feat in all_features if
#                             feat not in ['total_severity_score', 'num_reported_symptoms', 'avg_severity_score']]
#
#         st.multiselect("Select Your Symptoms:", options=sorted(symptom_features), key="user_symptoms")
#
#         if st.button("🔍 Predict Disease"):
#             if not st.session_state.user_symptoms:
#                 st.warning("Please select at least one symptom.")
#             else:
#                 symptom_vector = [1 if s in st.session_state.user_symptoms else 0 for s in symptom_features]
#                 selected_weights = [severity_dict.get(s.lower().strip(), 0) for s in st.session_state.user_symptoms]
#                 total_severity = sum(selected_weights)
#                 num_symptoms = len(st.session_state.user_symptoms)
#                 avg_severity = total_severity / num_symptoms if num_symptoms else 0
#                 input_vector = np.array(symptom_vector + [total_severity, num_symptoms, avg_severity]).reshape(1, -1)
#
#                 pred_encoded = model.predict(input_vector)[0]
#                 predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
#                 st.success(f"🧬 Predicted Disease: **{predicted_disease.title()}**")
#
#                 st.session_state.predicted = True
#                 st.session_state.predicted_disease = predicted_disease
#                 st.session_state.selected_weights = selected_weights
#                 st.session_state.show_desc = False
#                 st.session_state.show_adv = False
#
#                 now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 symptom_str = ", ".join(st.session_state.user_symptoms)
#                 with open("user_history.csv", "a") as f:
#                     f.write(f"{now},\"{symptom_str}\",{predicted_disease}\n")
#
#         if st.session_state.predicted:
#             with st.expander("🧠 More Analysis", expanded=True):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("📋 Show Description & Precautions", key="desc_btn"):
#                         st.session_state.show_desc = not st.session_state.show_desc
#                 with col2:
#                     if st.button("📊 Advanced Analysis", key="adv_btn"):
#                         st.session_state.show_adv = not st.session_state.show_adv
#
#                 if st.session_state.show_desc:
#                     key = st.session_state.predicted_disease.lower().strip()
#                     if key in disease_info:
#                         st.markdown("### 📝 Description:")
#                         st.info(disease_info[key]["desc"])
#                         st.markdown("### 🛡️ Precautions:")
#                         for step in disease_info[key]["precautions"]:
#                             st.markdown(f"- {step}")
#                     else:
#                         st.info("ℹ️ No description or precaution info available for this disease.")
#
#                 if st.session_state.show_adv:
#                     if hasattr(model, "feature_importances_"):
#                         importances = model.feature_importances_
#                         feature_df = pd.DataFrame({
#                             "Feature": all_features,
#                             "Importance": importances
#                         }).sort_values(by="Importance", ascending=False)
#                         st.markdown("### 🧠 Top Contributing Features")
#                         st.dataframe(feature_df.head(10))
#
#                     st.markdown("### 🔥 Symptom Severity Chart")
#                     fig, ax = plt.subplots()
#                     ax.barh(st.session_state.user_symptoms, st.session_state.selected_weights, color="orange")
#                     ax.set_xlabel("Severity Weight")
#                     ax.set_title("Symptom Severity")
#                     st.pyplot(fig)
#
#             if st.button("🔄 Clear & Predict Again"):
#                 st.session_state.predicted = False
#                 st.session_state.predicted_disease = ""
#                 st.session_state.selected_weights = []
#                 st.session_state.show_desc = False
#                 st.session_state.show_adv = False
#                 st.rerun()
#
#     except FileNotFoundError as e:
#         st.error(f"A required file was not found: {e.filename}. Please check your `models` and `data` directories.")
#
#
# # -------------------- MODULE 3: PNEUMONIA DETECTION --------------------
# elif st.session_state.app_mode == "Pneumonia":
#     st.header("🫁 Pneumonia Detection from Chest X-rays")
#     st.write("Upload a chest X-ray image or choose a sample from the sidebar.")
#     try:
#         model, device = load_pneumonia_model()
#         class_names = ['NORMAL', 'PNEUMONIA']
#         transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
#
#         image, filename = None, None
#         uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],
#                                          key="file_" + st.session_state.key_suffix)
#
#         if uploaded_file:
#             image = Image.open(uploaded_file).convert("RGB")
#             st.session_state.uploaded_image = image
#             st.session_state.selected_image = None
#             filename = uploaded_file.name
#         elif st.session_state.selected_image:
#             image_path = os.path.join("sample_images", st.session_state.selected_image)
#             image = Image.open(image_path).convert("RGB")
#             st.session_state.uploaded_image = image
#             filename = st.session_state.selected_image
#
#         if image:
#             st.image(image, caption=filename, use_container_width=True)
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("🔬 Predict"):
#                     input_tensor = transform(image).unsqueeze(0).to(device)
#                     with torch.no_grad():
#                         outputs = model(input_tensor)
#                         _, predicted = torch.max(outputs, 1)
#                         prediction = class_names[predicted.item()]
#                         st.success(f"Prediction: **{prediction}**")
#                         if filename:
#                             insert_prediction(filename, prediction)
#             with col2:
#                 if st.button("🔄 Clear Image"):
#                     st.session_state.uploaded_image = None
#                     st.session_state.selected_image = None
#                     st.session_state.key_suffix = str(uuid.uuid4())
#                     st.rerun()
#     except FileNotFoundError:
#         st.error("Model file not found. Please ensure `pneumonia_resnet18_model.pth` is present.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")




import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import datetime
import os
import sqlite3
import uuid
import matplotlib.pyplot as plt
import cohere
from dotenv import load_dotenv

load_dotenv()

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Unified Health Prediction", layout="wide")

# Custom CSS for chat bubbles
st.markdown("""
    <style>
        .stChatMessage[data-testid="stChatMessage"] {
            background-color: #FFFFFF; /* White background */
            border: 1px solid #D3D3D3; /* Light grey border for visibility */
            border-radius: 10px;      /* Rounded corners for a softer look */
            color: #000000;           /* Black text for readability on a white background */
            padding: 1em;             /* Adds some space inside the bubble */
        }
    </style>
""", unsafe_allow_html=True)


# -------------------- GEMINI API CONFIGURATION --------------------
# st.sidebar.title("Configuration") # <-- Optional: you can remove this title
# api_key = st.sidebar.text_input("Enter your Google API Key:", type="password", key="api_key_input") # <-- REMOVE THIS LINE

# --- Get API key from environment variables ---
api_key = os.getenv("COHERE_API_KEY") # <-- ADD THIS LINE

if not api_key:
    # If key is not found, show a warning in the sidebar
    st.sidebar.warning("Cohere API Key not found. Please add it to your .env file.")
    co = None # Set client to None if no key
else:
    try:
        # Initialize the Cohere Client
        co = cohere.Client(api_key)
        st.sidebar.success("Cohere AI client loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to configure Cohere client: {e}")
        co = None


# Function to get response from Cohere
def get_cohere_response(disease, query, chat_history):
    # System prompt to guide the model (called a preamble in Cohere)
    preamble = f"""
    You are a helpful and compassionate medical assistant chatbot.
    A user has been predicted to have '{disease}'. They are asking for more information.
    Your role is to provide clear, helpful, and safe information about the disease.

    IMPORTANT SAFETY RULE:
    - ALWAYS start your response with a clear, bold disclaimer: "Disclaimer: I am an AI assistant and not a medical professional. The information I provide is for educational purposes only. Please consult with a qualified healthcare provider for any medical advice, diagnosis, or treatment."
    - Do not provide a diagnosis. The diagnosis '{disease}' is already given.
    - Provide a general overview of the disease.
    - Explain common symptoms and potential risk factors.
    - Suggest general wellness and care tips that may help manage symptoms.
    - Crucially, advise the user on when it is important to see a doctor (e.g., if symptoms worsen, if they have specific concerns).
    - Maintain a supportive and understanding tone.
    """

    # Convert Streamlit chat history to Cohere's format
    cohere_chat_history = []
    for message in chat_history:
        if message["role"] == "user":
            cohere_chat_history.append({"role": "USER", "message": message["content"]})
        elif message["role"] == "assistant":
            cohere_chat_history.append({"role": "CHATBOT", "message": message["content"]})

    # Make the API call to Cohere's chat endpoint
    response = co.chat(
        model='command-r-plus',  # A powerful model for this task
        message=query,
        preamble=preamble,
        chat_history=cohere_chat_history
    )

    return response.text

# -------------------- SESSION STATE INITIALIZATION --------------------
def init_session_state():
    # General app state
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    # State for Symptom-Based module
    if "predicted" not in st.session_state:
        st.session_state.predicted = False
    if "predicted_disease" not in st.session_state:
        st.session_state.predicted_disease = ""
    if "user_symptoms" not in st.session_state:
        st.session_state.user_symptoms = []
    if "selected_weights" not in st.session_state:
        st.session_state.selected_weights = []
    if "show_desc" not in st.session_state:
        st.session_state.show_desc = False
    if "show_adv" not in st.session_state:
        st.session_state.show_adv = False
    # New state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # State for Pneumonia module
    if "key_suffix" not in st.session_state:
        st.session_state.key_suffix = str(uuid.uuid4())
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None


init_session_state()


# -------------------- HELPER FUNCTIONS & MODEL LOADING --------------------

# --- Pneumonia Module Functions ---
@st.cache_resource
def load_pneumonia_model():
    """Loads the pre-trained ResNet18 model for pneumonia detection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load("pneumonia_resnet18_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device


# --- Database Functions for Pneumonia History ---
DB_NAME = "predictions.db"


def create_pneumonia_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(filename, prediction):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO prediction_log (filename, prediction, timestamp) VALUES (?, ?, ?)",
              (filename, prediction, timestamp))
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT timestamp, filename, prediction FROM prediction_log ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows


create_pneumonia_db()


# --- Functions for Clearing State ---
def reset_all_states():
    """Resets the session state for all modules when going back to the main menu."""
    st.session_state.app_mode = None
    st.session_state.show_history = False

    # Reset symptom module state
    st.session_state.predicted = False
    st.session_state.predicted_disease = ""
    st.session_state.user_symptoms = []
    st.session_state.selected_weights = []
    st.session_state.show_desc = False
    st.session_state.show_adv = False
    st.session_state.chat_history = []

    # Reset pneumonia module state
    st.session_state.uploaded_image = None
    st.session_state.selected_image = None
    st.session_state.key_suffix = str(uuid.uuid4())


# -------------------- SIDEBAR --------------------
st.sidebar.title("Dashboard")

if st.session_state.app_mode == "Pneumonia":
    st.sidebar.header("🔍 Choose a Sample X-ray")
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        selected_sample = st.sidebar.selectbox(
            "Pick a sample image", [""] + sample_images, key="sample_" + st.session_state.key_suffix
        )
        if selected_sample:
            st.session_state.selected_image = selected_sample
            st.session_state.uploaded_image = None
    else:
        st.sidebar.warning("`sample_images` directory not found.")

st.sidebar.markdown("---")
st.sidebar.header("📂 Prediction History")

if st.sidebar.button("🔎 View Full History"):
    st.session_state.show_history = True

if st.session_state.show_history:
    st.title("🗂️ View Prediction History")
    history_choice = st.radio(
        "Select which history to view:",
        ["Heart Disease", "Symptom-Based Disease", "Pneumonia Detection"],
        index=None, key="history_choice"
    )

    if history_choice == "Heart Disease":
        history_file = "heart_history.csv"
        if os.path.exists(history_file):
            df = pd.read_csv(history_file, header=None, names=["DateTime", "Inputs", "Prediction"])
            st.dataframe(df)
        else:
            st.info("No prediction history found for Heart Disease.")

    elif history_choice == "Symptom-Based Disease":
        history_file = "user_history.csv"
        if os.path.exists(history_file):
            df = pd.read_csv(history_file, header=None, names=["DateTime", "Symptoms", "Prediction"])
            st.dataframe(df)
        else:
            st.info("No prediction history found for Symptom-Based Disease.")

    elif history_choice == "Pneumonia Detection":
        history = get_all_predictions()
        if history:
            df = pd.DataFrame(history, columns=["Timestamp", "Filename", "Prediction"])
            st.dataframe(df)
        else:
            st.info("No prediction history found for Pneumonia Detection.")

    if st.button("❌ Close History"):
        st.session_state.show_history = False
        st.rerun()

    st.stop()

# -------------------- MAIN APP LOGIC --------------------

if not st.session_state.app_mode:
    st.title("🩺 Unified Health Prediction Platform")
    st.markdown("Choose the type of prediction you want to perform from the options below.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💓 Heart Disease"):
            st.session_state.app_mode = "Heart"
            st.rerun()
    with col2:
        if st.button("🦠 Symptom-Based"):
            st.session_state.app_mode = "Symptom"
            st.rerun()
    with col3:
        if st.button("🫁 Pneumonia X-ray"):
            st.session_state.app_mode = "Pneumonia"
            st.rerun()
    st.markdown("---")
    st.info("You can view the prediction history for any module using the sidebar.")

if st.session_state.app_mode:
    if st.button("🔙 Back to Main Menu"):
        reset_all_states()
        st.rerun()

# -------------------- MODULE 1: HEART DISEASE --------------------
if st.session_state.app_mode == "Heart":
    st.header("💓 Heart Disease Prediction using KNN")
    try:
        model = joblib.load("models/knn_heart_disease_model.pkl")
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])

        cp_dict = {"Typical angina": [1, 0, 0], "Atypical angina": [0, 1, 0], "Non-anginal": [0, 0, 1],
                   "Asymptomatic": [0, 0, 0]}
        restecg_dict = {"Normal": [1, 0], "ST-T abnormality": [0, 1], "Left ventricular hypertrophy": [0, 0]}

        if st.button("🧠 Predict Heart Disease"):
            features = [age, 1 if sex == "Male" else 0, trestbps, chol, 1 if fbs == "Yes" else 0, thalch,
                        1 if exang == "Yes" else 0, oldpeak, *cp_dict[cp], *restecg_dict[restecg]]
            prediction = model.predict([features])[0]
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            st.success(f"🩺 Prediction: **{result}**")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inputs = f"age={age}, sex={sex}, bp={trestbps}, chol={chol}, fbs={fbs}, thalch={thalch}, exang={exang}, oldpeak={oldpeak}"
            with open("heart_history.csv", "a") as f:
                f.write(f"{now},\"{inputs}\",{result}\n")
    except FileNotFoundError:
        st.error(
            "Model file not found. Please ensure `models/knn_heart_disease_model.pkl` is in the correct directory.")

# -------------------- MODULE 2: SYMPTOM-BASED DISEASE --------------------
elif st.session_state.app_mode == "Symptom":
    st.header("🦠 Symptom-Based Disease Prediction")

    # Layout for the symptom prediction module
    pred_col, chat_col = st.columns([1, 1])

    with pred_col:
        st.subheader("1. Predict Your Condition")
        try:
            model = joblib.load("models/final_model.pkl")
            label_encoder = joblib.load("models/label_encoder.pkl")
            all_features = joblib.load("models/all_symptoms.pkl")
            disease_info = joblib.load("models/disease_info.pkl")
            df_sev = pd.read_csv("data/Symptom-severity.csv")

            df_sev.columns = ['Symptom', 'weight']
            df_sev['Symptom'] = df_sev['Symptom'].str.lower().str.strip().str.replace('_', ' ')
            severity_dict = dict(zip(df_sev['Symptom'], df_sev['weight']))
            symptom_features = [feat for feat in all_features if
                                feat not in ['total_severity_score', 'num_reported_symptoms', 'avg_severity_score']]

            st.multiselect("Select Your Symptoms:", options=sorted(symptom_features), key="user_symptoms")

            if st.button("🔍 Predict Disease"):
                if not st.session_state.user_symptoms:
                    st.warning("Please select at least one symptom.")
                else:
                    symptom_vector = [1 if s in st.session_state.user_symptoms else 0 for s in symptom_features]
                    selected_weights = [severity_dict.get(s.lower().strip(), 0) for s in st.session_state.user_symptoms]
                    total_severity = sum(selected_weights)
                    num_symptoms = len(st.session_state.user_symptoms)
                    avg_severity = total_severity / num_symptoms if num_symptoms else 0
                    input_vector = np.array(symptom_vector + [total_severity, num_symptoms, avg_severity]).reshape(1,
                                                                                                                   -1)

                    pred_encoded = model.predict(input_vector)[0]
                    predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
                    st.success(f"🧬 Predicted Disease: **{predicted_disease.title()}**")

                    st.session_state.predicted = True
                    st.session_state.predicted_disease = predicted_disease
                    st.session_state.selected_weights = selected_weights
                    # Reset previous analysis and chat
                    st.session_state.show_desc = False
                    st.session_state.show_adv = False
                    st.session_state.chat_history = []

                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    symptom_str = ", ".join(st.session_state.user_symptoms)
                    with open("user_history.csv", "a") as f:
                        f.write(f"{now},\"{symptom_str}\",{predicted_disease}\n")

            if st.session_state.predicted:
                with st.expander("🧠 More Analysis", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📋 Show Description & Precautions", key="desc_btn"):
                            st.session_state.show_desc = not st.session_state.show_desc
                    with col2:
                        if st.button("📊 Advanced Analysis", key="adv_btn"):
                            st.session_state.show_adv = not st.session_state.show_adv

                    if st.session_state.show_desc:
                        key = st.session_state.predicted_disease.lower().strip()
                        if key in disease_info:
                            st.markdown("### 📝 Description:")
                            st.info(disease_info[key]["desc"])
                            st.markdown("### 🛡️ Precautions:")
                            for step in disease_info[key]["precautions"]:
                                st.markdown(f"- {step}")
                        else:
                            st.info("ℹ️ No description or precaution info available for this disease.")

                    if st.session_state.show_adv:
                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                            feature_df = pd.DataFrame({
                                "Feature": all_features,
                                "Importance": importances
                            }).sort_values(by="Importance", ascending=False)
                            st.markdown("### 🧠 Top Contributing Features")
                            st.dataframe(feature_df.head(10))

                        st.markdown("### 🔥 Symptom Severity Chart")
                        fig, ax = plt.subplots()
                        ax.barh(st.session_state.user_symptoms, st.session_state.selected_weights, color="orange")
                        ax.set_xlabel("Severity Weight")
                        ax.set_title("Symptom Severity")
                        st.pyplot(fig)

                if st.button("🔄 Clear & Predict Again"):
                    st.session_state.predicted = False
                    st.session_state.predicted_disease = ""
                    st.session_state.selected_weights = []
                    st.session_state.show_desc = False
                    st.session_state.show_adv = False
                    st.session_state.chat_history = []
                    st.rerun()

        except FileNotFoundError as e:
            st.error(f"A required file was not found: {e.filename}. Please check your `models` and `data` directories.")

    # THIS IS THE NEW BLOCK TO ADD
    with chat_col:
        st.subheader("2. Chat with AI Assistant")

        # The 'co' client is initialized in the Cohere AI Configuration section
        if co is None:
            # This message is shown if the API key is missing or invalid
            st.warning("Please configure your Cohere API Key in the .env file to enable the chatbot.")

        elif st.session_state.predicted_disease:
            st.markdown(f"Ask me anything about **{st.session_state.predicted_disease.title()}**.")

            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input logic you provided
            if prompt := st.chat_input("What would you like to know?"):
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get Cohere response
                with st.spinner("Thinking..."):
                    response_text = get_cohere_response(
                        st.session_state.predicted_disease,
                        prompt,
                        st.session_state.chat_history[:-1]  # Exclude the last user message from history
                    )
                    # Add AI response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                # Rerun to display the new message
                st.rerun()

        else:  # This handles the case where no disease has been predicted yet
            st.info("Please predict a disease first to activate the chat.")

# -------------------- MODULE 3: PNEUMONIA DETECTION --------------------
elif st.session_state.app_mode == "Pneumonia":
    st.header("🫁 Pneumonia Detection from Chest X-rays")
    st.write("Upload a chest X-ray image or choose a sample from the sidebar.")
    try:
        model, device = load_pneumonia_model()
        class_names = ['NORMAL', 'PNEUMONIA']
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

        image, filename = None, None
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],
                                         key="file_" + st.session_state.key_suffix)

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            st.session_state.selected_image = None
            filename = uploaded_file.name
        elif st.session_state.selected_image:
            image_path = os.path.join("sample_images", st.session_state.selected_image)
            image = Image.open(image_path).convert("RGB")
            st.session_state.uploaded_image = image
            filename = st.session_state.selected_image

        if image:
            st.image(image, caption=filename, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔬 Predict"):
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted = torch.max(outputs, 1)
                        prediction = class_names[predicted.item()]
                        st.success(f"Prediction: **{prediction}**")
                        if filename:
                            insert_prediction(filename, prediction)
            with col2:
                if st.button("🔄 Clear Image"):
                    st.session_state.uploaded_image = None
                    st.session_state.selected_image = None
                    st.session_state.key_suffix = str(uuid.uuid4())
                    st.rerun()
    except FileNotFoundError:
        st.error("Model file not found. Please ensure `pneumonia_resnet18_model.pth` is present.")
    except Exception as e:
        st.error(f"An error occurred: {e}")







# import streamlit as st
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import numpy as np
# import pandas as pd
# import joblib
# import datetime
# import os
# import sqlite3
# import uuid
# import matplotlib.pyplot as plt
#
# # Import the new, more stable library for Gemini
# from google.cloud import aiplatform
# import vertexai
# from vertexai.generative_models import GenerativeModel
#
# # -------------------- PAGE SETUP AND STYLING --------------------
# st.set_page_config(page_title="Unified Health Prediction", layout="wide")
#
# # Custom CSS to make the chat bubbles white with a visible border
# st.markdown("""
#     <style>
#         .stChatMessage[data-testid="stChatMessage"] {
#             background-color: #FFFFFF;
#             border: 1px solid #D3D3D3;
#             border-radius: 10px;
#             color: #000000;
#             padding: 1em;
#         }
#     </style>
# """, unsafe_allow_html=True)
#
# # -------------------- VERTEX AI (GEMINI) INITIALIZATION --------------------
# # This section initializes the connection to Google's Vertex AI.
# # It uses your Project ID for authentication.
#
# try:
#     # --- ⚠️ ACTION REQUIRED: Replace with your Project ID below ---
#     PROJECT_ID = "health-prediction-432508"  # e.g., "my-first-project-123456"
#     REGION = "us-central1"
#
#     # Initialize the Vertex AI SDK
#     aiplatform.init(project=PROJECT_ID, location=REGION)
#
#     # Load the Gemini Pro model and start a chat session
#     model = GenerativeModel("gemini-pro")
#     chat = model.start_chat()
#
#     GEMINI_CONFIGURED = True
# except Exception as e:
#     # If initialization fails, store the error to display in the app
#     GEMINI_CONFIGURED = False
#     GEMINI_ERROR = e
#
#
# # -------------------- SESSION STATE INITIALIZATION --------------------
# def init_session_state():
#     keys_to_init = {
#         "app_mode": None, "show_history": False, "predicted": False,
#         "predicted_disease": "", "user_symptoms": [], "selected_weights": [],
#         "show_desc": False, "show_adv": False, "chat_history": [],
#         "key_suffix": str(uuid.uuid4()), "uploaded_image": None, "selected_image": None
#     }
#     for key, val in keys_to_init.items():
#         if key not in st.session_state:
#             st.session_state[key] = val
#
#
# init_session_state()
#
#
# # -------------------- HELPER FUNCTIONS & MODEL LOADING --------------------
# @st.cache_resource
# def load_pneumonia_model():
#     """Loads the pre-trained ResNet18 model for pneumonia detection."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = models.resnet18(weights=None)
#     model.fc = torch.nn.Linear(512, 2)
#     model.load_state_dict(torch.load("pneumonia_resnet18_model.pth", map_location=device))
#     model.to(device)
#     model.eval()
#     return model, device
#
#
# DB_NAME = "predictions.db"
#
#
# def create_pneumonia_db():
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute(
#         "CREATE TABLE IF NOT EXISTS prediction_log (id INTEGER PRIMARY KEY, filename TEXT, prediction TEXT, timestamp TEXT)")
#     conn.commit()
#     conn.close()
#
#
# def insert_prediction(filename, prediction):
#     conn = sqlite3.connect(DB_NAME)
#     c = conn.cursor()
#     c.execute("INSERT INTO prediction_log (filename, prediction, timestamp) VALUES (?, ?, ?)",
#               (filename, prediction, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
#     conn.commit()
#     conn.close()
#
#
# def get_all_predictions():
#     conn = sqlite3.connect(DB_NAME)
#     try:
#         rows = pd.read_sql_query("SELECT timestamp, filename, prediction FROM prediction_log ORDER BY timestamp DESC",
#                                  conn)
#     except pd.io.sql.DatabaseError:
#         rows = pd.DataFrame(columns=["timestamp", "filename", "prediction"])
#     conn.close()
#     return rows
#
#
# create_pneumonia_db()
#
#
# def get_gemini_response(disease, query):
#     """Sends a query to the Gemini model using the Vertex AI SDK."""
#     if not GEMINI_CONFIGURED:
#         return f"Error: Gemini AI could not be initialized. Details: {GEMINI_ERROR}"
#
#     prompt = f"""You are a helpful and compassionate medical assistant chatbot.
#     The user has been preliminarily identified with '{disease}'. Their question is: '{query}'
#     Please provide a clear, helpful, and safe response.
#     **IMPORTANT SAFETY RULE:** ALWAYS start your response with this exact disclaimer:
#     **Disclaimer: I am an AI assistant and not a medical professional. The information I provide is for educational purposes only. Please consult with a qualified healthcare provider for any medical advice, diagnosis, or treatment.**"""
#
#     response = chat.send_message(prompt)
#     return response.text
#
#
# def reset_all_states():
#     # Store the current app mode before clearing
#     current_mode = st.session_state.app_mode
#     # Re-initialize all keys to their defaults
#     init_session_state()
#     # Set the app_mode back for a clean navigation
#     st.session_state.app_mode = None
#     # Specifically clear user inputs that might persist
#     st.session_state.user_symptoms = []
#     st.session_state.chat_history = []
#
#
# # -------------------- SIDEBAR --------------------
# st.sidebar.title("Dashboard")
#
# if st.session_state.app_mode == "Pneumonia":
#     st.sidebar.header("🔍 Choose a Sample X-ray")
#     sample_dir = "sample_images"
#     if os.path.exists(sample_dir):
#         sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
#         selected_sample = st.sidebar.selectbox("Pick a sample image", [""] + sample_images,
#                                                key="sample_" + st.session_state.key_suffix)
#         if selected_sample:
#             st.session_state.selected_image = selected_sample
#             st.session_state.uploaded_image = None
#     else:
#         st.sidebar.warning("`sample_images` directory not found.")
#
# st.sidebar.markdown("---")
# st.sidebar.header("📂 Prediction History")
# if st.sidebar.button("🔎 View Full History"):
#     st.session_state.show_history = True
#
# # -------------------- MAIN APP LOGIC --------------------
# if st.session_state.show_history:
#     st.title("🗂️ View Prediction History")
#     history_choice = st.radio("Select which history to view:",
#                               ["Heart Disease", "Symptom-Based Disease", "Pneumonia Detection"], index=None,
#                               key="history_choice")
#
#     if history_choice == "Heart Disease":
#         if os.path.exists("heart_history.csv"):
#             st.dataframe(pd.read_csv("heart_history.csv", header=None, names=["DateTime", "Inputs", "Prediction"]))
#         else:
#             st.info("No prediction history found for Heart Disease.")
#
#     elif history_choice == "Symptom-Based Disease":
#         if os.path.exists("user_history.csv"):
#             st.dataframe(pd.read_csv("user_history.csv", header=None, names=["DateTime", "Symptoms", "Prediction"]))
#         else:
#             st.info("No prediction history found for Symptom-Based Disease.")
#
#     elif history_choice == "Pneumonia Detection":
#         st.dataframe(get_all_predictions())
#
#     if st.button("❌ Close History"):
#         st.session_state.show_history = False
#         st.rerun()
#     st.stop()
#
# if not st.session_state.app_mode:
#     st.title("🩺 Unified Health Prediction Platform")
#     st.markdown("Choose the type of prediction you want to perform from the options below.")
#     st.markdown("---")
#     col1, col2, col3 = st.columns(3)
#     if col1.button("💓 Heart Disease"): st.session_state.app_mode = "Heart"; st.rerun()
#     if col2.button("🦠 Symptom-Based"): st.session_state.app_mode = "Symptom"; st.rerun()
#     if col3.button("🫁 Pneumonia X-ray"): st.session_state.app_mode = "Pneumonia"; st.rerun()
# else:
#     if st.button("🔙 Back to Main Menu"):
#         reset_all_states()
#         st.rerun()
#
# # -------------------- MODULE 1: HEART DISEASE --------------------
# if st.session_state.app_mode == "Heart":
#     st.header("💓 Heart Disease Prediction using KNN")
#     try:
#         model = joblib.load("models/knn_heart_disease_model.pkl")
#         age = st.number_input("Age", 20, 100, 50)
#         sex = st.selectbox("Sex", ["Male", "Female"])
#         trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#         chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
#         fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
#         thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
#         exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
#         oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
#         cp = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
#         restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
#
#         cp_dict = {"Typical angina": [1, 0, 0], "Atypical angina": [0, 1, 0], "Non-anginal": [0, 0, 1],
#                    "Asymptomatic": [0, 0, 0]}
#         restecg_dict = {"Normal": [1, 0], "ST-T abnormality": [0, 1], "Left ventricular hypertrophy": [0, 0]}
#
#         if st.button("🧠 Predict Heart Disease"):
#             features = [age, 1 if sex == "Male" else 0, trestbps, chol, 1 if fbs == "Yes" else 0, thalch,
#                         1 if exang == "Yes" else 0, oldpeak, *cp_dict[cp], *restecg_dict[restecg]]
#             prediction = model.predict([features])[0]
#             result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
#             st.success(f"🩺 Prediction: **{result}**")
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             inputs = f"age={age}, sex={sex}, bp={trestbps}, chol={chol}, fbs={fbs}, thalch={thalch}, exang={exang}, oldpeak={oldpeak}"
#             with open("heart_history.csv", "a") as f:
#                 f.write(f"{now},\"{inputs}\",{result}\n")
#     except FileNotFoundError:
#         st.error(
#             "Model file not found. Please ensure `models/knn_heart_disease_model.pkl` is in the correct directory.")
#
#
# # -------------------- MODULE 2: SYMPTOM-BASED DISEASE --------------------
# elif st.session_state.app_mode == "Symptom":
#     st.header("🦠 Symptom-Based Disease Prediction")
#     pred_col, chat_col = st.columns([1, 1])
#
#     with pred_col:
#         st.subheader("1. Predict Your Condition")
#         try:
#             model = joblib.load("models/final_model.pkl")
#             label_encoder = joblib.load("models/label_encoder.pkl")
#             all_features = joblib.load("models/all_symptoms.pkl")
#             disease_info = joblib.load("models/disease_info.pkl")
#             df_sev = pd.read_csv("data/Symptom-severity.csv")
#             df_sev.columns = ['Symptom', 'weight']
#             df_sev['Symptom'] = df_sev['Symptom'].str.lower().str.strip().str.replace('_', ' ')
#             severity_dict = dict(zip(df_sev['Symptom'], df_sev['weight']))
#             symptom_features = [feat for feat in all_features if
#                                 feat not in ['total_severity_score', 'num_reported_symptoms', 'avg_severity_score']]
#             st.multiselect("Select Your Symptoms:", options=sorted(symptom_features), key="user_symptoms")
#
#             if st.button("🔍 Predict Disease"):
#                 if not st.session_state.user_symptoms:
#                     st.warning("Please select at least one symptom.")
#                 else:
#                     symptom_vector = [1 if s in st.session_state.user_symptoms else 0 for s in symptom_features]
#                     selected_weights = [severity_dict.get(s.lower().strip(), 0) for s in st.session_state.user_symptoms]
#                     total_severity = sum(selected_weights)
#                     num_symptoms = len(st.session_state.user_symptoms)
#                     avg_severity = total_severity / num_symptoms if num_symptoms else 0
#                     input_vector = np.array(symptom_vector + [total_severity, num_symptoms, avg_severity]).reshape(1,
#                                                                                                                    -1)
#                     pred_encoded = model.predict(input_vector)[0]
#                     predicted_disease = label_encoder.inverse_transform([pred_encoded])[0]
#                     st.success(f"🧬 Predicted Disease: **{predicted_disease.title()}**")
#                     st.session_state.predicted = True
#                     st.session_state.predicted_disease = predicted_disease
#                     st.session_state.selected_weights = selected_weights
#                     st.session_state.show_desc = False
#                     st.session_state.show_adv = False
#                     st.session_state.chat_history = []
#                     now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     symptom_str = ", ".join(st.session_state.user_symptoms)
#                     with open("user_history.csv", "a") as f:
#                         f.write(f"{now},\"{symptom_str}\",{predicted_disease}\n")
#
#             if st.session_state.predicted:
#                 with st.expander("🧠 More Analysis"):
#                     col1, col2 = st.columns(2)
#                     if col1.button(
#                         "📋 Show Description & Precautions"): st.session_state.show_desc = not st.session_state.show_desc
#                     if col2.button("📊 Advanced Analysis"): st.session_state.show_adv = not st.session_state.show_adv
#                     if st.session_state.show_desc:
#                         key = st.session_state.predicted_disease.lower().strip()
#                         if key in disease_info:
#                             st.markdown("### 📝 Description:");
#                             st.info(disease_info[key]["desc"])
#                             st.markdown("### 🛡️ Precautions:");
#                             for step in disease_info[key]["precautions"]: st.markdown(f"- {step}")
#                         else:
#                             st.info("ℹ️ No info available for this disease.")
#                     if st.session_state.show_adv:
#                         st.markdown("### 🔥 Symptom Severity Chart");
#                         fig, ax = plt.subplots();
#                         ax.barh(st.session_state.user_symptoms, st.session_state.selected_weights, color="orange");
#                         ax.set_xlabel("Severity Weight");
#                         st.pyplot(fig)
#                 if st.button("🔄 Clear & Predict Again"):
#                     st.session_state.predicted = False;
#                     st.session_state.predicted_disease = "";
#                     st.session_state.user_symptoms = [];
#                     st.rerun()
#
#         except FileNotFoundError as e:
#             st.error(f"A required file was not found: {e.filename}. Please check your project folders.")
#
#     with chat_col:
#         st.subheader("2. Chat with AI Assistant")
#         if not GEMINI_CONFIGURED:
#             st.error(f"Chatbot is not available. Error: {GEMINI_ERROR}")
#         elif st.session_state.predicted_disease:
#             st.markdown(f"Ask me anything about **{st.session_state.predicted_disease.title()}**.")
#
#             for message in st.session_state.chat_history:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])
#
#             if prompt := st.chat_input("What would you like to know?"):
#                 st.session_state.chat_history.append({"role": "user", "content": prompt})
#                 with st.chat_message("user"):
#                     st.markdown(prompt)
#
#                 with st.spinner("Thinking..."):
#                     response_text = get_gemini_response(st.session_state.predicted_disease, prompt)
#                     st.session_state.chat_history.append({"role": "assistant", "content": response_text})
#                     st.rerun()
#         else:
#             st.info("Please predict a disease first to activate the chat.")
#
# # -------------------- MODULE 3: PNEUMONIA DETECTION --------------------
# elif st.session_state.app_mode == "Pneumonia":
#     st.header("🫁 Pneumonia Detection from Chest X-rays")
#     st.write("Upload a chest X-ray image or choose a sample from the sidebar.")
#     try:
#         model, device = load_pneumonia_model()
#         class_names = ['NORMAL', 'PNEUMONIA']
#         transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
#
#         image, filename = None, None
#         uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],
#                                          key="file_" + st.session_state.key_suffix)
#
#         if uploaded_file:
#             image = Image.open(uploaded_file).convert("RGB")
#             filename = uploaded_file.name
#         elif st.session_state.selected_image:
#             image_path = os.path.join("sample_images", st.session_state.selected_image)
#             image = Image.open(image_path).convert("RGB")
#             filename = st.session_state.selected_image
#
#         if image:
#             st.image(image, caption=filename, use_container_width=True)
#             col1, col2 = st.columns(2)
#             if col1.button("🔬 Predict"):
#                 input_tensor = transform(image).unsqueeze(0).to(device)
#                 with torch.no_grad():
#                     outputs = model(input_tensor)
#                     _, predicted = torch.max(outputs, 1)
#                     prediction = class_names[predicted.item()]
#                     st.success(f"Prediction: **{prediction}**")
#                     if filename: insert_prediction(filename, prediction)
#             if col2.button("🔄 Clear Image"):
#                 st.session_state.uploaded_image = None
#                 st.session_state.selected_image = None
#                 st.session_state.key_suffix = str(uuid.uuid4())
#                 st.rerun()
#     except FileNotFoundError:
#         st.error("Model file not found. Please ensure `pneumonia_resnet18_model.pth` is present.")
#     except Exception as e:
#         st.error(f"An error occurred: {e}")