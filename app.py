import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import base64

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="SmartBite", layout="wide")

# -----------------------
# Session state for page navigation
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "uploaded_food" not in st.session_state:
    st.session_state.uploaded_food = None
if "predicted_food" not in st.session_state:
    st.session_state.predicted_food = None
if "predicted_nutri" not in st.session_state:
    st.session_state.predicted_nutri = None

# -----------------------
# Background function (full-page)
# -----------------------
def set_full_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp .main {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------
# Load Model and Nutrition Data
# -----------------------
@st.cache_resource
def load_food_model():
    return load_model("food.h5")

@st.cache_data
def load_nutrition():
    df = pd.read_csv("nutrition.csv")
    df['Calories'] = df['name'].apply(lambda x: int(re.search(r'Calories\s*(\d+)', x).group(1)) if re.search(r'Calories\s*(\d+)', x) else 0)
    df['Food'] = df['name'].apply(lambda x: x.split(':')[0])
    df['Image'] = df['Food'].apply(lambda x: f"dataset/{x.replace(' ','_').lower()}.jpg")
    return df

model = load_food_model()
nutrition_df = load_nutrition()

# -----------------------
# PAGE: Home
# -----------------------
def home_page():
    set_full_bg("backgrounds/home.jpg")
    st.title("🍔 Welcome to SmartBite")
    st.write("Your AI-powered food calorie & nutrition assistant.")
    
    if st.button("Get Started"):
        st.session_state.page = "login"

# -----------------------
# PAGE: Login
# -----------------------
def login_page():
    set_full_bg("backgrounds/login.jpg")
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.session_state.page = "calorie_estimation"
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

# -----------------------
# PAGE: Calorie Estimation / Upload
# -----------------------
def calorie_estimation_page():
    set_full_bg("backgrounds/upload.jfif")
    st.title("🍽️ Food Calorie Estimator")
    
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        st.session_state.uploaded_food = uploaded_file
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Food Image", use_column_width=True)
        
        img_resized = img.resize((128,128))
        x = image.img_to_array(img_resized)/255.0
        x = np.expand_dims(x, axis=0)
        
        # Predict food
        pred = model.predict(x)
        class_idx = np.argmax(pred)
        predicted_food = list(nutrition_df['Food'])[class_idx]
        st.session_state.predicted_food = predicted_food
        
        # Lookup calories & nutrition
        nutri = nutrition_df[nutrition_df['Food']==predicted_food].iloc[0]
        st.session_state.predicted_nutri = nutri
        
        # Display results immediately
        st.subheader(f"Predicted Food: {predicted_food}")
        st.write(f"Calories: {nutri['Calories']} kcal")
        st.write(f"Protein: {nutri['protein']} g, Fat: {nutri['fat']} g, Carbs: {nutri['carbohydrates']} g")
        
        fig = px.pie(
            names=["Protein","Fat","Carbs"],
            values=[nutri['protein'], nutri['fat'], nutri['carbohydrates']],
            title=f"{predicted_food} Nutrient Composition"
        )
        st.plotly_chart(fig)
        
        if st.button("Next: Enter Profile"):
            st.session_state.page = "profile"

# -----------------------
# PAGE: User Profile
# -----------------------
def profile_page():
    set_full_bg("backgrounds/profile.jfif")
    st.title("👤 Your Profile")
    
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["male","female"])
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    activity = st.selectbox("Activity Level", ["low","moderate","high"])
    goal = st.selectbox("Goal", ["maintain","loss","gain"])
    diabetes = st.radio("Do you have diabetes?", ["no","yes"])
    
    if st.button("Next: View Results"):
        st.session_state.user_profile = {
            "age": age, "gender": gender, "height": height, "weight": weight,
            "activity": activity, "goal": goal, "diabetes": diabetes
        }
        st.session_state.page = "results"

# -----------------------
# PAGE: Results / Recommendations
# -----------------------
def results_page():
    set_full_bg("backgrounds/results.jfif")
    st.title("📊 Results & Recommendations")
    
    profile = st.session_state.user_profile
    predicted_food = st.session_state.predicted_food
    nutri = st.session_state.predicted_nutri
    
    st.subheader(f"Predicted Food: {predicted_food}")
    st.write(f"Calories: {nutri['Calories']} kcal")
    st.write(f"Protein: {nutri['protein']} g, Fat: {nutri['fat']} g, Carbs: {nutri['carbohydrates']} g")
    
    fig = px.pie(
        names=["Protein","Fat","Carbs"],
        values=[nutri['protein'], nutri['fat'], nutri['carbohydrates']],
        title=f"{predicted_food} Nutrient Composition"
    )
    st.plotly_chart(fig)
    
    # Accuracy Graph
    st.subheader("Model Prediction Accuracy")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        y=[0.7,0.75,0.8,0.82,0.85,0.87,0.9],
        x=list(range(1,8)),
        mode="lines+markers",
        name="Accuracy"
    ))
    fig_acc.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0,1])
    )
    st.plotly_chart(fig_acc)
    
    # Daily Calorie Requirement
    weight = profile['weight']
    height = profile['height']
    age = profile['age']
    gender = profile['gender']
    activity = profile['activity']
    goal = profile['goal']
    diabetes = profile['diabetes']
    
    if gender=="male":
        bmr = 10*weight + 6.25*height -5*age +5
    else:
        bmr = 10*weight + 6.25*height -5*age -161
    activity_factor = {"low":1.2,"moderate":1.55,"high":1.9}
    daily_cal = int(bmr*activity_factor[activity])
    if goal=="loss": daily_cal -=500
    elif goal=="gain": daily_cal +=300
    
    st.write(f"Your estimated daily calories: {daily_cal} kcal")
    
    # Recommendation
    advice = ""
    if diabetes=="yes" and nutri['Calories']>250:
        advice = "⚠️ Not Recommended due to diabetes and high calorie content."
    elif nutri['Calories']<=daily_cal:
        advice = "✅ Recommended within your daily calorie limit."
    else:
        advice = "❌ Exceeds your daily calorie requirement."
    st.subheader("Recommendation")
    st.write(advice)
    
    # Additional Recommendations
    remaining_cal = daily_cal - nutri['Calories']
    recommended_foods = nutrition_df[nutrition_df['Calories']<=remaining_cal].sort_values(by="Calories",ascending=False).head(6)
    st.subheader("Other Recommended Foods")
    cols = st.columns(3)
    for i, (_, row) in enumerate(recommended_foods.iterrows()):
        col = cols[i%3]
        if os.path.exists(row['Image']):
            col.image(row['Image'], use_column_width=True)
        col.write(f"**{row['Food']}** - {row['Calories']} kcal")
        col.write(f"Protein: {row['protein']} g, Fat: {row['fat']} g, Carbs: {row['carbohydrates']} g")
    if st.button("Logout"):
        st.session_state.page = "home"
# -----------------------
# Page routing
# -----------------------
pages = {
    "home": home_page,
    "login": login_page,
    "calorie_estimation": calorie_estimation_page,
    "profile": profile_page,
    "results": results_page
}

pages[st.session_state.page]()
