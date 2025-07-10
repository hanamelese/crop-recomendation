
import streamlit as st
import base64
import joblib
from utils.nlp_utils import parse_input_to_features_or_chat

# 1. Page config
st.set_page_config(page_title="Crop ChatBot 🌱", page_icon="💬", layout="centered")

# 2. Encode local image
def get_base64_image_from_file(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

image_path = "data/potato.jpg"
image_base64 = get_base64_image_from_file(image_path)

# 3. CSS styling
st.markdown(f"""
    <style>
        .background {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("data:image/jpg;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
            filter: blur(1px) brightness(40%);
            z-index: -1;
        }}
        .stApp {{
            background: transparent;
            min-height: 100vh;
            position: relative;
            z-index: 0;
        }}
        div[data-testid="stChatInput"] {{
            position: fixed;
            bottom: 0;
            width: 80%;
            max-width: 800px;
            margin: 0 auto;
            background: linear-gradient(135deg, #fffacd, #f0fff0);
            z-index: 6;
            padding: 10px;
            border-radius: 12px;
        }}
        .chat-bubble {{
            animation: fadeInUp 0.7s ease-in-out;
            box-shadow: 0px 4px 16px rgba(15, 10, 10, 20);
            transition: all 0.3s ease-in-out;
        }}
        @keyframes fadeInUp {{
            0% {{ opacity: 0; transform: translateY(55px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    <div class="background"></div>
""", unsafe_allow_html=True)

# 4. Title
st.title("🌾 Crop Recommendation ChatBot")

# 5. Load ML model
model = joblib.load("model/crop_recommendation_model.pkl")

# 6. Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# 7. Custom bubble renderer
def render_bubble(role, content):
    if role == "user":
        icon = "👨‍🌾"
        bg = "rgba(254, 247, 224, 0.2)"
        border = "#e6b800"
        text = "#fffff0"
        width = "65%"
    else:
        icon = "🤖"
        bg = "rgba(230, 244, 234, 0.2)"
        border = "#34a853"
        text = "#fffff0"
        width = "95%"

    st.markdown(f"""
        <div class="chat-bubble" style="background-color:{bg}; padding:7px; border-radius:12px;
             border-left: 4px solid {border}; margin-bottom:10px; color:{text}; 
             box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15); width:{width};">
            <strong>{icon}</strong> {content}
        </div>
    """, unsafe_allow_html=True)

# 8. Show past messages
for msg in st.session_state.messages:
    render_bubble(msg["role"], msg["content"])

# 9. User input
prompt = st.chat_input("Ask about the best crop for your land...")

# 10. Process input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_bubble("user", prompt)

    try:
        result = parse_input_to_features_or_chat(prompt)

        if "features" in result:
            features = [float(result["features"][k]) for k in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]

            probs = model.predict_proba([features])[0]
            top_indices = probs.argsort()[-3:][::-1]
            top_crops = [model.classes_[i] for i in top_indices]

            crops = ", ".join(top_crops)

            response = (
                f"Based on your input, the following crops are well-suited for cultivation in your environment: {crops}.<br><br>"
                f"These recommendations are made considering your soil and climate conditions, which include:<br>"
                f"- Nitrogen: {features[0]}<br>"
                f"- Phosphorus: {features[1]}<br>"
                f"- Potassium: {features[2]}<br>"
                f"- Temperature: {features[3]}°C<br>"
                f"- Humidity: {features[4]}%<br>"
                f"- pH: {features[5]}<br>"
                f"- Rainfall: {features[6]} mm<br><br>"
                f"These nutrient levels, along with other environmental factors, align well with the optimal growing conditions for the suggested crops."
            )
        else:
            response = result["chat"]

    except Exception as e:
        response = (
            "🤖 I'm happy to chat! 🌱 You can ask me about:\n"
            "- Best crops for your farm\n"
            "- Soil nutrients and what they mean\n"
            "- Or just say hello!\n"
            f"(Error: {str(e)})"
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
    render_bubble("assistant", response)




