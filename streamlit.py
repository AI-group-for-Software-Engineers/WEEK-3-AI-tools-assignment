import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

#  Page Setup 
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered",
)

#  Compatibility Fix for NumPy 2.0 
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128

#  Load CNN Model 
@st.cache_resource
def load_cnn_model():
    return load_model("mnist_cnn_model.h5")

model = load_cnn_model()

#  Navigation Menu (Tabs) 
tab = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔢 Prediction", "👥 Team"],
)

#  Helper Functions 
def preprocess_image(image):
    """Convert uploaded or drawn image to model input format."""
    image = image.convert("L")  # grayscale
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def predict_digit(image):
    """Run prediction using CNN model."""
    preds = model.predict(image)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return predicted_class, confidence, preds


# ======================================================
# 🏠 HOME PAGE
# ======================================================
if tab == "🏠 Home":
    st.title("🧠 Handwritten Digit Recognition using CNN")
    st.markdown("""
    Welcome to the **MNIST Digit Classifier Web App**!  

    This project demonstrates the power of **Artificial Intelligence (AI)** and **Deep Learning**
    in recognizing handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

    ---
    ### 🎯 Purpose
    The goal of this web app is to:
    - Demonstrate how AI can interpret images through deep learning.
    - Allow users to **upload** or **draw digits** for live predictions.
    - Show how AI contributes to **innovation and education**, supporting **UN SDG 9: Industry, Innovation & Infrastructure**.

    ---
    ### 🚀 How to Use
    1. Go to the **🔢 Prediction** tab in the sidebar.  
    2. Upload a digit image or draw one using the built-in canvas.  
    3. Get instant predictions from the trained CNN model.

    ---
    **Technology Stack:** Streamlit | TensorFlow | NumPy | Pillow  
    **Dataset:** MNIST Handwritten Digits (70,000 samples)
    """)

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
        caption="Example MNIST Digits Dataset",
        use_column_width=True
    )

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and TensorFlow | MNIST CNN Model")


# ======================================================
# 🔢 PREDICTION PAGE
# ======================================================
elif tab == "🔢 Prediction":
    st.title("🔢 Handwritten Digit Prediction")
    st.write("Upload an image or draw a digit below to let the model recognize it!")

    #  File Upload Section 
    st.header("📁 Upload an Image")
    uploaded_file = st.file_uploader(
        "Upload a digit image (PNG or JPG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        processed_img = preprocess_image(image)
        digit, confidence, preds = predict_digit(processed_img)

        st.success(f"Predicted Digit: **{digit}**")
        st.info(f"Confidence: {confidence*100:.2f}%")
        st.bar_chart(preds[0])

    st.markdown("---")

    #  Drawing Section 
    st.header("✍️ Draw a Digit")
    st.write("Use your mouse or touch to draw a digit below (0–9):")

    canvas = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype(np.uint8))
        processed_img = preprocess_image(img)
        digit, confidence, preds = predict_digit(processed_img)

        st.success(f"Predicted Digit: **{digit}**")
        st.info(f"Confidence: {confidence*100:.2f}%")
        st.bar_chart(preds[0])

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and TensorFlow | MNIST CNN Model")


# ======================================================
# 👥 TEAM PAGE
# ======================================================
elif tab == "👥 Team":
    st.title("👥 Project Team Members")
    st.markdown("""
    ### Group Members
    | Name | Email |
    |------|--------|
    | **Obuye Emmanuel Chukwuemeke** | [obuyeemmanuel@gmail.com](mailto:obuyeemmanuel@gmail.com) |
    | **Anthonia Othetheaso** | [t27613850@gmail.com](mailto:t27613850@gmail.com) |
    | **Eunice Fagbemide** | [eunicefagbemide@gmail.com](mailto:eunicefagbemide@gmail.com) |
    | **Daizy Jepchumba Kiplagat** | [daisyjepchum@gmail.com](mailto:daisyjepchum@gmail.com) |
    | **Mark Ireri** | [markire.07@gmail.com](mailto:markire.07@gmail.com) |

    

    ---
    ### 💡 About
    This web application was created as part of the **PLP Week 3 Assignment on AI Tools**, 
    demonstrating the use of **Streamlit**, **TensorFlow**, and **Computer Vision** 
    to build an interactive and educational AI web app.
    """)

    st.markdown("---")
    st.caption("Developed by the PLP AI Tools Week 3 Group | 2025")
