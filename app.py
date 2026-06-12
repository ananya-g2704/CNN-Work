import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CNN Digit Classifier",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
}

.main-title {
    text-align:center;
    color:white;
    font-size:48px;
    font-weight:bold;
}

.sub-title {
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

.result-box {
    background:#22c55e;
    color:white;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

.conf-box {
    background:#3b82f6;
    color:white;
    padding:15px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/cnn_model.keras"
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🧠 CNN Digit Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Handwritten Digit Recognition using CNN</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:

    st.header("Project Information")

    st.write("""
    **Dataset:** MNIST

    **Model:** CNN

    **Framework:** TensorFlow

    **Frontend:** Streamlit
    """)

    st.write("### Architecture")

    st.code("""
Input
 ↓
Conv2D(32)
 ↓
MaxPooling
 ↓
Conv2D(64)
 ↓
MaxPooling
 ↓
Flatten
 ↓
Dense(128)
 ↓
Dense(10)
""")

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=300)

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------

    gray = image.convert("L")

    # Invert image
    gray = ImageOps.invert(gray)

    img_array = np.array(gray)

    # Thresholding
    img_array = np.where(
        img_array > 80,
        255,
        0
    ).astype(np.uint8)

    # Find bounding box
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if rows.any() and cols.any():

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        img_array = img_array[
            y_min:y_max+1,
            x_min:x_max+1
        ]

    # Convert back to image
    digit_img = Image.fromarray(img_array)

    # Resize to MNIST size
    digit_img = digit_img.resize((28, 28))

    processed = np.array(
        digit_img
    ).astype("float32") / 255.0

    with col2:
        st.subheader("Processed Image")
        st.image(processed, width=300)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    input_img = processed.reshape(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(
        input_img,
        verbose=0
    )

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-box">
        Predicted Digit : {digit}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="conf-box">
        Confidence : {confidence:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### Prediction Probabilities")

    probs = prediction[0]

    for i in range(10):

        st.progress(float(probs[i]))

        st.write(
            f"Digit {i}: {probs[i]*100:.2f}%"
        )