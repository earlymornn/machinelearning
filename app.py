import os
from io import BytesIO

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download

from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Fresh",
    "Highly Fresh",
    "Not Fresh"
]

@st.cache_resource
def load_cnn_model():
    path = os.path.join(
        MODEL_DIR,
        "cnn_fish_freshness.keras"
    )
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_knn_model():
    model = joblib.load(
        os.path.join(MODEL_DIR, "knn_glcm_model.pkl")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR, "knn_glcm_scaler.pkl")
    )

    return model, scaler


@st.cache_resource
def load_svm_model():
    model_path = hf_hub_download(
        repo_id="earlymornn/fishfresh-svm",
        filename="svm_hog_model.pkl"
    )
    return joblib.load(model_path)


@st.cache_resource
def load_svm_scaler():
    scaler_path = hf_hub_download(
        repo_id="earlymornn/fishfresh-svm",
        filename="svm_hog_scaler.pkl"
    )
    return joblib.load(scaler_path)

def preprocess_cnn(image_bytes):

    img = tf.keras.utils.load_img(
        BytesIO(image_bytes),
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

def extract_glcm(image_bytes):

    img = Image.open(
        BytesIO(image_bytes)
    ).convert("L")

    img = img.resize(IMAGE_SIZE)

    image_array = np.array(
        img,
        dtype=np.uint8
    )

    glcm = graycomatrix(
        image_array,
        distances=[1],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=256,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation"
    ]

    features = []

    for prop in properties:

        values = graycoprops(
            glcm,
            prop
        )

        features.extend(
            values.flatten()
        )

    return np.array(
        features
    ).reshape(1, -1)

def extract_hog(image_bytes):

    img = Image.open(
        BytesIO(image_bytes)
    ).convert("L")

    img = img.resize(IMAGE_SIZE)

    image_array = np.array(
        img,
        dtype=np.uint8
    )

    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    )

    return features.reshape(1, -1)

def get_class_name(label):

    if isinstance(label, (int, np.integer)):
        return CLASS_NAMES[int(label)]

    return str(label)

def predict_cnn(image_bytes):

    model = load_cnn_model()

    img_array = preprocess_cnn(image_bytes)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(prediction[0])
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        prediction[0][predicted_index] * 100
    )

    return predicted_class, confidence


def predict_knn(image_bytes):

    model, scaler = load_knn_model()

    features = extract_glcm(image_bytes)

    features_scaled = scaler.transform(
        features
    )

    prediction = model.predict(
        features_scaled
    )

    predicted_class = get_class_name(
        prediction[0]
    )

    probabilities = model.predict_proba(
        features_scaled
    )

    probability = float(
        np.max(probabilities[0]) * 100
    )

    return predicted_class, probability


def predict_svm(image_bytes):

    model, scaler = load_svm_model()

    features = extract_hog(image_bytes)

    features_scaled = scaler.transform(
        features
    )

    prediction = model.predict(
        features_scaled
    )

    predicted_class = get_class_name(
        prediction[0]
    )

    return predicted_class

st.set_page_config(
    page_title="FishFresh - Fish Freshness Classification",
    page_icon="🐟",
    layout="centered"
)

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #f1f7fb;
    color: #243746;
}

/* MAIN CONTENT */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* GLOBAL TEXT */
.stApp p,
.stApp label,
.stApp li,
.stApp span,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #243746;
}

/* HEADINGS */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #123e52 !important;
    font-weight: 750;
}

/* HEADER */
.main-title {
    text-align: center;
    color: #126782 !important;
    font-size: 42px;
    font-weight: 850;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #496779 !important;
    font-size: 16px;
    margin-bottom: 30px;
}

/* DIVIDER */
hr {
    border-color: #cbdde8 !important;
}

/* SELECTBOX */
.stSelectbox label,
.stFileUploader label {
    color: #243746 !important;
    font-weight: 700 !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #243746 !important;
}

.stSelectbox div[data-baseweb="popover"] {
    background-color: #ffffff !important;
}

.stSelectbox div[data-baseweb="popover"] * {
    color: #243746 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px solid #cbdde8;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stFileUploader"] section {
    background-color: #f8fbfd !important;
}

[data-testid="stFileUploader"] section * {
    color: #243746 !important;
}

/* BUTTON */
.stButton button {
    background-color: #126782 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-weight: 700 !important;
}

.stButton button p {
    color: #ffffff !important;
}

.stButton button:hover {
    background-color: #0d5269 !important;
    color: #ffffff !important;
}

/* INFO BOX */
[data-testid="stAlert"] {
    background-color: #e2f1f8 !important;
    border: 1px solid #a9ccdc !important;
}

[data-testid="stAlert"] * {
    color: #17465b !important;
}

/* RESULT CARD */
.result-card {
    background-color: #ffffff;
    border: 2px solid #126782;
    border-radius: 16px;
    padding: 25px;
    margin-top: 20px;
    margin-bottom: 20px;
    text-align: center;
}

.result-label {
    color: #496779 !important;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-value {
    color: #126782 !important;
    font-size: 30px;
    font-weight: 850;
    margin: 0;
}

/* METRIC */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #cbdde8;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricLabel"] * {
    color: #496779 !important;
}

[data-testid="stMetricValue"] {
    color: #126782 !important;
}

/* CAPTION */
.stCaption,
[data-testid="stCaptionContainer"] * {
    color: #526b7a !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #526b7a !important;
    font-size: 14px;
    padding-top: 15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🐟 FishFresh</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fish Freshness Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

st.markdown("### About FishFresh")

st.write(
    """
    FishFresh is a fish freshness classification system
    that analyzes fish-eye images using three different
    machine learning methods.

    Upload an image of a fish eye and select the
    classification method to predict its freshness level.
    """
)

st.markdown("### Choose Classification Method")

method_options = {
    "CNN": "cnn",
    "KNN + GLCM": "knn_glcm",
    "SVM + HOG": "svm_hog"
}

selected_method = st.selectbox(
    "Select a classification method:",
    options=list(method_options.keys())
)

method = method_options[selected_method]


method_descriptions = {
    "CNN": (
        "Convolutional Neural Network analyzes "
        "the image directly to identify visual patterns."
    ),

    "KNN + GLCM": (
        "K-Nearest Neighbors classifies fish freshness "
        "using texture features extracted with GLCM."
    ),

    "SVM + HOG": (
        "Support Vector Machine classifies fish freshness "
        "using edge and shape features extracted with HOG."
    )
}

st.info(
    method_descriptions[selected_method]
)

st.markdown("### Upload Fish Image")

uploaded_file = st.file_uploader(
    "Choose a fish-eye image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    try:

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Fish Image",
            use_container_width=True
        )

        st.write("")

        if st.button(
            "Predict Freshness",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                f"Analyzing image using {selected_method}..."
            ):

                if method == "cnn":

                    predicted_class, confidence = (
                        predict_cnn(image_bytes)
                    )

                elif method == "knn_glcm":

                    predicted_class, confidence = (
                        predict_knn(image_bytes)
                    )

                else:

                    predicted_class = predict_svm(
                        image_bytes
                    )

                    confidence = None

            st.divider()

            st.markdown("### Classification Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        FRESHNESS LEVEL
                    </div>
                    <div class="result-value">
                        {predicted_class}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"**Classification Method:** {selected_method}"
            )

            if confidence is not None:

                metric_label = (
                    "Confidence"
                    if method == "cnn"
                    else "Prediction Probability"
                )

                st.metric(
                    label=metric_label,
                    value=f"{confidence:.2f}%"
                )

            else:

                st.info(
                    "Confidence score is not available "
                    "for this SVM model."
                )

            st.caption(
                "Prediction results are generated by "
                "the selected machine learning model."
            )

    except Exception as e:

        st.error(
            f"An error occurred: {str(e)}"
        )

st.divider()

st.markdown(
    """
    <div class="footer">
        FishFresh - Fish Freshness Classification System
        <br>
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
