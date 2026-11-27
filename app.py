
import os
import numpy as np
import tensorflow as tf
import joblib

from io import BytesIO
from PIL import Image

from flask import Flask, request, jsonify, render_template

from skimage.feature import graycomatrix, graycoprops, hog


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Fresh",
    "Highly Fresh",
    "Not Fresh"
]

# Memuat model CNN
cnn_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "cnn_fish_freshness.keras")
)

# Memuat model KNN + GLCM
knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn_glcm_model.pkl")
)

knn_scaler = joblib.load(
    os.path.join(MODEL_DIR, "knn_glcm_scaler.pkl")
)

# Memuat model SVM + HOG
svm_model = joblib.load(
    os.path.join(MODEL_DIR, "svm_hog_model.pkl")
)

svm_scaler = joblib.load(
    os.path.join(MODEL_DIR, "svm_hog_scaler.pkl")
)


# PREPROCESSING CNN
def preprocess_cnn(image_bytes):

    img = tf.keras.utils.load_img(
        BytesIO(image_bytes),
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    img_array = tf.keras.utils.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# EKSTRAKSI FITUR GLCM
def extract_glcm(image_bytes):

    img = Image.open(BytesIO(image_bytes))
    img = img.convert("L")
    img = img.resize(IMAGE_SIZE)

    image_array = np.array(img, dtype=np.uint8)

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
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())

    return np.array(features).reshape(1, -1)


# EKSTRAKSI FITUR HOG
def extract_hog(image_bytes):

    img = Image.open(BytesIO(image_bytes))
    img = img.convert("L")
    img = img.resize(IMAGE_SIZE)

    image_array = np.array(img, dtype=np.uint8)

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


# KONVERSI LABEL PREDIKSI
def get_class_name(label):

    if isinstance(label, (int, np.integer)):
        return CLASS_NAMES[int(label)]

    return str(label)


# HALAMAN UTAMA
@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# PREDIKSI
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "Gambar belum diunggah"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Silakan pilih gambar terlebih dahulu"
        }), 400

    method = request.form.get("method", "cnn")

    valid_methods = [
        "cnn",
        "knn_glcm",
        "svm_hog"
    ]

    if method not in valid_methods:
        return jsonify({
            "error": "Metode klasifikasi tidak valid"
        }), 400

    try:

        image_bytes = file.read()

        if not image_bytes:
            return jsonify({
                "error": "File gambar kosong"
            }), 400

        # CNN
        if method == "cnn":

            img_array = preprocess_cnn(image_bytes)

            prediction = cnn_model.predict(
                img_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(prediction[0])
            )

            predicted_class = CLASS_NAMES[predicted_index]

            confidence = float(
                prediction[0][predicted_index] * 100
            )

            return jsonify({
                "method": "CNN",
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
                "confidence_label": "Confidence"
            })

        # KNN + GLCM
        elif method == "knn_glcm":

            features = extract_glcm(image_bytes)

            features_scaled = knn_scaler.transform(
                features
            )

            prediction = knn_model.predict(
                features_scaled
            )

            predicted_class = get_class_name(
                prediction[0]
            )

            probabilities = knn_model.predict_proba(
                features_scaled
            )

            confidence = float(
                np.max(probabilities[0]) * 100
            )

            return jsonify({
                "method": "KNN + GLCM",
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
                "confidence_label": "Prediction probability"
            })

        # SVM + HOG
        elif method == "svm_hog":

            features = extract_hog(image_bytes)

            features_scaled = svm_scaler.transform(
                features
            )

            prediction = svm_model.predict(
                features_scaled
            )

            predicted_class = get_class_name(
                prediction[0]
            )

            return jsonify({
                "method": "SVM + HOG",
                "prediction": predicted_class,
                "confidence": None,
                "confidence_label": "Not available"
            })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
