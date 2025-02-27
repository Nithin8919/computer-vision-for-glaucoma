import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/glaucoma_model_fp16.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load full model for Grad-CAM
full_model = tf.keras.models.load_model('models/glaucoma_model_efficientnet.h5')

def estimate_cdr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) >= 2:
        disc = max(contours, key=cv2.contourArea)
        cup = sorted(contours, key=cv2.contourArea, reverse=True)[1]
        disc_diameter = cv2.boundingRect(disc)[2]
        cup_diameter = cv2.boundingRect(cup)[2]
        return cup_diameter / disc_diameter if disc_diameter > 0 else 0.5
    return 0.5

def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    return "Referable Glaucoma" if pred > 0.5 else "Non-Referable Glaucoma", pred

def get_gradcam(image, model):
    img_array = np.expand_dims(np.array(image.resize((224, 224))) / 255.0, axis=0)
    
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    
    if not last_conv_layer:
        st.error("⚠️ No convolutional layer found in the model for Grad-CAM!")
        return None

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = cv2.resize(heatmap, (224, 224))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    
    img = np.array(image.resize((224, 224)))
    
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    
    return Image.fromarray(superimposed_img)

# Streamlit App UI
st.set_page_config(page_title="🩺 Glaucoma Vision Pro", layout="wide", page_icon="👁️")

st.title("🚀 AI-Powered Glaucoma Detection")
st.sidebar.header("📂 Upload Fundus Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="🖼 Uploaded Image", use_column_width=True)
    
    with st.spinner("✨ Analyzing Image..."):
        label, confidence = predict(image)
        img_np = np.array(image)
        cdr = estimate_cdr(img_np)
        severity = "🟢 Mild" if cdr < 0.6 else "🟠 Moderate" if cdr < 0.8 else "🔴 Severe"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Results")
        st.write(f"✅ **Prediction:** {label}")
        st.write(f"📈 **Confidence:** {confidence:.2%}")
        st.write(f"👁 **Estimated CDR:** {cdr:.2f}")
        st.write(f"⚠️ **Severity:** {severity}")
    
    with col2:
        st.subheader("🔬 Original Image")
        st.image(image, caption="Fundus Image", use_column_width=True)
    
    if st.sidebar.button("🔍 Explain with Grad-CAM"):
        with st.spinner("📡 Generating heatmap..."):
            gradcam_img = get_gradcam(image, full_model)
            if gradcam_img is not None:
                st.subheader("🧠 Model Focus (Grad-CAM)")
                st.image(gradcam_img, caption="Heatmap Overlay", use_column_width=True)
    
    if st.sidebar.button("📄 Generate Report"):
        report_content = f"""
        🩺 Glaucoma Detection Report
        -----------------------------
        ✅ Prediction: {label}
        📈 Confidence: {confidence:.2%}
        👁 Estimated CDR: {cdr:.2f}
        ⚠️ Severity: {severity}
        """
        st.sidebar.download_button(label="📥 Download Report", data=report_content, file_name="glaucoma_report.txt", mime="text/plain")
