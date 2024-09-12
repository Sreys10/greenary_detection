import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from datetime import datetime

st.title("Greenary Detection")

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Sidebar for control options
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, step=0.05)
min_contour_area = st.sidebar.slider("Minimum Contour Area", 500, 5000, 1000, step=100)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert image to BGR (OpenCV format)
    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert frame to HSV color space for green color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours of green regions
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # YOLO object detection on the original frame
    results = model(frame, conf=conf_threshold)

    # Check for greenery and draw bounding boxes
    greenery_detected = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > min_contour_area:
            greenery_detected = True
            # Draw rectangle around green area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label it as "Greenery"
            cv2.putText(frame, "Greenery", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert BGR to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show the processed image
    st.image(frame_rgb, caption="Processed Image", channels="RGB")

    # Display detection result
    if greenery_detected:
        st.success("Greenery detected in the image!")
    else:
        st.warning("No greenery detected in the image.")
else:
    st.info("Upload an image to start detection.")
