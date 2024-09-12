import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from datetime import datetime

st.title("Greenary Detection")

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Sidebar for control options
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, step=0.05)
min_contour_area = st.sidebar.slider("Minimum Contour Area", 500, 5000, 1000, step=100)

# Store bounding box records
bounding_box_records = []

# Toggle camera
camera_on = st.sidebar.checkbox("Turn Camera On", value=True)

if camera_on:
    # Start video capture
    source = cv2.VideoCapture(0)

    # Initialize variables
    frame_count = 0
    start_time = time.time()

    # Streamlit layout for displaying video
    frame_placeholder = st.empty()

    while source.isOpened():
        ret, frame = source.read()
        if not ret:
            st.warning("Failed to capture video stream.")
            break

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

        # Draw bounding boxes for green regions and log the information
        greenery_detected = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out small areas
            if cv2.contourArea(contour) > min_contour_area:
                greenery_detected = True
                # Draw rectangle around green area
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Label it as "Greenery"
                cv2.putText(frame, "Greenery", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Log the bounding box information with timestamp and FPS
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                bounding_box_records.append({
                    "timestamp": current_time,
                    "coordinates": (x, y, w, h),
                    "fps": fps
                })

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the Streamlit frame display
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Calculate FPS
        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        st.sidebar.text(f"FPS: {fps:.2f}")

        # Break the loop when the "Stop" button is pressed, using a unique key per iteration
        if st.sidebar.button("Stop", key=f"stop_button_{frame_count}"):
            break

    # Release the video source
    source.release()

else:
    # When camera is off, show the bounding box records
    st.sidebar.subheader("Bounding Box Records")
    if bounding_box_records:
        for record in bounding_box_records:
            st.sidebar.text(f"Time: {record['timestamp']}, Coordinates: {record['coordinates']}, FPS: {record['fps']:.2f}")
    else:
        st.sidebar.text("No bounding boxes recorded yet.")
