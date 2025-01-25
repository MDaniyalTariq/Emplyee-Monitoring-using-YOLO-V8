import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# Load the YOLO model
model = YOLO("last.pt")

# Streamlit app setup
st.set_page_config(page_title="Employee Monitoring System", layout="wide")
st.title("Employee Monitoring System")
st.sidebar.title("Options")

# Sidebar for video input
uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

# Initialize counters
ground_truth_count_person = st.sidebar.number_input("Ground Truth Count - Person", min_value=0, value=10)
ground_truth_count_cabinet = st.sidebar.number_input("Ground Truth Count - Cabinet", min_value=0, value=5)

detected_person_count = 0
detected_cabinet_count = 0
true_positives = 0
false_positives = 0
false_negatives = 0

# IoU calculation function
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Process video if uploaded
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 1:  # Person
                    label = "Person"
                    color = (255, 0, 0)
                    detected_person_count += 1
                elif cls_id == 0:  # Cabinet
                    label = "Cabinet"
                    color = (0, 255, 0)
                    detected_cabinet_count += 1
                else:
                    continue

                bbox = box.xyxy[0]
                x1, y1, x2, y2 = map(int, bbox)
                ground_truth_bbox = [50, 50, 150, 150]  # Replace with actual ground truth

                iou = calculate_iou([x1, y1, x2, y2], ground_truth_bbox)

                if iou > 0.5:
                    true_positives += 1
                else:
                    false_positives += 1

                time_in_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                time_label = f"Time: {time_in_seconds:.2f} sec"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({iou:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, time_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    # Display results
    st.sidebar.subheader("Results")
    st.sidebar.write(f"Detected Person Count: {detected_person_count}")
    st.sidebar.write(f"Detected Cabinet Count: {detected_cabinet_count}")
    st.sidebar.write(f"True Positives: {true_positives}")
    st.sidebar.write(f"False Positives: {false_positives}")
    st.sidebar.write(f"False Negatives: {ground_truth_count_person - true_positives}")
else:
    st.info("Please upload a video to start monitoring.")
