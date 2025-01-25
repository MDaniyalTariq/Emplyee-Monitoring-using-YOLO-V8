from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("last.pt")

# Define path to video file
source = "demo.mp4"

# Open the video file
cap = cv2.VideoCapture(source)

# Frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Ground truth for the number of objects (replace with actual values)
ground_truth_count_person = 10  # Example: number of people in the video
ground_truth_count_cabinet = 5  # Example: number of cabinets in the video

# Counters for detected objects
detected_person_count = 0
detected_cabinet_count = 0
true_positives = 0
false_positives = 0
false_negatives = 0

# Frame index initialization
frame_idx = 0


def calculate_iou(boxA, boxB):
    # Calculate the intersection over union (IoU) of two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# Initialize video writer to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Run inference on the video
while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit the loop if no more frames are available

    results = model(frame)

    # Loop through the detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Get the class ID

            if cls_id == 1:  # Class 1: 'person'
                label = "Person"
                color = (255, 0, 0)  # Blue color for 'person'
                detected_person_count += 1
            elif cls_id == 0:  # Class 0: 'cabinet'
                label = "Cabinet"
                color = (0, 255, 0)  # Green color for 'cabinet'
                detected_cabinet_count += 1
            else:
                continue  # Skip other classes

            # Get bounding box coordinates (x1, y1, x2, y2)
            bbox = box.xyxy[0]
            x1, y1, x2, y2 = map(int, bbox)

            # Example ground truth bounding box for testing (replace with actual ground truth data)
            ground_truth_bbox = [50, 50, 150, 150]  # Replace with the actual ground truth bounding box

            # Calculate IoU with ground truth
            iou = calculate_iou([x1, y1, x2, y2], ground_truth_bbox)

            # Set a threshold for IoU to consider it a true positive
            if iou > 0.5:
                true_positives += 1
            else:
                false_positives += 1

            # Calculate the time in the video when the object is detected
            time_in_seconds = frame_idx / fps
            time_label = f"Time: {time_in_seconds:.2f} sec"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({iou:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, time_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame with detections to the output video
    out.write(frame)
    frame_idx += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final statistics
print(f"Detected Person Count: {detected_person_count}")
print(f"Detected Cabinet Count: {detected_cabinet_count}")
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {ground_truth_count_person - true_positives}")
