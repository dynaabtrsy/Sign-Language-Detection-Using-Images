import cv2
import torch
import pathlib

# Fix for the 'PosixPath' error on Windows machines
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
pathlib.PosixPath = temp

# Set confidence to ultra-low just to see IF it detects anything at all
model.conf = 0.01

cap = cv2.VideoCapture(0)
print("Webcam started! Look at the terminal window below to see real-time detection counts.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Center crop to prevent stretching
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # --- COLOR FIX: Convert BGR to RGB ---
    rgb_frame = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
    
    # Pass the RGB frame to the model
    results = model(rgb_frame, size=416)

    # Print detections directly to your black terminal window
    # This will show us if the model sees anything at 1% confidence
    print(results.pandas().xyxy[0]) 

    # Render results back onto the original frame for display
    annotated_frame = results.render()[0]
    # Convert back to BGR so OpenCV displays the colors correctly
    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('Sign Language Detection Prototype', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
