import cv2
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open the video file

video = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(video.get(3)), int(video.get(4))))

# Read the video frame by frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert frame from BGR to RGB
    frame_rgb = frame[..., ::-1]

    # Perform inference
    results = model(frame_rgb)

    # Draw bounding boxes on the frame
    output_frame = results.render()[0]

    # Write the frame with bounding boxes to the output video
    out.write(output_frame)

    # Display the frame
    cv2.imshow('Frame', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and video writer objects
video.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
