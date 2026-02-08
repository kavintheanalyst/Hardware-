import depthai as dai
import cv2
import numpy as np
import tensorflow as tf
import os
import ssl
# Update the model path to your actual model directory
model_path = "C:/Users/kavin/Desktop/Mr.Mini/object_detection"  # Ensure this path is correct

# Check if the directory exists
if not os.path.isdir(model_path):
    print(f"Error: The directory '{model_path}' does not exist.")
else:
    print(f"Model path: {model_path}")

    # Check the directory contents
    try:
        contents = os.listdir(model_path)
        print(f"Contents of the directory: {contents}")

        # Check if 'saved_model.pb' is in the directory
        if 'saved_model.pb' in contents:
            print("'saved_model.pb' file found.")
        else:
            print("'saved_model.pb' file not found.")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    # Load TensorFlow model if the file is present
    if 'saved_model.pb' in contents:
        model = tf.saved_model.load(model_path)

        # Initialize DepthAI pipeline
        pipeline = dai.Pipeline()

        # Define sources and outputs
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        cam_depth = pipeline.create(dai.node.StereoDepth)
        cam_depth.setConfidenceThreshold(200)
        cam_depth.setDepthAlign(dai.CameraBoardSocket.RGB)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        cam_depth.depth.link(xout_depth.input)

        # Connect to the device and start the pipeline
        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            while True:
                in_rgb = q_rgb.get()
                in_depth = q_depth.get()
                
                frame = in_rgb.getCvFrame()
                depth_frame = in_depth.getFrame()

                # Perform object detection on RGB frame
                input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
                input_tensor = input_tensor[tf.newaxis, ...]

                detections = model(input_tensor)
                print(f"Detections: {detections}")  # Debugging: print detection results

                # Check if detections contain boxes and handle them
                if 'detection_boxes' in detections:
                    for detection in detections['detection_boxes']:
                        ymin, xmin, ymax, xmax = detection

                        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1], 
                                                      ymin * frame.shape[0], ymax * frame.shape[0])

                        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)

                # Display frames
                cv2.imshow("RGB", frame)
                cv2.imshow("Depth", depth_frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        cv2.destroyAllWindows()
