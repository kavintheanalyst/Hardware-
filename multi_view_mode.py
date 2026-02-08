import depthai as dai
import cv2
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

xout_rgb = pipeline.create(dai.node.XLinkOut) 
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

cam_mono_left = pipeline.create(dai.node.MonoCamera)
cam_mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
cam_mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

cam_mono_right = pipeline.create(dai.node.MonoCamera)
cam_mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
cam_mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.initialConfig.setConfidenceThreshold(200)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

cam_mono_left.out.link(stereo.left)
cam_mono_right.out.link(stereo.right)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the frames from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    view_mode = 'rgb'  # Initial view mode

    while True:
        if view_mode == 'rgb':
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif view_mode == 'gray':
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif view_mode == 'depth':
            in_depth = q_depth.get()
            frame = in_depth.getFrame()
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        
        # Break the loop on 'q' key press and switch view modes
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            view_mode = 'rgb'
        elif key == ord('g'):
            view_mode = 'gray'
        elif key == ord('d'):
            view_mode = 'depth'

    cv2.destroyAllWindows()

