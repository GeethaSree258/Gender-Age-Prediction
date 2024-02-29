# Additional Python Libraries Required :

OpenCV
pip install opencv-python

argparse
pip install argparse

# The contents of this Project :
opencv_face_detector.pbtxt, 
opencv_face_detector_uint8.pb, 
age_deploy.prototxt, 
age_net.caffemodel, 
gender_deploy.prototxt, 
gender_net.caffemodel,
   
# output:
Gender: Female
Age: 8-12 years
Gender: Female
Age: 25-32 years

# Code explanation: 

```python
import cv2
import math
import argparse
```
- These lines import necessary libraries: `cv2` for OpenCV, `math` for mathematical functions, and `argparse` for parsing command-line arguments.

```python
def highlightFace(net, frame, conf_threshold=0.7):
```
- This defines a function `highlightFace` that takes a neural network (`net`), a frame (image), and an optional confidence threshold (`conf_threshold` with a default value of 0.7).

```python
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
```
- It creates a copy of the input frame and stores its height and width.

```python
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
```
- It generates a blob (binary large object) from the frame using OpenCV's deep neural network (dnn) module.

```python
    net.setInput(blob)
    detections = net.forward()
```
- It sets the input of the neural network to the generated blob and performs a forward pass to obtain detections.

```python
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
```
- It iterates through the detected faces, checks if the confidence is above the threshold, and if so, extracts the face bounding box coordinates and draws a rectangle around the face on the copied frame.

```python
    return frameOpencvDnn, faceBoxes
```
- It returns the modified frame (`frameOpencvDnn`) and the list of face bounding boxes (`faceBoxes`).

```python
faceProto = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\opencv_face_detector.pbtxt"
faceModel = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\opencv_face_detector_uint8.pb"
ageProto = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\age_deploy.prototxt"
ageModel = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\age_net.caffemodel"
genderProto = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\gender_deploy.prototxt"
genderModel = r"C:\Users\M.Geethasree\OneDrive\Desktop\age & gender prediction\gender_net.caffemodel"
```
- These lines define file paths for face detection model files, age prediction model files, and gender prediction model files.

```python
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
```
- It defines mean values for model input normalization, age categories, and gender categories.

```python
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
```
- It reads the pre-trained models using the file paths specified earlier.

```python
video = cv2.VideoCapture(0)  # Change this to the path of your video file if needed
padding = 20
```
- It initializes video capture using OpenCV (`cv2.VideoCapture`). In this case, it uses the default camera (index 0). The variable `padding` is used to add extra space around the detected face.

```python
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
```
- It starts a loop that captures frames from the video stream. If there's no frame, it breaks out of the loop.

```python
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
```
- It calls the `highlightFace` function to get the modified frame and face bounding boxes. If there are no faces, it prints a message.

```python
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]
```
- It extracts the face region from the frame based on the bounding box coordinates and padding.

```python
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            print("Empty face region, skipping.")
            continue
```
- It checks if the face region is empty and skips further processing if true.

```python
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
```
- It generates a blob from the face region, sets it as input to the gender prediction model, obtains predictions, and extracts the gender. Similarly, it does the same for age prediction.

```python
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
```
- It overlays the predicted gender and age on the result image and displays it

.

```python
video.release()
cv2.destroyAllWindows()
```
- It releases the video capture object and closes all OpenCV windows.
