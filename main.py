import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt 

# Định nghĩa hàm imshow 
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

labelsPath = "YOLO/yolo/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size = (len(LABELS), 3))

print("Loading YOLO weights...")

weights_path = "YOLO/yolo/yolov3.weights" 
cfg_path = "YOLO/yolo/yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

print("Our YOLO Layers")
ln = net.getLayerNames()

print(len(ln), ln)

# Các ảnh nằm trong thư mục images
mypath = "YOLO/images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in file_names:
    # Đọc ảnh và lấy chiều dài và rộng
    image = cv2.imread(mypath + file)
    (H, W) = image.shape[:2]

    # Hàm trả về các Output Layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Đưa các images để predict
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB = True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    # Khởi tạo danh sách cho các box, class, 
    boxes = []
    confidences = []
    IDs = []

    # Lặp qua từng đầu ra của lớp
    for output in layerOutputs:
        # Lặp lại mỗi lần detection
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Giữ lại những dự đoán có thể nhất
            if confidence > 0.75:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Thêm tọa độ x,y vào bounding box
                boxes.append([x,y, int(width), int(height)])
                confidences.append(float(confidence))
                IDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw our bounding boxes and put our class label on the image
            color = [int(c) for c in COLORS[IDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
    imshow("YOLO Detections", image, size = 12)    















