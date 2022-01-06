import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 900, 900)
names = []

with open('coco.names', 'r') as f:
    for line in f:
        names.append(line)

# cap = cv2.VideoCapture(0)
filename = 'C:\\Users\\arkan\\Downloads\\WhatsApp Image 2021-12-11 at 4.19.21 PM.jpeg'
img = cv2.imread(filename)
# while True:
# _ , img = cap.read()
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True,
                             crop=False)

net.setInput(blob)
output_layer_names = net.getUnconnectedOutLayersNames()
layeroutputs = net.forward(output_layer_names)

boxes = []
confidences = []
name_ids = []
count = 0
objects = []

for o in layeroutputs:
    for i in o:
        s = i[5:]
        name_id = np.argmax(s)
        confidence = s[name_id]
        if confidence > 0.7:
            x_cent = int(i[0] * width)
            y_cent = int(i[1] * height)
            w = int(i[2] * width)
            h = int(i[3] * height)

            x = int(x_cent - w / 2)
            y = int(y_cent - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            name_ids.append(name_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(names[name_ids[i]])
    color = colors[i]
    conf = str(round(confidences[i], 2))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label + " " + conf, (x, y + 20), font,
                2, (0, 0, 0), 2)
    count += 1
    # objects.append(label)
cv2.imshow('Image', img)
print(count)
# key = cv2.waitKey(1)

# if key == 27 :
#     break

# cap.release()
# cv2.destroyAllWindows()