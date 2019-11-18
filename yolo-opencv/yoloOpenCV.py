import cv2
import numpy as np

# Load Yolo algorithm
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg') # deep nuralnetwork
classes = []  # object names dataset
with open('coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

# print(classes)
layer_names = net.getLayerNames()
output_Layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
# choose defferent colors
colors = np.random.uniform(0,255,size=(len(classes),3))

# print(outputLayers)
img = cv2.imread('testImg.png')

height, width, channels = img.shape
# detecting objects to give yolo
blob = cv2.dnn.blobFromImage(img,0.00392, (416,416), (0,0,0), True, crop=False)
# for b in blob:
#     for n, blob_img in enumerate(b):
#         cv2.imshow(str(n), blob_img)

# sending this blob to yolo alogorithm
net.setInput(blob)
outs = net.forward(output_Layers)
# print(outs)

# showing information on the screen
class_ids = []
boxes = []
confidences = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # object is detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # cv2.circle(img,(center_x, center_y),10, (0,255,0), 2)

            # Rectangle Coordinates
            x = int(center_x - w / 2)  # top left x
            y = int(center_y - h / 2)  # top left y
            # cv2.rectangle(img, (x,y), (x+w , y+h), (0,255,0), 2)  # draw rectangles around objects

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)# removing overlaping boxes
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = classes[class_ids[i]]
        color = colors[i]
        # print(label)
        cv2.rectangle(img,(x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x,y), font, 1, color, 1)


cv2.resize(img, None, fx=0.4, fy=0.4)
cv2.imshow("Yolo ", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
