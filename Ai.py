import cv2


img=cv2.imread('person.png')
names=[]
f1='files\\things.names'
with open(f1,'rt') as f:
    names=f.read().rstrip('\n').split('\n')
#print(names)
f2='files\\frozen_inference_graph.pb'
f3='files\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

n=cv2.dnn_DetectionModel(f2,f3)
n.setInputSize(400,400)
n.setInputScale(1.0/127.5)
n.setInputMean((127.5,127.5,127.5))
n.setInputSwapRB(True)#color
a1 ,a2 ,a3=n.detect(img,confThreshold=0.5)
#print(a1,a3)

for classIds , confidence,box in zip(a1.flatten(),a2.flatten(),a3):
    cv2.rectangle(img,box,color=(0,255,0),thickness=3)
    cv2.putText(img,names[classIds-1],
                (box[0]+10,box[1]+20),
                cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),thickness=3)

cv2.imshow('Ai',img)
cv2.waitKey(0)