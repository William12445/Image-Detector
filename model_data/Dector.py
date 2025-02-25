﻿import cv2
import numpy as np
import time

np.random.seed(20)

class Dector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, "r", encoding="UTF-8") as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0, "__Background__")
        
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        seeSet = set()
      
        if not cap.isOpened():
            print("Error opening file")
            return
        (success, image) = cap.read()
        
        startTime = 0
        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.4)
            bboxs = list(bboxs)
            
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIdx) != 0:
                for i in range(len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    
                    displayText = "{}: {:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, classColor, 2)
                    lineWidth = min(int(w * 0.3), int(h * 0.3))

                    cv2.line(image, (x, y), (x + lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y + lineWidth), classColor, thickness=5)
                    cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, thickness=5)
                    cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, thickness=5)
                    cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x, y + h), (x, y + h - lineWidth), classColor, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w - lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness=5)
            
            cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            
            # Show image in a resizable window
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", image)
            
            temp = str(classLabelID) + str(classLabel)
            seeSet.add(temp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Detected Objects:\n", str(seeSet))
                break
            
            (success, image) = cap.read()
        
        cap.release()
        cv2.destroyAllWindows()

