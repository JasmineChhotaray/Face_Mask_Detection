import os
import cv2
import sys
import logging
import argparse
import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from utils import write_image, key_action, init_cam



class VideoDetection:

    def __init__(self):
        self.model_path = r"E:\my_work\Spiced_Projects\final_project"
        self.face_path = r"E:\my_work\Spiced_Projects\final_project\face_model"
        self.conf_threshold = 0.90

    # Normal Image detection and providing Top 5 probability
    def run_normal_predictions(self):

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        logging.getLogger().setLevel(logging.INFO)

        # also try out this resolution: 640 x 360
        #webcam = init_cam(640, 480)
        labels = ["Face with Mask", "Face without Mask"]
        colors = [(0, 255, 0), (0, 0, 255)]

        key = None

        prototxtPath = os.path.sep.join([self.face_path, "deploy.prototxt"])
        weightsPath = os.path.sep.join(
            [self.face_path, "res10_300x300_ssd_iter_140000.caffemodel"])
        network = cv2.dnn.readNet(prototxtPath, weightsPath)
        model = keras.models.load_model(os.path.join(
            self.model_path, 'trained_mask_model.h5'))

        capture = cv2.VideoCapture(0)

        while capture.isOpened():
            # get key event
            key = key_action()
            
            flags, image = capture.read()
            #image = self.init_webcam(webcam)

            (h, w) = image.shape[:2]

            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104.0, 177.0, 123.0])
            network.setInput(blob)
            detections = network.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.conf_threshold:
                    bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = bbox.astype("int")

                    (x1, y1) = (max(0, x1), max(0, y1))
                    (x2, y2) = (min(w-1, x2), min(h-1, y2))

                    face = image[y1:y2, x1:x2]

                    if face.shape[0] == 0 or face.shape[1] == 0:
                        continue

                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))

                    face_array = keras.preprocessing.image.img_to_array(face)
                    #face_batch = np.expand_dims(face_array, axis=0)
                    face_processed = keras.applications.mobilenet_v2.preprocess_input(
                        face_array)
                    face_processed = np.expand_dims(face_processed, axis=0)

                    prediction = model.predict(face_processed)[0]
                    label_idx = np.argmax(prediction)

                    label = labels[label_idx]
                    color = colors[label_idx]

                    cv2.putText(image, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                else:
                    break

            cv2.imshow("Output", image)

            if key == 'q':
                break
        
        capture.release()
        cv2.destroyAllWindows()

    def init_webcam(self, webcam):

        # Capture frame-by-frame
        ret, frame = webcam.read()

        # fliping the image
        frame = cv2.flip(frame, 1)

        # draw a [224x224] rectangle into the frame, leave some space for the black border
        offset = 2
        width = 224
        x = 150
        y = 150

        # cv2.rectangle(img=frame,
        #               pt1=(x-offset, y-offset),
        #               pt2=(x+width+offset, y+width+offset),
        #               color=(0, 0, 0),
        #               thickness=2
        #               )

        # write the image without overlay
        # extract the [224x224] rectangle out of it
        image = frame[y:y+width, x:x+width, :]

        return image

    def capture_image(self, image):
        out_folder = sys.argv[1]
        # print(image)
        write_image(out_folder, image)

    def quit_webcam(self, webcam):
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = VideoDetection()
    obj.run_normal_predictions()
