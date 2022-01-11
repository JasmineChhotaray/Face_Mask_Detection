import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tensorflow import keras
#from train_mask_detection_model import store_image_paths

class MaskDetectionImage:

    def __init__(self, arg):
        self.image = arg['image']
        self.model_path = Path(arg['model'])
        self.face_path = r"E:\my_work\GitHub_Projects\Face_Mask_Detection\face_model"
        self.conf_threshold = 0.90

    def run(self):
        prototxtPath = os.path.sep.join([self.face_path, "deploy.prototxt"])
        weightsPath = os.path.sep.join([self.face_path, "res10_300x300_ssd_iter_140000.caffemodel"])
        network = cv2.dnn.readNet(prototxtPath, weightsPath)
        #test_image_paths = self.store_image_paths(self.image_path)
        model = keras.models.load_model(os.path.join(self.model_path, "trained_mask_model_1.h5"))

        # image = keras.preprocessing.image.load_img(test_image_paths[0], target_size=(224, 224))
        labels = ["Face with Mask", "Face without Mask"]
        colors = [(0, 255, 0), (0, 0, 255)]

        image = cv2.imread(self.image)
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104.0, 177.0, 123.0])
        network.setInput(blob)
        detections = network.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                # bounding box : (x1, y1, x2, y2)
                bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = bbox.astype("int")

                # ensuring that bounding box falls within the image
                (x1, y1) = (max(0, x1), max(0, y1))
                (x2, y2) = (min(w-1, x2), min(h-1, y2))

                face = image[y1:y2, x1:x2]

                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue

                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))

                face_array = keras.preprocessing.image.img_to_array(face)
                face_batch =np.expand_dims(face_array, axis=0)
                face_processed = keras.applications.mobilenet_v2.preprocess_input(face_batch) 

                prediction = model.predict(face_processed)[0]
                label_idx = np.argmax(prediction)

                label = labels[label_idx]
                color = colors[label_idx]

                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            else:
                break

        cv2.imshow("Output", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="path to input image")
    parser.add_argument("-m", "--model", type=str, help="path to trained mask detector model")

    args = parser.parse_args()

    obj = MaskDetectionImage(vars(args))
    obj.run()
