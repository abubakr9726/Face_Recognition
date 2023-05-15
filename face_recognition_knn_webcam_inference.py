import face_recognition
import cv2
import pickle
import time
import os
import re
# from sklearn.neighbors import KNeighborsClassifier
# import math
# from PIL import Image, ImageDraw
# import numpy as np


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def main():

    cap = cv2.VideoCapture("https://192.168.1.5:8080/video")
    model_file = "model/knn_model_trained.clf"

    classifier = None
    if os.path.isfile(model_file):
        with open(model_file, 'rb') as f:
            classifier = pickle.load(f)
    else:
        raise Exception(f"Invalid classifier path: {model_file}. Check if the path is correct or pretained model is present.")

    while True:

        ret, frame = cap.read()
        size = frame.shape[0:2]
        if ret:
                prev_time = time.time()
                predictions = predict(frame, knn_clf=classifier)
                if predictions != []:
                    # Print results on the console
                    for name, (top, right, bottom, left) in predictions:
                        print(f"- Found {name} at ({left}, {top})")

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # Put framePerSecond on the ouput screen at which we are processing camera feed
                framePerSecond = 1 / (time.time() - prev_time)
                cv2.putText(frame, "{0:.2f}-framePerSecond".format(framePerSecond), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
                cv2.imshow("output", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()