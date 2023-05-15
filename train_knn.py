import face_recognition
# import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os
import math
# from PIL import Image, ImageDraw
import re
# import numpy as np
import pandas as pd

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    trainable_data = []
    classes = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                trainable_data.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                classes.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(trainable_data))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(trainable_data, classes)

    # Save the trained KNN classifier
    if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

    # Save the trained data points to a CSV file
    df = pd.DataFrame(trainable_data, classes)
    df.to_csv(os.path.join(model_save_path, "knn_data.csv"), index=True)

    if model_save_path is not None:
        model = os.path.join(model_save_path, "knn_model_trained.clf")
        with open(model, 'wb') as f:
            pickle.dump(knn_clf, f)
        
    return knn_clf


def main():
    print("Training KNN classifier...")
    classifier_model = train(train_dir="dataset/train/", model_save_path="model", n_neighbors=2)
    print("Training completed!!!!!!!!!!!!")

if __name__ == "__main__":
    main()