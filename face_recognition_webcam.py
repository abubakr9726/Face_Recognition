import face_recognition
import cv2
import time
import PIL

def load_sample_img(img):

    known_img = face_recognition.load_image_file(img)
    face_locations_known = face_recognition.face_locations(known_img)
    known_encoding = face_recognition.face_encodings(known_img, known_face_locations=face_locations_known)[0]
    # print(known_encoding)
    known_img = cv2.cvtColor(known_img, cv2.COLOR_RGB2BGR)

    return known_img, face_locations_known, known_encoding 


def recognition(source, known_encoding):

    cap = cv2.VideoCapture(source)

    while True:
        ret, unknown_face = cap.read()
        size = unknown_face.shape[0:2]

        if ret:
            prev_time = time.time()
            face_locations_unknown = face_recognition.face_locations(unknown_face)
            # print(face_locations_unknown)
            if face_locations_unknown != []:
                # print(face_locations_unknown)
                for (top, right, bottom, left) in face_locations_unknown:
                    cv2.rectangle(unknown_face, (left, top), (right, bottom), (0, 255, 0), 2)
                unknown_encoding = face_recognition.face_encodings(unknown_face, known_face_locations=face_locations_unknown)[0]
                # print(len(unknown_encoding))
                results = face_recognition.compare_faces([known_encoding], unknown_encoding)

                if results[0]:
                    cv2.putText(unknown_face, "Face_matched", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    face_locations_unknown = []
                    unknown_encoding = []
                    # print("face matched")

                else:
                    cv2.putText(unknown_face, "Unknown", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    face_locations_unknown = []
                    unknown_encoding = []
                    # print("face not matched")

            else:
                face_locations_unknown = []
                unknown_encoding = []

            framePerSecond = 1 / (time.time() - prev_time)
            cv2.putText(unknown_face, "{0:.2f}-framePerSecond".format(framePerSecond), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 4)
                
            cv2.imshow("face", unknown_face)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()


def main():
    known_img, face_locations_known, known_encoding = load_sample_img("dataset/train/abubakr/abubakr3.jpg")
    known_img = cv2.cvtColor(known_img, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(known_img)
    pil_image.show()
    
    recognition(source="https://192.168.1.5:8080/video", known_encoding=known_encoding)
    

if __name__ == "__main__":
    main()


