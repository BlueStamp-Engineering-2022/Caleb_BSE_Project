import cv2
import json
import xml.etree.ElementTree as xml
import matplotlib
from pylab import *
from tkinter import *
from sklearn import datasets
import numpy
from IPython.display import display, clear_output
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
import argparse

#root = Tk()


svc_1 = SVC(kernel='linear')
faces = datasets.fetch_olivetti_faces()
print(faces.keys())
print(faces.data)





#root = xml.Element("")
#doc = xml.SubElement(root, "")

#xml.SubElement(doc, "field1", name="").text = ""
#xml.SubElement(doc, "field2", name="").text = ""

#tree = xml.ElementTree(root)
#tree.write("results.xml")


#for i in range(10):
#    face = faces.images[i]
#    subplot(1, 10, i+1)
#    plt.imshow(face.reshape((64, 64)), cmap = 'gray')
#    axis('off')
#plt.show()


class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0
    def increment_face(self):
        if(self.index + 1 >= len(self.imgs)):
            return self.index
        else:
            while(str(self.index) in self.results):
                print(self.index)
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        self.results[str(self.index)] = smile

trainer = Trainer()





def display_face(face):
    clear_output()
    plt.imshow(face, cmap='gray')
    axis('off')
    plt.show()

def update_smile():
    trainer.record_result(smile=True)
    trainer.increment_face()
    display_face(trainer.imgs[trainer.index])

def update_no_smile():
    trainer.record_result(smile=False)
    trainer.increment_face()
    display_face(trainer.imgs[trainer.index])








#button_smile = Button(root, text='smile', command = update_smile)
#button_no_smile = Button(root, text='sad face', command = update_no_smile)



#button_smile = Button(root, text="smile", command = update_smile)
#button_no_smile = Button(root, text="sad face", command = update_no_smile)

#button_smile.pack()
#button_no_smile.pack()




#display_face(trainer.imgs[trainer.index])

    #button_smile.pack()
    #button_no_smile.pack()

#root.mainloop()

results = json.load(open('results.xml'))


trainer.results = results



yes, no = (sum([trainer.results[x] == True for x in trainer.results]), sum([trainer.results[x] == False for x in trainer.results]))
bar([0, 1], [no, yes])
ylim(0, max(yes, no))
xticks([0.4, 1.4], ['no smile', 'smile']);


#smiling_indices = [int(i) for i in trainer.results if trainer.results[i] == True]

#fig = plt.figure(figsize=(12, 12))
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#for i in range(len(smiling_indices)):
    #plot the images in a matrix of 20x20
#    p = fig.add_subplot(20, 20, i + 1)
#    p.imshow(faces.images[smiling_indices[i]], cmap=plt.cm.bone)

   # label the image with the target value
#    p.text(0, 14, "smiling")
#    p.text(0, 60, str(i))
#    plt.show()

#not_smiling_indices = [int(i) for i in trainer.results if trainer.results[i] == False]

#dig = plt.figure(figsize=(12, 12))
#dig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#for i in range(len(smiling_indices)):
    # plot the images in a matrix of 20x20
#    p = dig.add_subplot(20, 20, i + 1)
#    p.imshow(faces.images[smiling_indices[i]], cmap=plt.cm.bone)

    # label the image with the target value
#    p.text(0, 14, "smiling")
#    p.text(0, 60, str(i))
#    p.axis('off')
#    plt.show()

indices = [int(i) for i in trainer.results]

data = faces.data[indices, :]

target = [trainer.results[i] for i in trainer.results]
target = array(target).astype(int32)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), random_state=0, shuffle=True)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

evaluate_cross_validation(svc_1, X_train, y_train, 5)


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)





def detect_face(frame):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gray, detected_faces

def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    extracted_face = gray[y+int(vertical_offset):y+h,x+int(horizontal_offset):x-int(horizontal_offset+w)]
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 64. / extracted_face.shape[0]))
    new_extracted_face = new_extracted_face.astype(float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face

def predict_face_is_smiling(extracted_face):
    svc_1.reshape(-1, 1)
    return svc_1.predict(extracted_face.ravel())

def test_recognition(c1, c2):
    subplot(121)
    extracted_face1 = extract_face_features(gray1, face1[0], (c1, c2))
    imshow(extracted_face1, cmap='gray')
    print(predict_face_is_smiling(extracted_face1))
    subplot(122)
    extracted_face2 = extract_face_features(gray2, face2[0], (c1, c2))
    imshow(extracted_face2, cmap='gray')
    print(predict_face_is_smiling(extracted_face2))


def make_map(facefile):
    c1_range = linspace(0, 0.35)
    c2_range = linspace(0, 0.3)
    result_matrix = nan * zeros_like(c1_range * c2_range[:, newaxis])
    gray, detected_faces = detect_face(cv2.imread(facefile))
    for face in detected_faces[:1]:
        for ind1, c1 in enumerate(c1_range):
            for ind2, c2 in enumerate(c2_range):
                extracted_face = extract_face_features(gray, face, (c1, c2))
                result_matrix[ind1, ind2] = predict_face_is_smiling(extracted_face)
    return (c1_range, c2_range, result_matrix)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()


    # detect faces
    gray, detected_faces = detect_face(frame)

    face_index = 0

    # predict output
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            # draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # extract features
            extracted_face = extract_face_features(gray, face, (0.03, 0.05)) #(0.075, 0.05)

            # predict smile
            prediction_result = predict_face_is_smiling(extracted_face)

            # draw extracted face in the top right corner
            frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # annotate main image with a label
            if prediction_result == 1:
                cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
            else:
                cv2.putText(frame, "not smiling",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

            # increment counter
            face_index += 1


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
