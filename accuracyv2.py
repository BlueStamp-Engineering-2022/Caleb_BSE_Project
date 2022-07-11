from __future__ import print_function
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
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import zoom
from matplotlib.patches import Rectangle



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



results = json.load(open('results.xml'))


trainer.results = results




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


input_face = cv2.imread('/Users/josephyu/Downloads/IMG_1684.jpg')
cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
detected_faces = faceCascade.detectMultiScale(gray, 1.1, 4)
print(detected_faces)
for (x,y,w,h) in detected_faces:
    if w > 0:
        horizontal_offset = round(0.15 * w)
        vertical_offset = round(0.2 * h)
        print(horizontal_offset)
        print(vertical_offset)
        print(gray.shape[0])
        print(gray.shape[1])
        roi = gray[y:y+h, x:x+w]
        print(roi.shape[0])
        print(roi.shape[1])
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x+w-horizontal_offset]
        print(extracted_face.shape[0])
        print(extracted_face.shape[1])
        new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 64. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(float32)
        new_extracted_face /= float(new_extracted_face.max())
        display_face(new_extracted_face[:, :])
        #new_extracted_face = new_extracted_face.ravel().reshape(1, -1)

        print(svc_1.predict(new_extracted_face.ravel().reshape(1, -1)))
