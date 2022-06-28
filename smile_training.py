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


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

        #-- In each face, detect eyes


    return frame_gray, faces


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face

    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    extracted_face = gray#[y+int(vertical_offset):y-int(vertical_offset+h), x+int(horizontal_offset):x-int(horizontal_offset+w)]

    new_extracted_face = zoom(extracted_face, (4096. / extracted_face.shape[0], 4096. / extracted_face.shape[1]))


    new_extracted_face = new_extracted_face.astype(float32)
    #dimming image

    new_extracted_face /= float(new_extracted_face.max())

    return new_extracted_face

def predict_face_is_smiling(extracted_face):
    print(svc_1.predict(extracted_face))
    print(svc_1.predict(extracted_face).sum())
    return svc_1.predict(extracted_face).sum()/svc_1.predict(extracted_face).shape

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default="/Users/josephyu/Desktop/BlueStamp/haarcascade_frontalface_alt.xml")
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)



camera_device = args.camera
#-- 2. Read the video stream
cap = cv2.VideoCapture(camera_device)

while True:
    # Capture frame-by-frame
    time.sleep(1)
    ret, frame = cap.read()

    # detect faces
    gray, detected_faces = detectAndDisplay(frame)

    face_index = 0

    for (x,y,w,h) in detected_faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = gray[y:y+h,x:x+w]
    # predict output

    for (x,y,w,h) in detected_faces:

        if w > 100:



            extracted_face = extract_face_features(gray, (x,y,w,h),  (0.075, 0.05))#(0.03, 0.05)
            # predict smile

            prediction_result = predict_face_is_smiling(extracted_face)            # draw extracted face in the top right corner
            #frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # annotate main image with a label
            label = "Smiling" if prediction_result >= 0.5 else "Not Smiling"

            cv2.putText(frame, label,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

#            # increment counter
            face_index += 1


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cv2.destroyAllWindows()
