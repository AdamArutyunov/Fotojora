import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.layers import GaussianNoise
from keras.optimizers import SGD
from imutils import paths


matplotlib.use("Agg")

dataset = "ingredients"
model_path = "output/test_model.model"
label_bin = "output/test_model.pickle"
plot = "output/test_model_plot.png"

print("[INFO] loading images...")
data = []
labels = []
 
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)
 
for imagePath in imagePaths:
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    except Exception as e:
        print("[WARNING]", e)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2,
                                                  random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Sequential()
model.add(GaussianNoise(0.01))
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(256, activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(len(lb.classes_), activation="softmax"))

INIT_LR = 0.01
EPOCHS = 75

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=EPOCHS, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))
 
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
print(H.history.keys())
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot)

print("[INFO] serializing network and label binarizer...")
model.save(model_path)
f = open(label_bin, "wb")
f.write(pickle.dumps(lb))
f.close() 
