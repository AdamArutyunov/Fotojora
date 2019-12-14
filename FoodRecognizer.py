import argparse
import pickle
import cv2
from keras.models import load_model


class FoodRecognizer:
    def __init__(self, model, label_bin, size, flatten):
        self.label_bin = label_bin
        self.size = size
        self.width, self.height = self.size
        self.flatten = flatten
        
        print("[INFO] loading network and label binarizer...")
        self.model = load_model(model)
        self.lb = pickle.loads(open(self.label_bin, "rb").read())

    def load_image(self, image_file):
        image = cv2.imread(image_file)
        output = image.copy()
        image = cv2.resize(image, self.size)
        image = image.astype("float") / 255.0

        if self.flatten > 0:
            image = image.flatten()
            image = image.reshape((1, image.shape[0]))

        else:
            image = image.reshape((1, image.shape[0], image.shape[1],
                                   image.shape[2]))

        self.image = image

    def recognize(self):
        preds = self.model.predict(self.image)
        print(preds)
        i = preds.argmax(axis=1)[0]
        label = self.lb.classes_[i]

        return label

if __name__ == "__main__":
    model_path = "output/test_model.model"
    label_path = "output/test_model.pickle"

    size = (32, 32)
    flatten = 1

    MFR = FoodRecognizer(model_path, label_path, size, flatten)
    MFR.load_image("images/kolbasa.jpg")
    print(MFR.recognize())
            
 
            
