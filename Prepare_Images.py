from PIL import Image, ImageFilter
from imutils import paths
import random

dataset = "ingredients"
 
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(44)
random.shuffle(imagePaths)
 
for imagePath in imagePaths:
    try:
        image = Image.open(imagePath)
        image = image.filter(ImageFilter.BLUR)
        image = image.filter(ImageFilter.BLUR)
        formatt = imagePath.split(".")[-1]
        image.save(imagePath + "_blur." + formatt)
    except Exception as e:
        print("[WARNING]", e)
        
