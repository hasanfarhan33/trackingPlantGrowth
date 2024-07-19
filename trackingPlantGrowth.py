import cv2 as cv 
import sys 
import numpy as np
import xlsxwriter
from PIL import Image
import easyocr


# Use this function to resize the image
def ResizeWithAspectRatio(image, width = None, height = None, inter=cv.INTER_AREA): 
    dim = None 
    (h, w) = image.shape[:2]
    
    if width is None and height is None: 
        return image 
    
    if width is None: 
        r = height / float(h)
        dim = (int(w * r), height)
    else: 
        r = width / float(w)
        dim = (width, int(h * r))
        
    return cv.resize(image, dim, interpolation=inter)

# Getting image metadata 
def get_date_taken(image_path): 
    exif = Image.open(image_path)._getexif() 
    if not exif: 
        raise Exception("Image {0} does not have EXIF data.".format(image_path))
    return exif[36867]

def get_sample_name(image_path):
    # Image Preprocessing 
    reader = easyocr.Reader(['en'])
    output_array = reader.readtext(image_path, detail=0)
    return output_array[-1]
    

        
img_path = "plant_images\\cropped_images\\UGAN01_1.jpg"

img_date_time = get_date_taken(img_path)

# Getting text from image 
sample_name = get_sample_name(img_path)
print("SAMPLE NAME:", sample_name)

img = cv.imread(cv.samples.findFile(img_path))

if img is None: 
    sys.exit("Could not read image.")

resized_img = ResizeWithAspectRatio(img, width = 500)



hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, (40, 100, 20), (80, 255, 255))

print("Hue, saturation, value image shape:", hsv.shape)
print("Mask image shape:", mask.shape)

# with np.printoptions(threshold=np.inf):
#     print(mask)

print("Number of white pixels in the image:", cv.countNonZero(mask))
# print("Number of black pixels using np:", np.sum(mask == 0))

cv.imshow("OG_Image", resized_img)
cv.imshow("Green_Mask", mask)


k = cv.waitKey(0)
cv.destroyAllWindows()

if k == ord("s"): 
    cv.imwrite("saved_image.png", img)