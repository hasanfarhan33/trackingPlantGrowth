import cv2 as cv 
import sys 
import xlsxwriter
from PIL import Image
import easyocr
import os
import pandas as pd 
from glob import glob
import time 
import numpy as np 
from pathlib import Path 

def debugging_shit(): 
    img_path = r"plant_images/cropped_images/UGAN01_1.jpg"

    img_date_time = get_date_taken(img_path).split(" ")
    print("DATE TIME ARRAY: ", img_date_time)
    print("DATE: ", img_date_time[0])
    print("TIME: ", img_date_time[-1])

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
    # print("Mask image shape:", mask.shape)
    print("Number of white pixels in the image:", cv.countNonZero(mask))
    print("TOTAL PIXELS: ", mask.shape[0] * mask.shape[1])

    # generate_excel_file(img_path)

    # print("Number of black pixels using np:", np.sum(mask == 0))

    cv.imshow("OG_Image", resized_img)
    cv.imshow("Green_Mask", mask)
    
    k = cv.waitKey(0)
    cv.destroyAllWindows()

    if k == ord("s"): 
        cv.imwrite("saved_image.png", img)


'''
LIMITATIONS: 
    Sizes of images are not exactly the same (White percentage value differs)
    Photos are not taken in a consistent lighting and height (Saturations and values of colors differ)
    Not every single green pixel is properly selected 
    Getting sample name is not 100% accurate 
'''

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

def getting_pixel_counts(img_path, show_images = False): 
    img = cv.imread(img_path)
    if img is None: 
        sys.exit("Could not read image")
    
    resized_img_og = ResizeWithAspectRatio(img, width = 500)
    resized_img = cv.resize(img, (500, 500), interpolation= cv.INTER_LINEAR)
    
    # Converting the image to HSV 
    hsv_img = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
    
    # Creating a mask for green colors 
    mask = cv.inRange(hsv_img, (40, 100, 20), (80, 255, 255))
    
    if show_images: 
        cv.imshow("Original Image", img)
        cv.imshow("Normal IMage", resized_img_og)
        cv.imshow("Resized Image", resized_img)
        cv.imshow("White Pixels", mask)
        
        k = cv.waitKey(0)
        cv.destroyAllWindows()
    
    white_pixel_count = cv.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    return white_pixel_count, total_pixels
    
    
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

  
def generate_excel_file(folder_path, file_name):
    # Actual dimensions of the paper in mm 
    paper_width_mm, paper_height_mm = 17, 18 
    paper_area_mm = paper_width_mm * paper_height_mm
    
    # Variables required to create the excel file 
    excel_file_name = file_name + ".xlsx"
    workbook = xlsxwriter.Workbook(excel_file_name)
    worksheet = workbook.add_worksheet()
    
    # Creating the headers
    header_data = ["FILE NAME", "SAMPLE NAME", "DATE", "TIME", "PAPER AREA (PX)", "SINGLE MM^2 PIXELS", "WHITE PIXEL COUNT", 
                   "TOTAL PIXELS", "WHITE PIXEL PERCENTAGE", "WHITE PIXELS MM"]
    header_format = workbook.add_format({'bold': True,
                                         'bottom': 2,})
    # Writing the headers of the excel file
    for col_num, data in enumerate(header_data): 
        worksheet.write(0, col_num, data, header_format)
        
    # Processing the image file 
    image_files = glob(os.path.join(folder_path, "*.jpg"))
    row = 1
    col = 0
    for cur_img_path in image_files: 
        row_data = [] 
        file_name = os.path.basename(cur_img_path)
        
        # Getting data from file 
        sample_name = get_sample_name(cur_img_path)
        img_date_time = get_date_taken(cur_img_path).split(" ")
        img_date = img_date_time[0]
        img_time = img_date_time[-1]
        
        # Getting paper area in pixels 
        paper_area_px = get_paper_area_px(cur_img_path)
        
        # How many pixels is 1 mm^2? 
        single_mm_px = round((paper_area_px / paper_area_mm), 2)
        
        # Getting data from image 
        white_pixels, total_pixels = getting_pixel_counts(cur_img_path, show_images = False)
        white_percent = (white_pixels / total_pixels) * 100 
        white_percent_rounded = round(white_percent, 2)
        
        white_pixels_area_mm = round((white_pixels / single_mm_px), 2)
        
        row_data.extend([file_name, sample_name, img_date, img_time, paper_area_px, single_mm_px, white_pixels, total_pixels, f"{white_percent_rounded}%", white_pixels_area_mm])
        
        worksheet.write_row(row, 0, tuple(row_data))
        row += 1
    
    worksheet.autofit()
    workbook.close()
    print("THE EXCEL FILE HAS BEEN GENERATED")


def get_paper_area_px(img_path, save_squares = False): 
    squares_folder = "extracted_squares"  
    img = cv.imread(img_path)
    if img is None: 
        sys.exit("No image found")
    
    resized_img = ResizeWithAspectRatio(img, 500)
    gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 9)
    sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpen = cv.filter2D(blur, -1, sharpen_kernel)
    
    thresh = cv.threshold(sharpen, 160, 255, cv.THRESH_OTSU)[1] 
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours and filter using threshold area 
    cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    min_area = 7_000 
    max_area = 9_000 
    
    for c in cnts: 
        area = cv.contourArea(c) 
        # print(area)
        if area > min_area and area < max_area: 
            x, y, w, h = cv.boundingRect(c)
            ROI = resized_img[y:y+h, x:x+w] 
        
            # Creating a folder and saving all the extracted rectangles there 
            if not os.path.exists(squares_folder):
                os.makedirs(squares_folder)
            else: 
                if save_squares: 
                    square_file_name = Path(img_path).stem
                    cv.imwrite(os.path.join(squares_folder, "{}_square.png".format(square_file_name)), ROI)
                else: 
                    break 
                    # Get the dimensions of the Detected Square 
                    # print("Area of the square has been extracted: ", area)
                    # return area       
                    # cv.imshow("Extracted Image with Area", ROI) 
            break 
    
    return area 
    

def get_extracted_square_area(folder_path, file_name = None, save_squares = False):
    image_files = glob(os.path.join(folder_path, "*.jpg"))
    squares_folder = "extracted_squares"
    for cur_img in image_files: 
        cur_file = os.path.basename(cur_img)
        # print(Path(cur_file).stem)
        
        # Load image, grayscale, median blur, sharpen
        loaded_img = cv.imread(os.path.join(folder_path, cur_file))
        resized_img = ResizeWithAspectRatio(loaded_img, 500)
        gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 9)
        sharpen_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        sharpen = cv.filter2D(blur, -1, sharpen_kernel)
        
        # Threshold, and morph close 
        thresh = cv.threshold(sharpen, 160, 255, cv.THRESH_OTSU)[1] 
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 1)
        
        # Find contours and filter using threshold area 
        cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]  
        
        min_area = 7_000 
        max_area = 9_000 
        
        # Getting the area of the square
        for c in cnts: 
            area = cv.contourArea(c)
            print(area)
            if area > min_area and area < max_area: 
                x, y, w, h = cv.boundingRect(c)
                ROI = resized_img[y:y+h, x:x+w] 
            
                # Creating a folder and saving all the extracted rectangles there 
                if not os.path.exists(squares_folder):
                    os.makedirs(squares_folder)
                else: 
                    if save_squares: 
                        square_file_name = Path(cur_file).stem
                        cv.imwrite(os.path.join(squares_folder, "{}_square.png".format(square_file_name)), ROI)
                    else: 
                        # Get the dimensions of the Detected Square 
                        print("Area of the square has been extracted: ", area)      
                        # cv.imshow("Extracted Image with Area", ROI) 
                break         
        # break 
        return area


if __name__ == "__main__":
    # Folder where all the plant images are 
    img_folder = r"plant_images/cropped_images"
    
    img_path = r"plant_images/cropped_images/UGAN01_1.jpg"
    
    # get_extracted_square_area(img_folder, save_squares=False)
    # get_paper_area_px(img_path=img_path)
    
    start_time = time.time()
    generate_excel_file(img_folder, "generate_file")
    print("TIME TAKEN TO GENERATE EXCEL FILE:", int(time.time() - start_time), "seconds")

    # Getting pixel counts 
    # white_pixel_count, total_pixels = getting_pixel_counts(img_path = r"plant_images/image.jpg", show_images = True)
    # print(white_pixel_count, total_pixels)

    # IGNORE THIS - IT IS A MULTILINE COMMENT! 
    '''
    # Detecting squares 
    # Load image, grayscale, median blur, sharpen image 
    # loaded_image = cv.imread(r"plant_images/cropped_images/UGAN01_1.jpg")
    # cv.imshow("NON RESIZED", loaded_image)
    # loaded_image = ResizeWithAspectRatio(loaded_image, width = 500)
    # cv.imshow("RESIZED", loaded_image)
    # gray = cv.cvtColor(loaded_image, cv.COLOR_BGR2GRAY)
    # blur = cv.medianBlur(gray, 9)
    # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1],[-1,-1,-1]])
    # sharpen = cv.filter2D(blur, -1, sharpen_kernel)

    # Getting pixel values 
    # for i in range(sharpen.shape[0]):
        # for j in range(sharpen.shape[1]): 
    #         print(sharpen[i][j])

    # Threshold and morph close 
    # thresh = cv.threshold(sharpen, 160, 255, cv.THRESH_OTSU)[1] 
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 1)

    # Find contours and filter using threshold area 
    # cnts = cv.findContours(close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]  

    # Printing the total area of the image 
    # print(loaded_image.size)

    # min_area = 5_000
    # max_area = 10_000  
    # image_number = 0

    # for c in cnts: 
        # area = cv.contourArea(c)
        # print(area)
        # if area > min_area and area < max_area: 
            # x, y, w, h = cv.boundingRect(c)
            # ROI = loaded_image[y:y+h, x:x+w]
            # cv.imwrite("ROI_{}.png".format(image_number), ROI)
            # cv.rectangle(loaded_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # image_number += 1
            # break 

    # cv.imshow('gray', gray)
    # cv.imshow('blur', blur)
    # cv.imshow('sharpen', sharpen)
    # cv.imshow('close', close)
    # cv.imshow('thresh', thresh)
    # cv.imshow('image', loaded_image)
    '''

    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows() 
        print("The program has been terminated!")