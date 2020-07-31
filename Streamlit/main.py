# Required Imports
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import os
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
# Place your tesseract.exe path here
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Required functions

def rectify(h):
    '''
        This function takes in the image, and tries to rectify it. The four corners of the square 
        might be in any random order. So what I have done here is ordered the four corners in a 
        particular order. I have taken this order:
        Top Left and Right Then Bottom Left and Right
    '''
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def run(image_path):
    '''
        This function is used to find the boundries of the Sudoku Grid.
    '''

    img = cv2.imread(image_path)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img, biggest, -1, (0, 255, 0), 10)
    plt.imshow(img)

    return biggest, img


def numbers_detect(biggest, image_path):

    '''
        This function is used to detect the numbers in the provided Sudoku image using OCR Recognition. 
    '''
    biggest = rectify(biggest)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
    retval = cv2.getPerspectiveTransform(biggest, h)
    warp = cv2.warpPerspective(gray, retval, (450, 450))

    updated_img = warp
    updated_img = cv2.adaptiveThreshold(updated_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 10)

    # For numbers extraction using Deep Learning use the commented code in the end of this file.
    Game_board = [[0 for x in range(9)] for y in range(9)]
    for x in range(0, 9):
        for y in range(0, 9):

            morph = updated_img[((50 * x) + 3):((50 * x) + 47), ((50 * y) + 3):((50 * y) + 47)]

            text = pytesseract.image_to_string(morph, lang='eng', config='--psm 6 tessedit_char_whitelist=0123456789')

            if "1" in text:
                Game_board[x][y] = 1
                print(1, end=" ")
            elif "2" in text:
                Game_board[x][y] = 2
                print(2, end=" ")
            elif "3" in text:
                Game_board[x][y] = 3
                print(3, end=" ")
            elif "4" in text:
                Game_board[x][y] = 4
                print(4, end=" ")
            elif "5" in text:
                Game_board[x][y] = 5
                print(5, end=" ")
            elif "6" in text:
                Game_board[x][y] = 6
                print(6, end=" ")
            elif "7" in text:
                Game_board[x][y] = 7
                print(7, end=" ")
            elif "8" in text:
                Game_board[x][y] = 8
                print(8, end=" ")
            elif "9" in text:
                Game_board[x][y] = 9
                print(9, end=" ")
            else:
                Game_board[x][y] = 0
                print(0, end=" ")
        print()


    return Game_board


def norm(a):
    return (a / 9) - .5

def denorm(a):
    return (a + .5) * 9

def inference_sudoku(sample):
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''

    feat = sample

    model_test = load_model('solver.h5')

    while (1):

        out = model_test.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if (mask.sum() == 0):
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)

    return pred

def view_results(output_board):
    '''
        This function writes the solved Sudoku puzzle on a 9 x 9 grid to display this as image in UI
    '''
    image = Image.open('result.jpg')
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('Roboto-Light.ttf', size=35)

    # starting position of the message

    for i in range(0, 9):
        for j in range(0, 9):
            (x, y) = ((50 * i) + 55, (50 * j) + 30)
            name = str(output_board[i][j])
            #         name = '0'
            color = 'rgb(0, 0, 255)'  # white color
            draw.text((x, y), name, fill=color, font=font)
            image.save('trial.jpg')

            image = Image.open('trial.jpg')
            draw = ImageDraw.Draw(image)

    return 'trial.jpg'
    
    
def solve_sudoku(image):
    '''
        Main Solver Function
    '''

    biggest, img = run(image)
    Game_board = numbers_detect(biggest, image)
    input_game = ''
    for i in range(0, 9):
        for j in range(0, 9):
            input_game = input_game + str(Game_board[j][i])
            input_game = input_game + ' '
            
    input_game = input_game.replace('\n', '')
    input_game = input_game.replace(' ', '')
    input_game = np.array([int(j) for j in input_game]).reshape((9, 9, 1))
    input_game = norm(input_game)
    output = inference_sudoku(input_game)
    output_board = np.array(output).reshape(9, 9)
    print("Output: ")
    print(output_board)
    output_file_name = view_results(output_board)
    
    
    return output_file_name





### For Webpage UI using Streamlit

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

st.title("Sudoku Solver")
folder_path = '.'
filename = file_selector(folder_path=folder_path)
print(filename)
print(filename[2:])
st.image(filename[2:], caption='Uploaded Image.', use_column_width=True)
st.write("")
st.write("Solving...")
label = solve_sudoku(filename[2:])
st.image('trial.jpg', caption='Predicted Result.', use_column_width=True)




## Code for number detection using Deep Learning Model.

'''
biggest = rectify(biggest)
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)
retval = cv2.getPerspectiveTransform(biggest, h)
warp = cv2.warpPerspective(gray, retval, (450, 450))


updated_img = warp


updated_img = cv2.adaptiveThreshold(updated_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 10)


Game_board = [[0 for x in range(9)] for y in range(9)]

for x in range(0, 9):
    for y in range(0, 9):
        morph = updated_img[((50 * x) + 3):((50 * x) + 47), ((50 * y) + 3):((50 * y) + 47)]
        # Make temp/grid/ folder structure
        address = 'temp/grid/'+str(x)+str(y)+'.jpg'
        cv2.imwrite(address, morph)
#         print("morph.shape  :",morph.shape)
        model_test = load_model('cnn_mnist.h5')
        imvalue = imageprepare(address)
#         print("imvalue:  ",imvalue)
        imvalue = np.array(imvalue)
        imvalue = imvalue.reshape(1, 28, 28, 1)
#         print(imvalue.shape)
#         print("imvalue.shape:  ",imvalue.shape)
        digit = model_test.predict_classes(imvalue)
        text = str(digit[0])
        print(text)
        try:
            if "1" in text:
                Game_board[x][y] = 1
#                 print(1, end=" ")
            elif "2" in text:
                Game_board[x][y] = 2
#                 print(2, end=" ")
            elif "3" in text:
                Game_board[x][y] = 3
#                 print(3, end=" ")
            elif "4" in text:
                Game_board[x][y] = 4
#                 print(4, end=" ")
            elif "5" in text:
                Game_board[x][y] = 5
#                 print(5, end=" ")
            elif "6" in text:
                Game_board[x][y] = 6
#                 print(6, end=" ")
            elif "7" in text:
                Game_board[x][y] = 7
#                 print(7, end=" ")
            elif "8" in text:
                Game_board[x][y] = 8
#                 print(8, end=" ")
            elif "9" in text:
                Game_board[x][y] = 9
#                 print(9, end=" ")
        except:
            Game_board[x][y] = 0
#             print(0, end=" ")

print("\n", Game_board)
'''





