# Sudoku-Solver-with-Deep-Learning
A simple Sudoku solver using Deep Learning and OpenCV

The source code is in main.py in Streamlit folder.

# Dependencies
The following dependencies are required:

numpy
opencv
pygame
pytesseract
keras
tensorflow
pandas
numpy
sklearn

If you are using the Streamlit UI code, you should update the path of the Tesseract OCR in the code and run the main.py using 'streamlit run main.py' command after making the Deep Learning model file from 'Sudoku_Solver.ipynb'

Note:
This project can be implemented using two ways. 
1. You can train a custom Deep Learning Model for character recognition. The Digits_Recognition.ipynb file is one such model. But the recognition speed is low using this process. 
2. Using Tesseract OCR.

# Dataset
https://www.kaggle.com/bryanpark/sudoku  - For training Sudoku model

https://www.kaggle.com/c/digit-recognizer/data  -  For number detection using Deep Learning.

# Credits
Inspired from shivaverma and robovirmani.
