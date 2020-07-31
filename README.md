# Sudoku-Solver-with-Deep-Learning
This project uses OpenCV for detecting numbers from the uploaded image and Deep Learning solving Sudoku Puzzle.

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

If you are using the Streamlit UI code, you should update the path of the Tesseract OCR in the code and run the main.py using 'streamlit run main.py' command after making the Deep Learning model file from 'Sudoku_Solver.ipynb'. Place the saved model in strealit folder.

Note:
This project can be implemented using two ways. 
1. You can train a custom Deep Learning Model for character recognition. The Digits_Recognition.ipynb file is one such model. But the recognition speed is low using this process. 
2. Using Tesseract OCR.

# Dataset
https://www.kaggle.com/bryanpark/sudoku  - For training Sudoku model

https://www.kaggle.com/c/digit-recognizer/data  -  For number detection using Deep Learning.

# Results

## Given Input:

![result1](https://user-images.githubusercontent.com/50202237/89007751-fd4b2600-d326-11ea-8665-062bff232705.jpg)


## Predicted Output:

![result2](https://user-images.githubusercontent.com/50202237/89007756-ff14e980-d326-11ea-92ea-ce7301be89e4.jpg)

# Credits
Inspired from shivaverma and robovirmani.

