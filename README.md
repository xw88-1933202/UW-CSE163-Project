# UW-CSE163-Project
Final project for CSE163: Intermediate Data Programming in University of Washington

Authors: Xuqing Wu, Yuzhuo Ma, Zhuoyi Zhao

Proposal Link: https://docs.google.com/document/d/1X6JcnAoaaaaei1FJA52cRCE076oZflmOH0kIG1Mkfd0/edit?usp=sharing

Dataset Link: https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health

## Getting Started
### Installation Requirements
To run these code, you will need Python 3.x installed on your computer, as well as the following modules:
* pandas
* sklearn

To get started, follow these steps:
1. Clone the repository to your local machine.
2. Install the required Python modules by running the following command in your terminal: `pip install pandas sklearn`


Q1 code instruction: To reproduce the result we get in Q1, all you need to do is to upload the dataset and run the predict_insurance_claim.py. There are detailed comments as instructions on what each part of the code is doing in the file. After running, in the console you will see some warnings, such as "deprecated in NumPy 1.20". These are just problems with the packages, and they have no effect on our results, just ignore them.

Q2 code instruction: 

Q3 code instruction: To reproduce the result we get in Q3, you don’t need to download the dataset since the dataset "insurance_data.csv”is in the same directory as the "diabetic.py" file. To run the code, navigate to the "diabetic.py" file in your terminal and run the following command: `python diabetic.py` The code will read in the data from the "insurance_data.csv" file, clean it, build a machine learning model, and output the accuracy score of the model. Additionally, it will run hyperparameter tuning to find the best combination of hyperparameters for the model and output the accuracy scores for each combination in descending order.

**_Please note that:_** you may encounter the error message reporting deprecate error by running this code. It is an internal issue inside the sklearn library (it was basically saying that sklearn was calling some numpy library functions that are deprecated) You could try to update the sklearn library to the latest version (e.g. via pip command) and see if the developers have fixed this, or config Python to suppress these warnings at runtime. However, thiese warnings don't break the code and cause other errors. So, you can ignore the depreate error message.
