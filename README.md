# UW-CSE163-Project
Final project for CSE163: Intermediate Data Programming in University of Washington

Authors: Xuqing Wu, Yuzhuo Ma, Zhuoyi Zhao

Proposal Link: https://docs.google.com/document/d/1X6JcnAoaaaaei1FJA52cRCE076oZflmOH0kIG1Mkfd0/edit?usp=sharing

Dataset Link: https://www.kaggle.com/datasets/thedevastator/insurance-claim-analysis-demographic-and-health

Final Report Link: https://docs.google.com/document/d/115dKF0ZKfSlQjxm6yZ5EVV5fNgKNFIqVSfaBCTN41NM/edit?usp=sharing

Final PowerPoint Link: https://docs.google.com/presentation/d/1txZ4AZmEMBhEEfXJ-2tS3Y-Nr83N7iTB56U5Ek4hoOU/edit?usp=sharing

## Getting Started
### Installation Requirements
To run these code, you will need Python 3.x installed on your computer(you can download it from the official website: https://www.python.org/downloads/), as well as the following modules:
* pandas
* sklearn

To get started, follow these steps:
1. Clone the repository to your local machine.
2. Install the required Python modules by running the following command in your terminal: `pip install pandas sklearn`


**Q1 code instruction:** To reproduce the result we get in Q1, all you need to do is to upload the dataset and run the predict_insurance_claim.py. There are detailed comments as instructions on what each part of the code is doing in the file.

**Q2 code instruction:** To reproduce the result we get in Q2, you don’t need to download the dataset since the dataset "insurance_data.csv” is in the same directory as the each file. There are detailed comments as instructions on what each part of the code is doing in the file. To run the code following `bar plot.py`, `pie chat.py`, and `scatter plot.py`, you will see the each plot from each file. You can choose the 'male' button or 'female' button in the bar to see the different plot.

**Q3 code instruction:** To reproduce the result we get in Q3, you don’t need to download the dataset since the dataset "insurance_data.csv”is in the same directory as the "diabetic.py" file. To run the code, navigate to the "diabetic.py" file in your terminal and run the following command: `python diabetic.py` The code will read in the data from the "insurance_data.csv" file, clean it, build a machine learning model, and output the accuracy score of the model. Additionally, it will run hyperparameter tuning to find the best combination of hyperparameters for the model and output the accuracy scores for each combination in descending order.

**_Please note that:_** You may encounter the error message reporting a deprecated error by running this code. It is an internal issue inside the sklearn library, which basically means that sklearn is calling some numpy library functions that are deprecated. You could try to update the sklearn library to the latest version (e.g. via the pip command) and see if the developers have fixed this, or config Python to suppress these warnings at runtime. However, these warnings don't break the code or cause other errors, so you can ignore the deprecated error message.

**test_file:**
