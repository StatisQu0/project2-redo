# Project2-Redo
       
# *Project 2 - redo - README.md File*
---
Group 1 - Project 2 has attempted to take two sets of data and create a model that predicts a buy or sell signal from relatively uncorrelated data.  Jet Fuel prices (daily) and Spirit Airlines (SAVE) stock price.  One would think there might be a weak correlation and we find that without modification using the model, there is a fundamental correlation of 9.2%.  Not a high correlation for making stock trading decisions.  Nonetheless, we wanted to see if by utilizing supervised machine learning we could improve upon that correlation and assess trading signals by training the model on 3 years of data (over a 5 year period) and the test data on the remaining 2 year period.  

We utilized the price changes in each data set to feed into our model.  X was our Jet Fuel prices and y (our dependent variable) was the stock price.  

We did the following:

# Instructions

The appropriate Jupyter Notebook to open is the "Final Product.ipynb."  The README.md has also been updated.  


Use the raw code files to complete the steps that the instructions outline. 

1. Model

      A. We pulled data for Jet Fuel from the Fred database via API

      B. We pulled data for Spirit Airlines (SAVE) from Yahoo and loaded into the program via CSV.

      C. Preprocessed the data to get the dataset to only include the dates available for the shortest dataset, stock price.

      D. Set the index as the Date

      E. Got the datasets into a single DataFrame with Jet_Fuel as one column and Close (closing price of SAVE) as the other column.

      F. We got the percentage change for each column in our DataFrame.

            i. Simply by using pct_change(), the correlation increased to greater than 15%. 

      G. We tested if the Actual Returns (of the stock) were greater or less than Zero.  

            i. Above Zero - 1
            ii. Less than Zero - -1
            
      H. We set the training and testing datasets and ran them through our model utilizing Logistic Regression.  
      

2. Fit

3. Transform

4. Predict

5. Evaluate

      A. Our Classification Report yielded the following results:

                  precision    recall  f1-score   support

            -1.0       0.49      0.43      0.46       229
             1.0       0.55      0.62      0.59       263

       accuracy                            0.53       492
       macro avg       0.52      0.52      0.52       492
       weighted avg    0.53      0.53      0.53       492



## Technologies
This program was built entirely in tandem with the prepared questions in a Jupyter Notebook using Python and the associated libraries noted above.  It was also built using Windows 10 on a Dell Laptop PC.  


---

## Installation Guide

### *This is how the libraries are imported into the program.  These import statements reside at the top of the code and are executed first.*

Written in python and utiizing the following libraries:

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


pandas: https://pandas.pydata.org/

numpy: https://numpy.org/

requests: https://pypi.org/project/requests/

json: https://docs.python.org/3/library/json.html

Path (from pathlib): https://docs.python.org/3/library/pathlib.html

matplotlib.pyplot: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html

seaborn: https://seaborn.pydata.org/

DateOffset from pandas.tseries.offsets: https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html

StandardScaler from sklearn.preprocessing: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

svm: http://scikit-learn.org/stable/modules/svm.html

sklearn: https://scikit-learn.org/

LogisticRegression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

classification_report from sklearn.metrics: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html



## Contributors

All of the work was performed by Qristian Walker, Richard Kell and Christopher Todd Garner

---

## License

You may use this code as you see fit as long as any copy and paste is done so with proper sourcing of materials back to this repository.                                      
