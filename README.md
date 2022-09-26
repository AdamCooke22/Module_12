# MODULE 12 CHALLENGE : Credit Risk Resampling

For this challenge we are to use various techniques to train and evaluate models with imbalanced classes. We start this assignment by using historical lending activity to build a model that can identify the creditworthiness of borrowers. We use a logistic regression model to compare two versions of the dataset, the original, and a oversampled version of the dataset. For both versions of the dataset, we count the target cases, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.




---

## Technologies

This project leverages python 3.7 with the following packages:

* [Pandas](https://github.com/google/pandas) - Pandas is a powerfull tool for data analysis and manipulation. Pandas provides a plethora of useful functions that make it easy to express, analyze, and manipulate data.


* [scikit-learn](https://scikit-learn.org/stable/) - This is a machine learning library for the python programming language. This library allows for the use of multiple machine learning models, tools, and algorithms.



---

## Installation Guide

Before running the application first install the following dependencies.

```
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')

```

---

## Usage

To use the credit risk resampling file simply clone the repository and open the credit_risk_resampling.ipynb file in jupyter notebook.

Upon opening the file you will have the option to run the whole note book and that will provide you with all of the calculations, evaluations, and visualizations for the analysis of the clustering data. Some screenshots of that in action can be seen below via this link below.

* [SCREENSHOTS](https://github.com/AdamCooke22/module_12/tree/main/screenshots) 

## Contributors

Completed by Adam Cooke

---

## License

MIT
