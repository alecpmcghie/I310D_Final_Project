# Heart Disease Model
Content Contributers: Marcus Hanania, Alex Sanders, Max Diamond, Alec McGhie

## Brief introduction

The goal of this project is to use logistic regression to predict heart disease from our dataset on incoming patients into a cleveland hospital

## Dataset

Our dataset was collected from a

## Methodology

## Conclusion

# Tabel of contents and code breakdown

## Model Creation `Predicating_heart_disease.ipynb`

## Logistical Regression Model `logistic_regression_model.pkl`

## Model Demo `Heart_Disease_Prediction.py`
This model demo uses pythons TKinter and pickle librarys to create an interactive gui that runs the user data through `logistic_regression_model.pkl`. When using the model the doctor would enter in the values for the patent and then these values are run through the model. After the model concludes wether the data is positive or negitive the label at the bottom will be changed to reflect the output of the model. It is important to remember that our model is not a doctor and all of the assumptions from the model should not be used to diagnose heart disease.
To use this model save `Heart_Disease_Prediction.py` and `logistic_regression_model.pkl` into the same folder on your computer. Then open `Heart_Disease_Prediction.py` in your IDE of choice. At this point make sure that the file path for the model is correct in line 6. From here you can run the code from the IDE and the TKinter GUI should open. While using the application make sure that all of the entry boxes have valid data before running the test.
## Model Demo Video `Heart Disease Model Demo.mp4`
This file contains two patients from the dataset one with a postive diagnosis of heart disease and one with a negitive diagnosis for heart disease.
