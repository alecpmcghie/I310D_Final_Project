# Heart Disease Model
Content Contributers: Marcus Hanania, Alex Sanders, Max Diamond, Alec McGhie

## Brief introduction

The goal of this project is to use logistic regression to predict heart disease from our dataset on incoming patients into a cleveland hospital


# Tabel of contents and code breakdown

## Model Creation `Predicating_heart_disease.ipynb`
The creation of this model required us to use the Sklearn Lab Logistic Regression library to make the regression model. The process of creating the model is outlined in the python notebook in section 2. Below is a code block for all of the code we used to create the model.
```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(heart_disease_features, 
                                                    heart_disease_labels, 
                                                    test_size=0.2, 
                                                    random_state=42)
                                                    
# Create a logistic regression model
logreg = LogisticRegression(max_iter= 1000)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Predict on the test data
y_pred = logreg.predict(X_test)
```

After creating this model we evaluated the metrics with the code below and got the output
```python
# Evaluate the model using various metrics such as accuracy, precision, recall, F1-score, etc.
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))

> Accuracy: 0.9259259259259259
> Precision: 0.9473684210526315
> Recall: 0.8571428571428571
> F1-score: 0.9
```

## Model Analysis `Predicating_heart_disease.ipynb`
#### Feature Importance
To analyse the model we first needed to find the varibles that had the most significant effect on the outcome of the patients heart disease status. To find this we used the code below.
```python
# Plot the feature importances (coefficients) of the logistic regression model
plt.figure(figsize=(10,5))
plt.bar(heart_disease_features.columns, logreg.coef_[0])
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Coefficients')
plt.title('Importance of Input Features in the Logistic Regression Model')
plt.show()

# Find the most important feature and its index
most_important_feature_index = np.argmax(np.abs(logreg.coef_[0]))
most_important_feature_name = heart_disease_features.columns[most_important_feature_index]

feature_importance = pd.DataFrame({"Feature": heart_disease_features.columns, "Importance": logreg.coef_[0]})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print(feature_importance)


>                     Feature  Importance
> 1                       Sex    1.047216
> 11  Number of vessels fluro    0.836211
> 8           Exercise angina    0.675063
> 2           Chest pain type    0.646917
> 9             ST depression    0.476928
> 10              Slope of ST    0.444288
> 12                 Thallium    0.328358
> 6               EKG results    0.092292
> 3                        BP    0.023329
> 4               Cholesterol    0.005032
> 7                    Max HR   -0.011386
> 0                       Age   -0.013954
> 5              FBS over 120   -0.668776

```
![Feature Importance](https://user-images.githubusercontent.com/123593094/235363877-1bc29aaf-29c4-43a2-9f45-25fcd4050f1e.png)
#### Heart Disease by Age Range
In this block of code we wanted to display the age ranges where heart disease is the most prevelent in the hospital data. For this we used `pyplot`.
```python
def calculate_hd_percentage(dataframe):
    heart_disease_count = dataframe['Heart Disease'].sum()
    percentage = (heart_disease_count / len(dataframe)) * 100
    return percentage

male_age_ranges = [(18, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80)]
female_age_ranges = [(18, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80)]

# Create a new DataFrame with age, sex, and heart disease data
age_sex_hd_data = heart_disease_df[['Age', 'Sex', 'Heart Disease']].copy()

# Replace sex codes with 'Male' and 'Female' for readability
age_sex_hd_data['Sex'] = age_sex_hd_data['Sex'].replace({1: 'Male', 0: 'Female'})

# Calculate the percentage of heart disease cases in each age range for both genders
male_percentages = [calculate_hd_percentage(age_sex_hd_data[(age_sex_hd_data['Sex'] == 'Male') &
                                                            (age_sex_hd_data['Age'] >= min_age) &
                                                            (age_sex_hd_data['Age'] <= max_age)])
                    for min_age, max_age in male_age_ranges]

female_percentages = [calculate_hd_percentage(age_sex_hd_data[(age_sex_hd_data['Sex'] == 'Female') &
                                                              (age_sex_hd_data['Age'] >= min_age) &
                                                              (age_sex_hd_data['Age'] <= max_age)])
                      for min_age, max_age in female_age_ranges]

# Calculate the number of data points in each age range for both genders
male_data_counts = [len(age_sex_hd_data[(age_sex_hd_data['Sex'] == 'Male') &
                                       (age_sex_hd_data['Age'] >= min_age) &
                                       (age_sex_hd_data['Age'] <= max_age)])
                    for min_age, max_age in male_age_ranges]

female_data_counts = [len(age_sex_hd_data[(age_sex_hd_data['Sex'] == 'Female') &
                                         (age_sex_hd_data['Age'] >= min_age) &
                                         (age_sex_hd_data['Age'] <= max_age)])
                      for min_age, max_age in female_age_ranges]

# Generate bar graphs for both genders
x_labels_male = [f'{min_age}-{max_age}' for min_age, max_age in male_age_ranges]
x_labels_female = [f'{min_age}-{max_age}' for min_age, max_age in female_age_ranges]

fig, ax = plt.subplots(2, 1, figsize=(10, 10))


# Formating the Male's Plot
male_bar = ax[0].bar(x_labels_male, male_percentages, color='blue')
ax[0].set_ylabel('Percentage of Heart Disease Cases')
ax[0].set_title('Percentage of Heart Disease Cases by Age Range (Male)')
ax[0].set_ylim(0, 100)

# Formating the Female's Plot
female_bar = ax[1].bar(x_labels_female, female_percentages, color='pink')
ax[1].set_ylabel('Percentage of Heart Disease Cases')
ax[1].set_title('Percentage of Heart Disease Cases by Age Range (Female)')
ax[1].set_ylim(0, 100)

# Annotate bars with the number of data points
for i, rect in enumerate(male_bar):
    height = rect.get_height()
    count = male_data_counts[i]
    ax[0].annotate(f'{count}', xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    if count <= 2:
        rect.set_color('red')

for i, rect in enumerate(female_bar):
    height = rect.get_height()
    count = female_data_counts[i]
    ax[1].annotate(f'{count}', xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    if count <= 2:
        rect.set_color('red')

# Create custom legend handles and labels
male_legend_elements = [Line2D([0], [0], marker='s', color='w', label='Male', markerfacecolor='blue', markersize=10),
                        Line2D([0], [0], marker='s', color='w', label='Skewed Data (1-2)', markerfacecolor='red', markersize=10)]

female_legend_elements = [Line2D([0], [0], marker='s', color='w', label='Female', markerfacecolor='pink', markersize=10),
                          Line2D([0], [0], marker='s', color='w', label='Skewed Data (1-2)', markerfacecolor='red', markersize=10)]

ax[0].legend(handles=male_legend_elements)
ax[1].legend(handles=female_legend_elements)

# Displaying the plots
plt.tight_layout()
plt.show()
```
Here are the graphs that are displayed from this code segment 
![image](https://user-images.githubusercontent.com/123593094/235364308-b70dd21b-89cc-4a7e-b32b-62ae7248390d.png)




## Logistical Regression Model `logistic_regression_model.pkl`
This model is created and saved through the python notebook above and provides the ability for the user to test data on the model using the `pickle` library.

## Model Demo `Heart_Disease_Prediction.py`
This model demo uses pythons TKinter and pickle librarys to create an interactive gui that runs the user data through `logistic_regression_model.pkl`. When using the model the doctor would enter in the values for the patent and then these values are run through the model. After the model concludes wether the data is positive or negitive the label at the bottom will be changed to reflect the output of the model. It is important to remember that our model is not a doctor and all of the assumptions from the model should not be used to diagnose heart disease.
#### How to use the model demo
<br> To use this model save `Heart_Disease_Prediction.py` and `logistic_regression_model.pkl` into the same folder on your computer. Then open `Heart_Disease_Prediction.py` in your IDE of choice. At this point make sure that the file path for the model is correct in line 6. From here you can run the code from the IDE and the TKinter GUI should open. While using the application make sure that all of the entry boxes have valid data before running the test.
## Model Demo Video `Heart Disease Model Demo.mp4`
This file contains two patients from the dataset one with a postive diagnosis of heart disease and one with a negitive diagnosis for heart disease.
