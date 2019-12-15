import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import copy
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

path_to_train = '/home/mateusz1/Downloads/healthcare-dataset-stroke-data/train_2v.csv'

table = pd.read_csv(path_to_train)

# table.columns
# Index(['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
#        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
#        'smoking_status', 'stroke'],
#       dtype='object')


# diving into variables

# table.gender.unique()
# array(['Male', 'Female', 'Other'], dtype=object)

# age

# table.age.unique().shape
# (104,)
#

# table.hypertension.unique()
# array([0, 1])

# table.heart_disease.unique()
# array([0, 1])

# table.ever_married.unique()
# array(['No', 'Yes'], dtype=object)

# table.work_type.unique()
# array(['children', 'Private', 'Never_worked', 'Self-employed', 'Govt_job'],
#       dtype=object)

# table.Residence_type.unique()
# array(['Rural', 'Urban'], dtype=object)

# table.avg_glucose_level.unique().shape
# (12543,)

# table.bmi.unique().shape
# (556,)

# table.smoking_status.unique()
# array([nan, 'never smoked', 'formerly smoked', 'smokes'], dtype=object)

# and finally the y values :)

# table.stroke.unique()
# array([0, 1])

# clean the dataset

table.isnull().values.sum()
# 14754 --> that many values to clean, however

# table.smoking_status.isnull().values.sum()
# 13292 --> first thought is that there is that many children

# sum(table.age < 18)
# 7541 --> around 57% of nans in smoking status is due to young age

# looking for more NaN data
# Index(['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
#        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
#        'smoking_status', 'stroke'],
# getting dummies of string-type variables

# table.gender.isnull().values.sum() --> 0
# table.age.isnull().values.sum() --> 0
# table.hypertension.isnull().values.sum() --> 0
# table.heart_disease.isnull().values.sum() --> 0
# table.ever_married.isnull().values.sum() --> 0
# table.work_type.isnull().values.sum() --> 0
# table.Residence_type.isnull().values.sum() --> 0
# table.avg_glucose_level.isnull().values.sum() --> 0
# table.bmi.isnull().values.sum() --> 1462
# 1462+13292 = 14754 which is the total number of missing data in the dataset

# 13292/43400 = 0.3062672811059908  ==> one third of the data from smokers is nans, I will not fill it in with anything,
# as I have no clue what to fill that with. after some inspection of data I noticed, that there is no easily noticeable
# characteristic of data that could explain occurrences of nans (like young age)

# 1462/43400 = 0.0336 ==> 3% of data for bmi has a nan as a value, I will fill that in with a median

# drop data for smokers
table_non_smokers = table.copy()
table_non_smokers = table_non_smokers.dropna(subset=['smoking_status'])
# fill in the data for bmi
# table_non_smokers.bmi.median() ==> 28.9

values = {'bmi': 28.9}
table_non_smokers = table_non_smokers.fillna(value=values)
# control check
# table_non_smokers.bmi.isnull().sum() ==> 0

# dropping an irrelevant and target column
y = table_non_smokers.stroke
table_non_smokers = table_non_smokers.drop(columns=['id', 'stroke'])

table_dummies = pd.get_dummies(table_non_smokers, columns=['smoking_status', 'Residence_type',
                                                       'work_type', 'ever_married', 'gender'
                                                       ])

# standardize age, bmi, avg_glucose_level

scaler = StandardScaler()
scaler.fit(table_dummies['age'].values.reshape(-1, 1))
table_dummies['age'] = scaler.transform(table_dummies['age'].values.reshape(-1, 1))

scaler = StandardScaler()
scaler.fit(table_dummies['bmi'].values.reshape(-1, 1))
table_dummies['bmi'] = scaler.transform(table_dummies['bmi'].values.reshape(-1, 1))

scaler = StandardScaler()
scaler.fit(table_dummies['avg_glucose_level'].values.reshape(-1, 1))
table_dummies['avg_glucose_level'] = scaler.transform(table_dummies['avg_glucose_level'].values.reshape(-1, 1))

# is the dataset balanced?

# table.stroke.sum() ==> 783
# 783/43400 = 0.018 ==> it is not balanced
# hence I need to be careful, because with unbalanced dataset, model can just learn to predict that there was
# no stroke at all
# there are two techniques that I am familiar with to proceed now -- SMOTE and ROSE. In my opinion SMOTE produced
# better results and both outperform not oversampling at all

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(table_dummies, y)

# X_res.shape ==> (58940, 20)
# y_res.shape ==> (58940, ) ==> now it is balanced
# y_res.sum() ==> 29470

# now train test split 80%, 20% and fit logistic regression

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# ==> 0.8139633525619274 - that makes a reasonable model accuracy, I will proceed and draw conclusions
# from models coefficients


def f(arr):
    return np.exp(arr)/(1+np.exp(arr))  # to get back from logistic regression


featureScores = pd.concat([pd.DataFrame(table_dummies.columns), pd.DataFrame(f(clf.coef_)[0])], axis=1)
featureScores.columns = ['feature', 'score']  # naming the dataframe columns
print(featureScores.nlargest(20, 'score'))  # print features in a decreasing order

#                            feature     score
# 0                              age  0.860480
# 3                avg_glucose_level  0.545778
# 4                              bmi  0.464191
# 2                    heart_disease  0.355446
# 1                     hypertension  0.341566
# 19                    gender_Other  0.071630
# 11          work_type_Never_worked  0.030606
# 14              work_type_children  0.019388
# 16                ever_married_Yes  0.007515
# 12               work_type_Private  0.007163
# 17                   gender_Female  0.006693
# 18                     gender_Male  0.006655
# 9             Residence_type_Urban  0.006451
# 13         work_type_Self-employed  0.005732
# 8             Residence_type_Rural  0.005632
# 7            smoking_status_smokes  0.004571
# 5   smoking_status_formerly smoked  0.003860
# 6      smoking_status_never smoked  0.003142
# 10              work_type_Govt_job  0.002752
# 15                 ever_married_No  0.002211

# what indicates, that smoking, residence type, work type, gender, or marriage status have little influence.
# knowing that smoking should factor in, I proceed with a different method