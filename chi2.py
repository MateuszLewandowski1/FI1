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
table_min_max = table.copy()
table_min_max = table_min_max.dropna(subset=['smoking_status'])
# fill in the data for bmi
# table_non_smokers.bmi.median() ==> 28.9

values = {'bmi': 28.9}
table_min_max = table_min_max.fillna(value=values)
# control check
# table_non_smokers.bmi.isnull().sum() ==> 0

# dropping an irrelevant and target column
y = table_min_max.stroke
table_min_max = table_min_max.drop(columns=['id', 'stroke'])

table_min_max = pd.get_dummies(table_min_max, columns=['smoking_status', 'Residence_type',
                                                       'work_type', 'ever_married', 'gender'
                                                       ])

# standardize age, bmi, avg_glucose_level

scaler = MinMaxScaler()
scaler.fit(table_min_max['age'].values.reshape(-1, 1))
table_min_max['age'] = scaler.transform(table_min_max['age'].values.reshape(-1, 1))

scaler = MinMaxScaler()
scaler.fit(table_min_max['bmi'].values.reshape(-1, 1))
table_min_max['bmi'] = scaler.transform(table_min_max['bmi'].values.reshape(-1, 1))

scaler = MinMaxScaler()
scaler.fit(table_min_max['avg_glucose_level'].values.reshape(-1, 1))
table_min_max['avg_glucose_level'] = scaler.transform(table_min_max['avg_glucose_level'].values.reshape(-1, 1))

# is the dataset balanced?

# table.stroke.sum() ==> 783
# 783/43400 = 0.018 ==> it is not balanced
# hence I need to be careful, because with unbalanced dataset, model can just learn to predict that there was
# no stroke at all
# there are two techniques that I am familiar with to proceed now -- SMOTE and ROSE. In my opinion SMOTE produced
# better results and both SMOTE and ROSE outperform not oversampling at all

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(table_min_max, y)
# y_res.sum()
# X_res.shape ==> (58940, 20)
# y_res.shape ==> (58940, ) ==> now it is balanced
# y_res.sum() ==> 29470

# now train test split 80%, 20% and fit logistic regression



bestfeatures = SelectKBest(score_func=chi2, k=10)
feat = bestfeatures.fit(X_res, y_res)
dfscores = pd.DataFrame(feat.scores_)
dfcolumns = pd.DataFrame(X_res.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(20, 'Score'))  # print features in a decreasing order

#                              Specs        Score
# 15                 ever_married_No  3139.175402
# 0                              age  1907.424914
# 2                    heart_disease  1346.054867
# 13         work_type_Self-employed  1064.316714
# 1                     hypertension   915.124613
# 14              work_type_children   631.000000
# 3                avg_glucose_level   475.422358
# 5   smoking_status_formerly smoked   451.139438
# 16                ever_married_Yes   427.374683
# 6      smoking_status_never smoked   257.717870
# 10              work_type_Govt_job   234.734401
# 12               work_type_Private   156.665160
# 11          work_type_Never_worked   101.000000
# 18                     gender_Male    65.481794
# 17                   gender_Female    64.221562
# 7            smoking_status_smokes    17.294796
# 19                    gender_Other     9.000000
# 8             Residence_type_Rural     8.186469
# 4                              bmi     2.665180
# 9             Residence_type_Urban     1.930166



