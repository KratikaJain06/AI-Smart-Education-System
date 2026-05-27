import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import r2_score


df=pd.read_csv("./data/corrupted_student_lifestyle_dataset.csv")

#Clean the dataset

#Remove duplicates
df.drop_duplicates(inplace=True)

#Rows with missing values
df.dropna(inplace = True)

#Fix column names
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns)

#Remove junk column if present
if 'unnamed: 4' in df.columns:
    df.drop(columns=['unnamed: 4'], inplace=True)

#Remove inconsistency
df['stress_level']=df['stress_level'].astype(str).str.lower().str.strip()

#Remove outliers using inter-quartile range

plt.boxplot(df['study_hours_per_day'])

plt.title('Boxplot of Study Hours')
plt.ylabel('Study Hours')

plt.show()

Q1 = df['study_hours_per_day'].quantile(0.25)
Q3 = df['study_hours_per_day'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['study_hours_per_day'] < upper) & 
        (df['study_hours_per_day'] > lower)]

#Plot graphs

plt.hist(df['study_hours_per_day'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Study Hours')
plt.ylabel('Frequency')
plt.show()

plt.hist(df['sleep_hours_per_day'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Sleep Hours')
plt.ylabel('Frequency')
plt.show()

plt.boxplot(df['study_hours_per_day'])

plt.title('Boxplot of Study Hours')
plt.ylabel('Study Hours')

plt.show()

plt.scatter(df['study_hours_per_day'], df['gpa'],color='skyblue',marker='o')
plt.xlabel('Study Hours')
plt.ylabel('GPA')
plt.show()

plt.scatter(df['sleep_hours_per_day'], df['gpa'],color='skyblue',marker='o')
plt.xlabel('Sleep Hours')
plt.ylabel('GPA')
plt.show()

plt.scatter(df['social_hours_per_day'], df['gpa'],color='skyblue',marker='o')
plt.xlabel('Social Media Hours')
plt.ylabel('GPA')
plt.show()

counts=df['stress_level'].value_counts()
ind=counts.index
plt.bar(ind,counts,color='skyblue')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.show()

c=df['physical_activity_hours_per_day'].value_counts()
i=c.index
plt.bar(i,c,color='skyblue')
plt.xlabel('Physical Activities per Day')
plt.ylabel('Count')
plt.show()

plt.pie(counts,labels=ind)
plt.show()

co_mtx = df.corr(numeric_only=True)
sns.heatmap(co_mtx, cmap="YlGnBu", annot=True)
plt.show()

#model building

#encoding
le=LabelEncoder()
df['stress_encoded'] = le.fit_transform(df['stress_level'])
#train-test-split
X=df[['study_hours_per_day','sleep_hours_per_day','stress_encoded','physical_activity_hours_per_day']]
Y=df['gpa']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=200)
#linear regression
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
#prediction
y_pred=regr.predict(X_test)
print(df['gpa'])
print(y_pred)
#accuracy
score = r2_score(y_test, y_pred)
print("R2 Score:", score)
plt.scatter(y_test, y_pred)

plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")

plt.title("Actual vs Predicted GPA")

plt.show()

coefficients = pd.DataFrame(
    regr.coef_,
    X.columns,
    columns=['Coefficient']
)

print(coefficients)

df.to_csv("cleaned_data.csv", index=False)