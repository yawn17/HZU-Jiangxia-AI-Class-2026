import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

df = pd.read_csv('./dev.csv')
# print(df.head())

data = df['word'].values
# print(data)

transform = TfidfVectorizer()
feature_vector = transform.fit_transform(data)

# print(feature_vector.data)

label = df['label'].values
x_train, x_test, y_train, y_test = train_test_split(feature_vector, label,test_size=0.2, random_state=22)

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average='macro'))
print(recall_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='macro'))








