import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=1)
dtc = DecisionTreeClassifier(random_state=0)
svm = SVC()
nb = MultinomialNB()
mlp = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0)
gbc = GradientBoostingClassifier(n_estimators=10)

dc = ['greek', 'southern_us' ,'filipino', 'indian' ,'jamaican', 'spanish', 'italian',
 'mexican' ,'chinese' ,'british' ,'thai', 'vietnamese' ,'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan' ,'russian']

data = pd.read_json('train.json')


x = data['ingredients']
data['all_ind'] = data['ingredients'].map(';'.join)

tfd = TfidfVectorizer(use_idf=True)
x = tfd.fit_transform(data['all_ind'])

enc = LabelEncoder()
y = enc.fit_transform(data.cuisine)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

svm.fit(x_train, y_train)

logr.fit(x_train, y_train)
rfc.fit(x_train, y_train)
dtc.fit(x_train, y_train)
svm.fit(x_train, y_train)
mlp.fit(x_train, y_train)
gbc.fit(x_train, y_train)
nb.fit(x_train, y_train)

ylogr_predict = logr.predict(x_test)
rfcy_predict = rfc.predict(x_test)
dtcy_predict = dtc.predict(x_test)
svmy_predict = svm.predict(x_test)
mlpy_predict = mlp.predict(x_test)
gbcy_predict = gbc.predict(x_test)
nby_predict = nb.predict(x_test)

print('Logistic:', accuracy_score(y_test, ylogr_predict))
print('Random Forest:', accuracy_score(y_test, rfcy_predict))
print('Decision Tree:', accuracy_score(y_test, dtcy_predict))
print('Support Vector:', accuracy_score(y_test, svmy_predict))
print('MLP:', accuracy_score(y_test,  mlpy_predict))
print('Gradient Boosting:', accuracy_score(y_test,  gbcy_predict))
print('Naive Bayes:', accuracy_score(y_test,  nby_predict))
