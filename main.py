import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/ukshr/PycharmProjects/Tesla-StockPrice-Prediction/TSLA.csv")

df = df.drop(['Adj Close'], axis=1)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df['day'] = df['Date'].dt.day

df['month'] = df['Date'].dt.month

df['year'] = df['Date'].dt.year

df['is_quarter_end'] = np.where(df['Date'].dt.is_quarter_end, 1, 0)

df['open-close'] = df['Open'] - df['Close']

df['low-high'] = df['Low'] - df['High']

df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)

    sb.distplot(df[col])

plt.tight_layout()

plt.show()

plt.figure(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)

print(X_train.shape, X_valid.shape)

models = [LogisticRegression(), SVC(kernel='poly', probability=True), RandomForestClassifier()]

for i in range(3):

    models[i].fit(X_train, Y_train)
    print(f'{models[i]} : ')
    print('Training AUC : ', roc_auc_score(Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation AUC : ', roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:, 1]))
    print()

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(Y_valid, models[i].predict(X_valid))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {models[i]}')
    plt.show()


