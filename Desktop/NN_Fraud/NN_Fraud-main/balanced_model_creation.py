import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')

pd.set_option("display.width", 500)
pd.set_option("display.max.columns", 50)
pd.set_option("display.max.rows", 50)

# Veriyi yükleme ve ön işleme
df = pd.read_csv("card_transdata.csv")
df.drop_duplicates(inplace=True)

# Eksik değerlere sahip sütunları kontrol etme
print("ISNULL:")
print(df.isnull().sum(),"\n")

fraud_1_count = df[df["fraud"] == 1].shape[0]  # fraud == 1 olanların sayısını bulma

# fraud == 1 olan verilere eşit miktarda rastgele fraud == 0 olan verileri seçme
fraud_0_data = df[df["fraud"] == 0].sample(n=fraud_1_count, random_state=42)
fraud_1_data = df[df["fraud"] == 1]  # fraud == 1 olan verileri ayrı bir DataFrame'e dönüştürme

balanced_data = pd.concat([fraud_0_data, fraud_1_data])  # Hem fraud == 0 hem de fraud == 1 olan verileri birleştirme

print(balanced_data["fraud"].value_counts())  # Sonuçları kontrol etme

X = np.array(balanced_data.drop(columns="fraud"))
y = np.array(balanced_data["fraud"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=123, shuffle=True)

# y_train ve y_test veri tiplerini kontrol etme
print("\nX_train veri tipi:", X_train.dtype)
print("y_train veri tipi:", y_train.dtype)

# y_train içindeki fraud (1) ve non-fraud (0) değerlerinin sayısını bulma
y_train_fraud_count = np.sum(y_train == 1)
y_train_non_fraud_count = np.sum(y_train == 0)

# y_test içindeki fraud (1) ve non-fraud (0) değerlerinin sayısını bulma
y_test_fraud_count = np.sum(y_test == 1)
y_test_non_fraud_count = np.sum(y_test == 0)

# Sonuçları yazdırma
print("\nEğitim verisi için fraud (1) sayısı:", y_train_fraud_count)
print("Eğitim verisi için non-fraud (0) sayısı:", y_train_non_fraud_count)
print("Test verisi için fraud (1) sayısı:", y_test_fraud_count)
print("Test verisi için non-fraud (0) sayısı:", y_test_non_fraud_count)

# Eğitim verisi için yüzdelik oranları hesaplama
y_train_fraud_percent = (y_train_fraud_count / len(y_train)) * 100
y_train_non_fraud_percent = (y_train_non_fraud_count / len(y_train)) * 100

# Test verisi için yüzdelik oranları hesaplama
y_test_fraud_percent = (y_test_fraud_count / len(y_test)) * 100
y_test_non_fraud_percent = (y_test_non_fraud_count / len(y_test)) * 100

# Sonuçları yazdırma
print("\nEğitim verisi için fraud (1) yüzdesi: {:.2f}%".format(y_train_fraud_percent))
print("Eğitim verisi için non-fraud (0) yüzdesi: {:.2f}%".format(y_train_non_fraud_percent))
print("Test verisi için fraud (1) yüzdesi: {:.2f}%".format(y_test_fraud_percent))
print("Test verisi için non-fraud (0) yüzdesi: {:.2f}%".format(y_test_non_fraud_percent))


# Neural Network modelini tanımlayan fonksiyon
def build_and_train_nn(X_train, y_train, X_test, y_test, input_dim, epochs=10):
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_dim))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))  # Binary classification için son katman

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(patience=5, verbose=1)])

    return model, history


# KNN modelini tanımlayan fonksiyon
def build_and_train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)

    y_train_pred = knn_model.predict(X_train)
    y_test_pred = knn_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\nKNN - Eğitim Doğruluğu: {:.2f}%".format(train_accuracy * 100))
    print("KNN - Test Doğruluğu: {:.2f}%".format(test_accuracy * 100))
    print("\nKNN - Test Sınıflandırma Raporu:\n", classification_report(y_test, y_test_pred))

    return knn_model


from sklearn.svm import SVC
# SVM Modelini tanımlayan fonksiyon
def build_and_train_svm(X_train, y_train, X_test, y_test):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\nSVM - Eğitim Doğruluğu: {:.2f}%".format(train_accuracy * 100))
    print("SVM - Test Doğruluğu: {:.2f}%".format(test_accuracy * 100))
    print("\nSVM - Test Sınıflandırma Raporu:\n", classification_report(y_test, y_test_pred))

    return svm_model

from sklearn.ensemble import RandomForestClassifier

def build_and_train_random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\nRandom Forest - Eğitim Doğruluğu: {:.2f}%".format(train_accuracy * 100))
    print("Random Forest - Test Doğruluğu: {:.2f}%".format(test_accuracy * 100))
    print("\nRandom Forest - Test Sınıflandırma Raporu:\n", classification_report(y_test, y_test_pred))

    return rf_model

# Random Forest modelini eğitme
rf_model = build_and_train_random_forest(X_train, y_train, X_test, y_test)

from sklearn.ensemble import GradientBoostingClassifier

def build_and_train_gradient_boosting(X_train, y_train, X_test, y_test):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)

    y_train_pred = gb_model.predict(X_train)
    y_test_pred = gb_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\nGradient Boosting - Eğitim Doğruluğu: {:.2f}%".format(train_accuracy * 100))
    print("Gradient Boosting - Test Doğruluğu: {:.2f}%".format(test_accuracy * 100))
    print("\nGradient Boosting - Test Sınıflandırma Raporu:\n", classification_report(y_test, y_test_pred))

    return gb_model

# Gradient Boosting modelini eğitme
gb_model = build_and_train_gradient_boosting(X_train, y_train, X_test, y_test)

"""# NN Modelini eğitme
input_dim = X_train.shape[1]
model, history = build_and_train_nn(X_train, y_train, X_test, y_test, input_dim, epochs=10)"""

# KNN modelini eğitme
knn_model = build_and_train_knn(X_train, y_train, X_test, y_test, n_neighbors=5)


