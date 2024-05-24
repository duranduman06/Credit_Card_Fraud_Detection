import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from warnings import filterwarnings
from keras.models import load_model

# Uyarıları ignore et
filterwarnings(action='ignore')

# Pandas ayarları
pd.set_option("display.width", 500)
pd.set_option("display.max.columns", 50)
pd.set_option("display.max.rows", 50)

# Veriyi yükleme ve ön işleme
df = pd.read_csv("card_transdata.csv")
df.drop_duplicates(inplace=True)

# Fraud ve non-fraud örnekleri ayırma
fraud_df = df[df['fraud'] == 1]
non_fraud_df = df[df['fraud'] == 0]

# Fraud ve non-fraud örneklerinin sayısını eşitleme
fraud_count = len(fraud_df)
non_fraud_count = len(non_fraud_df)
non_fraud_df = non_fraud_df.sample(n=fraud_count, random_state=42)

# Eşitlenmiş veri setini oluşturma
balanced_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)

# Outlier Değeri silme işlemi
balanced_df.drop(balanced_df["distance_from_last_transaction"].idxmax(), inplace=True)

# Min-max scaling işlemi
for column in balanced_df.columns[:-1]:  # 'fraud' sütununu dışarıda bırak
    min_val = balanced_df[column].min()
    max_val = balanced_df[column].max()
    balanced_df[column] = (balanced_df[column] - min_val) / (max_val - min_val)

X = balanced_df.drop(columns=['fraud'])
y = balanced_df['fraud']

# Veriyi eğitim, validation ve test kümelerine ayırma
X_train, X_validation_test, y_train, y_validation_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_validation, X_test, y_validation, y_test = train_test_split(X_validation_test, y_validation_test, test_size=0.5, random_state=42, stratify=y_validation_test)

def build_and_train_model(X_train, y_train, X_test, y_test, input_dim, epochs):
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
                        validation_data=(X_test, y_test))

    return model, history

# Modeli oluşturup eğitelim
model, history = build_and_train_model(X_train, y_train, X_validation, y_validation, input_dim=X_train.shape[1], epochs=10)

# Model performansını test etme
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")
test_accuracy = accuracy_score(y_test, y_pred)
test_cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix plot
plt.figure(figsize=(6, 4))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0.5, 1.5], ['Non-Fraud', 'Fraud'])
plt.yticks([0.5, 1.5], ['Non-Fraud', 'Fraud'])
plt.show()


# Modelin eğitim ve doğrulama (validation) setlerindeki performansını ölçme
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Eğitim, doğrulama ve test veri setlerindeki dağılım bilgileri
train_data_info = f"Training Data: Total {len(y_train)}, Fraud {sum(y_train)}, Non-Fraud {len(y_train) - sum(y_train)}"
val_data_info = f"Validation Data: Total {len(y_validation)}, Fraud {sum(y_validation)}, Non-Fraud {len(y_validation) - sum(y_validation)}"
test_data_info = f"Test Data: Total {len(y_test)}, Fraud {sum(y_test)}, Non-Fraud {len(y_test) - sum(y_test)}"




# Loss plot
plt.figure(figsize=(6, 4))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Accuracy plot
plt.figure(figsize=(6, 4))
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Model performansı
print("Test Data Info:")
print(test_data_info)
print("\nTest Accuracy:", test_accuracy)
print("Test Confusion Matrix:")
print(test_cm)

print("\n" + "="*30)
print("Data Distribution:")
print(train_data_info)
print(val_data_info)
print(test_data_info)

# Modeli kaydetme
#model.save("fraud_detection_model.h5")
print("Model saved as 'fraud_detection_model.h5")