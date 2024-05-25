import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from warnings import filterwarnings

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
print("fraud_count", fraud_count)
print("non-fraud_count", non_fraud_count, "\n")
non_fraud_df = non_fraud_df.sample(n=fraud_count, random_state=42)

# Eşitlenmiş veri setini oluşturma
balanced_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)

# Outlier Değeri silme işlemi
balanced_df.drop(balanced_df["distance_from_last_transaction"].idxmax(), inplace=True)
print(balanced_df.describe(), "\n")
print(balanced_df.info(), "\n")
print(balanced_df.isnull().sum(), "\n")

# Balanced dataset için dağılım grafiği
plt.figure(figsize=(6, 4))
sns.countplot(x='fraud', data=balanced_df)
plt.title('Class Distribution in Balanced Dataset')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'])
plt.show()

# Box plot analizi
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
sns.boxplot(x=balanced_df['distance_from_home'])
plt.title('Box Plot of Distance from Home')
plt.subplot(3, 1, 2)
sns.boxplot(x=balanced_df['distance_from_last_transaction'])
plt.title('Box Plot of Distance from Last Transaction')
plt.subplot(3, 1, 3)
sns.boxplot(x=balanced_df['distance_from_last_transaction'])
plt.title('Box Plot of Distance from Last Transaction')
plt.tight_layout()
plt.show()

# Min-max scaling işlemi
scaling_params = {}
for column in balanced_df.columns[:-1]:  # 'fraud' sütununu dışarıda bırak
    min_val = balanced_df[column].min()
    max_val = balanced_df[column].max()
    scaling_params[column] = {'min': min_val, 'max': max_val}
    balanced_df[column] = (balanced_df[column] - min_val) / (max_val - min_val)

# Save scaling parameters to a text file
with open('scaling_params.txt', 'w') as f:
    for column, params in scaling_params.items():
        f.write(f"{column},{params['min']},{params['max']}\n")

X = balanced_df.drop(columns=['fraud'])
y = balanced_df['fraud']

# Etiket sayılarını sayma
fraud_count = y.sum()  # fraud etiketlerinin toplam sayısı
non_fraud_count = len(y) - fraud_count  # non-fraud etiketlerinin toplam sayısı
print(f"Toplam 'fraud' (1) örneği sayısı: {fraud_count}")
print(f"Toplam 'non-fraud' (0) örneği sayısı: {non_fraud_count}")

# Fonksiyonlar
def evaluate_model(model, model_name, X, y, n_splits=10, random_state=42):
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cm = np.zeros((2, 2), dtype=int)

    best_accuracy = 0
    best_model = None
    best_cm = None

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

        iteration_cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm += iteration_cm

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_cm = iteration_cm

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)

    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f"\n{model_name} Model Performance Metrics (Average ± Std):")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")

    # Compute average confusion matrix
    cm = cm / n_splits
    print(f"\n{model_name} Average Confusion Matrix:")
    print(cm.astype(int))  # Print as integer

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Fraud', 'Fraud'])
    plt.yticks(tick_marks, ['Non-Fraud', 'Fraud'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(int(cm[i, j]), fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    print(f"\nBest Model ({model_name}) selected with Accuracy = {best_accuracy:.4f}")

    # Best Confusion Matrix
    print(f"\n{model_name} Best Confusion Matrix:")
    print(best_cm)

    # Modeli .pkl dosyası olarak kaydetme
    #filename = f"{model_name}_best_model.pkl"
    #joblib.dump(best_model, filename)
    #print(f"Best Model saved as {filename}")

    return best_model

# KNN için değerlendirme
knn = KNeighborsClassifier(n_neighbors=7)
print("\nKNN Model Performance:")
evaluate_model(knn, "KNN", X, y)

# Logistic Regression için değerlendirme
log_reg = LogisticRegression(random_state=42)
print("\nLogistic Regression Model Performance:")
evaluate_model(log_reg, "Logistic Regression", X, y)

# Naive Bayes için değerlendirme
nb = GaussianNB()
print("\nNaive Bayes Model Performance:")
evaluate_model(nb, "Naive Bayes", X, y)
