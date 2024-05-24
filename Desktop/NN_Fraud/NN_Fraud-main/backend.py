from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import datetime
import haversine as hs

#DB connection
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.olwlkyaowkgdrbonvxpz:KdHg2RbLxjxhRlL7@aws-0-eu-central-1.pooler.supabase.com:5432/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class Client(db.Model):
    cl_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    homelocationX = db.Column(db.Float)
    homelocationY = db.Column(db.Float)

class Transaction(db.Model):
    tr_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    tr_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    cl_id = db.Column(db.Integer, db.ForeignKey('client.cl_id'))
    retailer = db.Column(db.Integer, nullable=False)
    used_chip = db.Column(db.Integer)
    used_pin = db.Column(db.Integer)
    tr_price = db.Column(db.Float,nullable=False)
    online_order = db.Column(db.Integer)
    transactionlocationX = db.Column(db.Float)
    transactionlocationY = db.Column(db.Float)

# Create database tables within Flask application context
with app.app_context():
    db.create_all()

# Modeli yükle
model = load_model("balanced_model.keras")

@app.route('/')
def index():
    clients = Client.query.all()
    transactions = Transaction.query.all()
    return render_template('index.html', clients=clients, transactions=transactions)

@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.form.get('client_id')
    transaction_id = request.form.get('transaction_id')
    # Tahmin sonucunu döndür ve index.html'e gönder
    return render_template('index.html')


@app.route('/add_client', methods=['GET', 'POST'])
def add_client():
    if request.method == 'POST':
        name = request.form['name']
        homelocationX = request.form['homelocationX']
        homelocationY = request.form['homelocationY']

        new_client = Client(name=name, homelocationX=homelocationX, homelocationY=homelocationY)
        db.session.add(new_client)
        db.session.commit()

        return redirect(url_for('index'))

    return render_template('client_form.html')


@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    if request.method == 'POST':
        client_id = request.form['client_id']
        retailer = request.form['retailer']
        used_chip = request.form['used_chip']
        used_pin = request.form['used_pin']
        online_order = request.form['online_order']
        tr_price=request.form['tr_price']
        transactionlocationX = request.form['transactionlocationX']
        transactionlocationY = request.form['transactionlocationY']

        new_transaction = Transaction(cl_id=client_id, retailer=retailer, used_chip=used_chip, used_pin=used_pin,
                                      online_order=online_order, transactionlocationX=transactionlocationX,
                                      transactionlocationY=transactionlocationY, tr_price=tr_price)
        db.session.add(new_transaction)
        db.session.commit()

        return redirect(url_for('index'))
    clients = Client.query.all()  # Fetch all clients
    return render_template('transaction_form.html', clients=clients)

@app.route('/clients')
def clients():
    clients = Client.query.all()
    return render_template('clients.html', clients=clients)

@app.route('/transactions/<int:client_id>')
def transactions(client_id):
    client = Client.query.get(client_id)
    transactions = Transaction.query.filter_by(cl_id=client_id).all()
    return render_template('transactions.html', client=client, transactions=transactions)

# Add this route to handle transaction deletion asynchronously
@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    # Find the transaction
    transaction = Transaction.query.get(transaction_id)
    if transaction:
        # Delete the transaction
        db.session.delete(transaction)
        db.session.commit()
        return "Transaction deleted successfully", 200
    else:
        # Handle error when transaction is not found
        return "Transaction not found", 404

@app.route('/transaction_perdict/<int:client_id>/<int:transaction_id>', methods=['GET', 'POST'])
def predict_transaction(transaction_id,client_id):
    # Fetch the selected client and transaction
    client = Client.query.get(client_id)
    transaction = Transaction.query.get(transaction_id)

    if client and transaction:
        # Calculate distance from home using Haversine formula
        distance_from_home = hs.haversine((client.homelocationX, client.homelocationY),
                                          (transaction.transactionlocationX, transaction.transactionlocationY))

        # Fetch previous transactions for the client
        previous_transactions = Transaction.query.filter_by(cl_id=client_id).order_by(Transaction.tr_date.desc()).all()

        # Calculate distance from last transaction if there are previous transactions
        if len(previous_transactions)>1:
            last_transaction = previous_transactions[1]
            distance_from_last_transaction = hs.haversine((last_transaction.transactionlocationX, last_transaction.transactionlocationY),
                                                           (transaction.transactionlocationX, transaction.transactionlocationY))
        else:
            # If no previous transactions, set distance_from_last_transaction to 0
            distance_from_last_transaction = 0

        # Calculate median purchase price
        all_transaction_prices = [prev_transaction.tr_price for prev_transaction in previous_transactions]
        median_purchase_price = np.median(all_transaction_prices) if all_transaction_prices else 0

        # Calculate ratio to median purchase price
        ratio_to_median_purchase_price = transaction.tr_price / median_purchase_price if median_purchase_price != 0 else 0
        retailers = [prev_transaction.retailer for prev_transaction in previous_transactions]
        if transaction.retailer in retailers:
            repeat_purchase = 1
            if retailers.count(transaction.retailer) == 1:
                repeat_purchase = 0
        else:
            repeat_purchase = 0
        # Extract features from client and transaction data
        feature1 = distance_from_home
        feature2 = distance_from_last_transaction
        feature3 = ratio_to_median_purchase_price
        feature4 = repeat_purchase
        feature5 = transaction.used_chip
        feature6 = transaction.used_pin
        feature7 = transaction.online_order
        features1 = {
            "Distance from home": distance_from_home,
            "Distance from last transaction": distance_from_last_transaction,
            "Ratio to median purchase price": ratio_to_median_purchase_price,
            "repeat_purchase": repeat_purchase,
            "Used chip": transaction.used_chip,
            "Used pin": transaction.used_pin,
            "Online order": transaction.online_order,
            "median_purchase_price":median_purchase_price,

        }

        # Scale the features
        features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7]])

        # Make prediction
        prediction = model.predict(features)

        if prediction >= 0.5:
            prediction_result = "Fraud"
        else:
            prediction_result = "Not Fraud"

        return render_template('predict_transaction.html', prediction=prediction_result, clients=Client.query.all(),
                               transactions=Transaction.query.all(),features=features1)
    else:
        # Handle error when client or transaction is not found
        prediction_result = "Client or transaction not found"

    return render_template('clients.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)



import numpy as np
from collections import Counter

# Veriler
data = np.array([
    [55, 110, 275, 169, 1],  # "Yes" için 1
    [65, 150, 268, 151, 1],  # "Yes" için 1
    [50, 120, 225, 140, 0],  # "No" için 0
    [40, 130, 351, 115, 0],  # "No" için 0
    [60, 125, 203, 170, 0],  # "No" için 0
    [48, 100, 305, 165, 0],  # "No" için 0
    [25, 103, 185, 120, 1],  # "Yes" için 1
    [53, 170, 270, 155, 1],  # "Yes" için 1
    [67, 140, 328, 128, 1],  # "Yes" için 1
    [72, 115, 400, 172, 1]   # "Yes" için 1
])

# Yeni gelen hasta
new_patient = np.array([70, 175, 200, 150])

# Her bir kayıt ile yeni hasta arasındaki Öklid mesafelerini hesaplayan fonksiyon
def calculate_distances(data, new_patient):
    distances = []
    for row in data:
        distance = np.sqrt(np.sum((row[:-1] - new_patient) ** 2))  # Son sütun sınıf bilgisini içermediği için -1
        distances.append((distance, row[-1]))  # Mesafe ve sınıf bilgisi tuple olarak saklanır
    distances.sort()  # Mesafelere göre sırala
    return distances

# KNN için en yakın k komşuyu bulan fonksiyon
def find_k_nearest_neighbors(distances, k):
    neighbors = [distance[1] for distance in distances[:k]]  # En yakın k komşuyu al
    return neighbors

# KNN ile sınıf tahmini yapan fonksiyon
def knn(neighbors):
    majority_vote = Counter(neighbors).most_common(1)[0][0]  # Çoğunluk oylaması
    return majority_vote

# K = 3 için hesaplamaları yap
k = 3
distances = calculate_distances(data, new_patient)
print("distance",distances)
nearest_neighbors = find_k_nearest_neighbors(distances, k)
prediction = knn(nearest_neighbors)

# Tahmin edilen sınıfı yazdır
if prediction == 1:
    print("Tahmin edilen kalp hastalığı durumu: Yes")
else:
    print("Tahmin edilen kalp hastalığı durumu: No")
