import joblib
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import datetime
import haversine as hs

# DB connection
app = Flask(__name__)
app.config[
    'SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.olwlkyaowkgdrbonvxpz:KdHg2RbLxjxhRlL7@aws-0-eu-central-1.pooler.supabase.com:5432/postgres'
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
    tr_price = db.Column(db.Float, nullable=False)
    online_order = db.Column(db.Integer)
    transactionlocationX = db.Column(db.Float)
    transactionlocationY = db.Column(db.Float)


# Create database tables within Flask application context
with app.app_context():
    db.create_all()

# Load scaling parameters
with open('scaling_params.txt', 'r') as f:
    scaling_params = {}
    for line in f:
        column, min_val, max_val = line.strip().split(',')
        scaling_params[column] = {'min': float(min_val), 'max': float(max_val)}

# Load models
KNN_model = joblib.load('KNN_best_model.pkl')
log_model = joblib.load('Logistic Regression_best_model.pkl')
NB_model = joblib.load('Naive Bayes_best_model.pkl')
ann_model = load_model("fraud_detection_model.h5")


@app.route('/')
def index():
    clients = Client.query.all()
    transactions = Transaction.query.all()
    return render_template('index.html', clients=clients, transactions=transactions)


@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.form.get('client_id')
    transaction_id = request.form.get('transaction_id')
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
        tr_price = request.form['tr_price']
        transactionlocationX = request.form['transactionlocationX']
        transactionlocationY = request.form['transactionlocationY']

        new_transaction = Transaction(cl_id=client_id, retailer=retailer, used_chip=used_chip, used_pin=used_pin,
                                      online_order=online_order, transactionlocationX=transactionlocationX,
                                      transactionlocationY=transactionlocationY, tr_price=tr_price)
        db.session.add(new_transaction)
        db.session.commit()

        return redirect(url_for('index'))
    clients = Client.query.all()
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


@app.route('/transaction_predict/<int:client_id>/<int:transaction_id>', methods=['GET', 'POST'])
def predict_transaction(client_id, transaction_id):
    client = Client.query.get(client_id)
    transaction = Transaction.query.get(transaction_id)

    if client and transaction:
        distance_from_home = hs.haversine((client.homelocationX, client.homelocationY),
                                          (transaction.transactionlocationX, transaction.transactionlocationY))

        previous_transactions = Transaction.query.filter_by(cl_id=client_id).order_by(Transaction.tr_date.desc()).all()

        if len(previous_transactions) > 1:
            last_transaction = previous_transactions[1]
            distance_from_last_transaction = hs.haversine(
                (last_transaction.transactionlocationX, last_transaction.transactionlocationY),
                (transaction.transactionlocationX, transaction.transactionlocationY))
        else:
            distance_from_last_transaction = 0

        all_transaction_prices = [t.tr_price for t in previous_transactions]
        median_purchase_price = np.median(all_transaction_prices) if all_transaction_prices else 0

        ratio_to_median_purchase_price = transaction.tr_price / median_purchase_price if median_purchase_price != 0 else 0

        retailers = [t.retailer for t in previous_transactions]
        repeat_purchase = 1 if transaction.retailer in retailers else 0

        features = np.array([[
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_purchase,
            transaction.used_chip,
            transaction.used_pin,
            transaction.online_order
        ]])

        features_dict = {
            "Distance from home": distance_from_home,
            "Distance from last transaction": distance_from_last_transaction,
            "Ratio to median purchase price": ratio_to_median_purchase_price,
            "Repeat purchase": repeat_purchase,
            "Used chip": transaction.used_chip,
            "Used pin": transaction.used_pin,
            "Online order": transaction.online_order
        }

        # Normalize the features
        features[:, 0] = (features[:, 0] - scaling_params['distance_from_home']['min']) / (
                    scaling_params['distance_from_home']['max'] - scaling_params['distance_from_home']['min'])
        features[:, 1] = (features[:, 1] - scaling_params['distance_from_last_transaction']['min']) / (
                    scaling_params['distance_from_last_transaction']['max'] -
                    scaling_params['distance_from_last_transaction']['min'])
        features[:, 2] = (features[:, 2] - scaling_params['ratio_to_median_purchase_price']['min']) / (
                    scaling_params['ratio_to_median_purchase_price']['max'] -
                    scaling_params['ratio_to_median_purchase_price']['min'])
        features[:, 3] = (features[:, 3] - scaling_params['repeat_retailer']['min']) / (
                    scaling_params['repeat_retailer']['max'] - scaling_params['repeat_retailer']['min'])
        features[:, 4] = (features[:, 4] - scaling_params['used_chip']['min']) / (
                    scaling_params['used_chip']['max'] - scaling_params['used_chip']['min'])
        features[:, 5] = (features[:, 5] - scaling_params['used_pin_number']['min']) / (
                    scaling_params['used_pin_number']['max'] - scaling_params['used_pin_number']['min'])
        features[:, 6] = (features[:, 6] - scaling_params['online_order']['min']) / (
                    scaling_params['online_order']['max'] - scaling_params['online_order']['min'])

        KNN_prediction = KNN_model.predict(features)[0]
        log_prediction = log_model.predict(features)[0]
        NB_prediction = NB_model.predict(features)[0]
        ann_prediction = ann_model.predict(features)[0][0]

        predictions = {
            'KNN Prediction': 'Fraud' if KNN_prediction == 1 else 'Not Fraud',
            'Logistic Regression Prediction': 'Fraud' if log_prediction == 1 else 'Not Fraud',
            'Naive Bayes Prediction': 'Fraud' if NB_prediction == 1 else 'Not Fraud',
            'ANN Prediction': 'Fraud' if ann_prediction >= 0.5 else 'Not Fraud'
        }

        return render_template('predict_transaction.html', client=client, transaction=transaction,
                               predictions=predictions, features=features_dict)


if __name__ == '__main__':
    app.run(debug=True)
