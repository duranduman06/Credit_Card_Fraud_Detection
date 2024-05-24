from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import datetime
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.olwlkyaowkgdrbonvxpz:KdHg2RbLxjxhRlL7@aws-0-eu-central-1.pooler.supabase.com:5432/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#DB connection

class client(db.Model):
    cl_id=db.Column(db.Integer,autoincrement=True, primary_key=True)
    name=db.Column(db.String(50),unique=True,nullable=False)
    homelocationX=db.Column(db.Float)
    homelocationY=db.Column(db.Float)
class Transaction(db.Model):
    tr_id = db.Column(db.Integer,primary_key=True,autoincrement=True,)
    tr_date=db.Column(db.DateTime,nullable=False, default=datetime.datetime.now())
    cl_id=db.Column(db.Integer, db.ForeignKey('client.cl_id'))
    retailer=db.Column(db.Integer,nullable=False)
    used_chip=db.Column(db.Integer)
    used_pin=db.Column(db.Integer)
    online_order=db.Column(db.Integer)
    transactionlocationX=db.Column(db.Float)
    transactionlocationY=db.Column(db.Float)
with app.app_context():
    db.create_all()