from datetime import datetime
from .. import db, flask_bcrypt

def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))

#class User(db.Model):
#    """ User Model for storing user related details """
#    __tablename__ = "user"
#
#    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#    email = db.Column(db.String(255), unique=True, nullable=False)
#    registered_on = db.Column(db.DateTime, nullable=False)
#    admin = db.Column(db.Boolean, nullable=False, default=False)
#    public_id = db.Column(db.String(100), unique=True)
#    username = db.Column(db.String(50), unique=True)
#    password_hash = db.Column(db.String(100))
#
#    @property
#    def password(self):
#        raise AttributeError('password: write-only field')
#
#    @password.setter
#    def password(self, password):
#        self.password_hash = flask_bcrypt.generate_password_hash(password).decode('utf-8')
#
#    def check_password(self, password):
#        return flask_bcrypt.check_password_hash(self.password_hash, password)
#
#    def __repr__(self):
#        return "<User '{}'>".format(self.username)



class HDDSMARTPrediction(db.Model):

    __tablename__ = 'hdd_smart_prediction'
    #__tablename__ = 'hddsmartprediction'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    model = db.Column(db.String(50))
    serial_number = db.Column(db.String(50))
    capacity_bytes = db.Column(db.BigInteger)
    failure = db.Column(db.Integer)
    prediction = db.Column(db.Float) 

    def __init__(self, model, serial_number, capacity_bytes, failure, prediction):
        self.timestamp = get_timestamp() #datetime.utcnow()
        self.model = model
        self.serial_number = serial_number
        self.capacity_bytes = capacity_bytes
        self.failure = failure
        self.prediction = prediction

