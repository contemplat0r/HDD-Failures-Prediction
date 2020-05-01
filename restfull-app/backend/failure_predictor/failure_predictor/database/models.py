from datetime import datetime

from failure_predictor.database import db

def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))


class HDDSMARTPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    timestamp = db.Column(db.DateTime)
    model = db.Column(db.String(40))
    serial_number = db.Column(db.String(40))
    capacity_bytes = db.Column(db.Integer)
    failure = db.Column(db.Integer)
    prediction = db.Column(db.Float) 

    def __init__(self, model, serial_number, capacity_bytes, failure, prediction):
        self.timestamp = get_timestamp() #datetime.utcnow()
        self.model = model
        self.serial_number = serial_number
        self.capacity_bytes = capacity_bytes
        self.failure = failure
        self.prediction = prediction

    #def __repr__(self):
    #    return '<Post %r>' % self.title
