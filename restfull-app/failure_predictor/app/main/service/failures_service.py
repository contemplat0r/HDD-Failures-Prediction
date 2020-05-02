import uuid
import datetime

from app.main import db
from app.main.model.failures import HDDSMARTPrediction

def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))

def save_new_hdd_smart_prediction(data):
    hdd_smart_prediction = HDDSMARTPrediction.query.filter_by(timestamp=data['timestamp'], model=data['model'], serial_number=data['serial_number']).first()
    if not hdd_smart_prediction:
        new_hdd_smart_prediction = HDDSMARTPrediction(
            timestamp=get_timestamp()
            model=data['model'],
            serial_number=data['serial_number']
            capacity_bytes=data['capacity_bytes'],
            failure=data['failure']
            prediction=data['prediction']
            #password=data['password'],
            #registered_on=datetime.datetime.utcnow()
        )
        save_changes(new_hdd_smart_prediction)
        response_object = {
            'status': 'success',
            'message': "Successfully created."
        }
        return response_object, 201
    else:
        response_object = {
            'status': 'fail',
            'message': 'User already exists. Please Log in.',
        }
        return response_object, 409


def get_all_users():
    return HDDSMARTPrediction.query.all()


def get_a_user(public_id):
    return User.query.filter_by(public_id=public_id).first()


def save_changes(data):
    db.session.add(data)
    db.session.commit()
