import uuid
import datetime

from app.main import db
from app.main.model.failures import HDDSMARTPrediction

def get_timestamp():
    return datetime.now().strftime(("%Y-%m-%d %H:%M:%S"))


def save_new_hdd_smart_predictions(predictions):
    #HDDSMARTPrediction.__table__.insert().execute(predictions)
    return db.engine.execute(HDDSMARTPrediction.__table__.insert(), predictions)


def get_all_failure_predictions():
    return HDDSMARTPrediction.query.all()


#def get_a_user(public_id):
#    return User.query.filter_by(public_id=public_id).first()


def save_changes(data):
    db.session.add(data)
    db.session.commit()
