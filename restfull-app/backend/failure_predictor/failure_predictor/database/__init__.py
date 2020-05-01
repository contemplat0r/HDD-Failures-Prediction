from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def reset_database():
    from faulure_predictor.database.models import HDDSMARTPrediction  # noqa
    db.drop_all()
    db.create_all()
