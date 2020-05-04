# app/__init__.py

from flask_restplus import Api
from flask import Blueprint

from .main.controller.hdd_smart_prediction import api as hdd_smart_prediction_ns

blueprint = Blueprint('api', __name__)

api = Api(
	blueprint,
	title="RESTPLUS API for HDD FAILURES PREDICTION by SMART DATA",
	version='0.1',
	description='failures prediction web service'
    )

api.add_namespace(hdd_smart_prediction_ns, path='/prediction')
