# app/__init__.py

from flask_restplus import Api
from flask import Blueprint

#from .main.service import failures_service
from .main.controller.hdd_smart_prediction import api as hdd_smart_prediction_ns
#from .main.controller import hdd_smaprt_prediction #import api as hdd_smaprt_prediction_ns
#hdd_smaprt_prediction_ns = hdd_smaprt_prediction_ns.api

blueprint = Blueprint('api', __name__)

api = Api(
	blueprint,
	title="RESTPLUS API for HDD FAILURES PREDICTION by SMART DATA",
	version='0.1',
	description='failures prediction web service'
    )

api.add_namespace(hdd_smart_prediction_ns, path='/prediction')
