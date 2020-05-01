from flask_restplus import fields
from failure_predictor.api.restplus import api

smart_prediction = api.model(
        'S.M.A.R.T prediction',
        {
            #'timestamp': fields.String(readOnly=True, description='Prediction Timestamp'),
            'timestamp': fields.DateTime,
            'model': fields.String(required=True, description='HDD model'),
            'serial_number': fields.String(required=True, description='HDD serial number'),
            'capacity_bytes': fields.Integer,
            'failure': fields.Integer,
            'prediction': fields.Float
        }
    )

#smart_prediction = api.model(
#        'S.M.A.R.T prediction',
#        [
#            {
#                #'timestamp': fields.String(readOnly=True, description='Prediction Timestamp'),
#                'timestamp': fields.DateTime,
#                'model': fields.String(required=True, description='HDD model'),
#                'serial_number': fields.String(required=True, description='HDD serial number'),
#                'capacity_bytes': fields.Integer,
#                'failure': fields.Integer,
#                'prediction': fields.Float
#            }
#        ]
#    )
