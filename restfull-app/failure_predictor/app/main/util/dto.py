from flask_restplus import Namespace, fields


class UserDto:
    api = Namespace('user', description='user related operations')
    user = api.model('user', {
        'email': fields.String(required=True, description='user email address'),
        'username': fields.String(required=True, description='user username'),
        'password': fields.String(required=True, description='user password'),
        'public_id': fields.String(description='user Identifier')
    })

class HDDSMARTPreictionDto:
    api = Namespace('SMARTPrediction', description='HDDSMARTPrediction related operations')
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
