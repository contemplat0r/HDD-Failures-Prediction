from flask_restplus import Namespace, fields


#class HDDSMARTPredictionDto:
#    api = Namespace('HDDSMARTPrediction', description='HDDSMARTPrediction related operations')
#    smart_prediction = api.model(
#	    'SMARTPrediction',
#	    {
#		#'timestamp': fields.String(readOnly=True, description='Prediction Timestamp'),
#		'timestamp': fields.DateTime,
#		'model': fields.String(required=True, description='HDD model'),
#		'serial_number': fields.String(required=True, description='HDD serial number'),
#		'capacity_bytes': fields.Integer,
#		'failure': fields.Integer,
#		'prediction': fields.Float
#	    }
#	)

class HDDSMARTPredictionDto:
    api = Namespace('HDDSMARTPrediction', description='HDDSMARTPrediction related operations')
    smart_prediction = api.model(
	    'SMARTPrediction',
            {
                #'timestamp': fields.String(readOnly=True, description='Prediction Timestamp'),
                'timestamp': fields.DateTime,
                'model': fields.String(required=True, description='HDD model', example="XYZ"),
                'serial_number': fields.String(required=True, description='HDD serial number', example="HD00"),
                'capacity_bytes': fields.Integer(example=1000000000),
                'failure': fields.Integer,
                'prediction': fields.Float
            }
	)
    
    prediction_list_example = {
            "prediction_list": [
                {
                    "timestamp": "2020-05-04T18:30:47.588Z",
                    "model": "XYZ",
                    "serial_number": "HD00",
                    "capacity_bytes": 1000000000,
                    "failure": 0,
                    "prediction": 0
                },
                {
                    "timestamp": "2020-05-04T18:30:47.588Z",
                    "model": "UVT",
                    "serial_number": "00DH",
                    "capacity_bytes": 1000000001,
                    "failure": 1,
                    "prediction": 1
                }

            ]
        }
    smart_prediction_list = api.model(
            'SMARTPredictionList',
            {
                'prediction_list': fields.List(fields.Nested(smart_prediction), example=prediction_list_example)
            }
        )


#todo_fields = {
#    "task": fields.String,
#}
#
#todo_list_fields = {
#    "tasks": fields.List(fields.Nested(todo_fields))
#}
#
#class Todo(Resource):
#    @marshal_with(todo_fields)
#    def get(self, todo_id):
#        abort_if_todo_doesnt_exist(todo_id)
#        return TODOS[todo_id]
#
#class TodoList(Resource):
#    @marshal_with(todo_list_fields)
#    def get(self):
#        return TODOS
