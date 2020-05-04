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
                'model': fields.String(required=True, description='HDD model'),
                'serial_number': fields.String(required=True, description='HDD serial number'),
                'capacity_bytes': fields.Integer,
                'failure': fields.Integer,
                'prediction': fields.Float
            }
	)
    
    smart_prediction_list = api.model(
            'SMARTPredictionList',
            {
                'prediction_list': fields.List(fields.Nested(smart_prediction))
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
