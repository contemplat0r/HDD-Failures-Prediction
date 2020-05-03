import datetime
from flask import request
from flask_restplus import Resource

from ..util.dto import HDDSMARTPredictionDto
from ..service.failures_service import save_new_hdd_smart_predictions, get_all_failure_predictions

api = HDDSMARTPredictionDto.api
_hdd_smart_prediction = HDDSMARTPredictionDto.smart_prediction


@api.route('/')
class HDDSMARTPredictionList(Resource):
    @api.doc('hdd_smart_predictions_list')
    @api.marshal_list_with(_hdd_smart_prediction, envelope='data')
    def get(self):
        """List all failure predictions"""
        return get_all_failure_predictions()

    @api.response(201, "failure predictions successfully saved.")
    @api.doc('save new predictions')
    #@api.expect(_hdd_smart_prediction, validate=True)
    def post(self):
        """Save a new predictions """
        timestamp = datetime.datetime.utcnow()
        data = request.json
        for single_prediction in data:
            single_prediction['timestamp'] = timestamp
        return save_new_hdd_smart_predictions(data)


#@api.route('/<public_id>')
#@api.param('public_id', 'The User identifier')
#@api.response(404, 'User not found.')
#class User(Resource):
#    @api.doc('get a user')
#    @api.marshal_with(_user)
#    def get(self, public_id):
#        """get a user given its identifier"""
#        user = get_a_user(public_id)
#        if not user:
#            api.abort(404)
#        else:
#            return user
