import logging

from flask import request, jsonify
from flask_restplus_patched import Resource
from flask_jwt_extended import jwt_refresh_token_required

from app import response
from app.extensions.api import Namespace
from app.library import jwt

from .models import Users

log = logging.getLogger(__name__)
api = Namespace('auth', description="Authentication")

@api.route('/')
class Login(Resource):
    def get(self):
        try:
            token = jwt.decode()
            return response.ok(token, '')
        except Exception as e:
            print(e)

    def post(self):
        try:
            identifier = request.json['phone']
            req_source = request.json['app_id']

            user = None #Users.query.filter_by(identifier=identifier).first()
            if not user:
                user = Users()
                user.identifier = identifier
                user.req_source = req_source

                # return response.badRequest([], 'Empty....')

            # if not user.checkPassword(password):
            #     return response.badRequest([], 'Your credentials is invalid')

            data = singleTransform(user, withTodo=False)
            access_token = jwt.encode(data)
            refresh_token = jwt.encode(data, access=False)

            return response.ok({
                # 'data': data,
                'access_token': access_token,
                'refresh_token': refresh_token,
            }, "")
        except Exception as e:
            return response.badRequest('', e)

@api.route('/refresh')
class Refresh(Resource):
    @jwt_refresh_token_required
    def post(self):
        try:
            current_user = jwt.get_jwt_identity()
            new_token = jwt.encode(current_user)
        except Exception as e:
            error_info = str(e)
            print("refresh failed with error {error_info}")
            ret = {
                "access_token": "" 
            }
            return response.unAuthorized(ret, error_info)

        ret = {
            'access_token': new_token#create_access_token(identity=current_user)
        }
        return response.ok(ret, "")


def singleTransform(users, withTodo=True):
    data = {
        'id': users.id,
        'identifier': users.identifier,
        'req_source': users.req_source,
    }
