import datetime
from functools import wraps
from flask import request
from app import response
from flask_jwt_extended import (create_access_token, create_refresh_token,
                                get_jwt_identity, decode_token, get_current_user)


def required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            decode()
        except Exception as e:
            return response.unAuthorized('', 'Unauthorized!')
        return fn(*args, **kwargs)

    return wrapper


def encode(array, access=True):
    if access:
        return create_access_token(array, fresh=True)
    else:
        return create_refresh_token(array)


def decode():
    authorization = request.headers.get('Authorization')
    string = authorization.split(' ')
    decoded = decode_token(string[1])
    return decoded


def getCurrentUser():
    return get_current_user()


def getIdentity():
    return get_jwt_identity()
