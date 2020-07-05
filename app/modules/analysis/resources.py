# encoding: utf-8
# pylint: disable=bad-continuation
"""
RESTful API Team resources
--------------------------
"""

import logging
import uuid
from datetime import datetime

from flask import Flask, redirect, url_for
from flask_login import current_user
from flask_restplus_patched import Resource
from flask_restplus import fields
from flask_restplus._http import HTTPStatus

from app.library import jwt
from app.extensions import db
from app.extensions.api import Namespace, abort
from app.modules.users import permissions
# from app.modules.users.models import User


from . import parameters, schemas
from .models import CharacterAnalysis as CA_Model
from .models import AnalysisReport as AR_Model
from .models import MindMatch as MCM_Model
from .models import MindSpecMatch as MSM_Model

from .models import AI_USER_INFOModel as User_Model


from mind.manager import img_store_manager
from cv.align import get_points, warpImage
from cv.utils import *
from mock import object as mock_obj


log = logging.getLogger(__name__)  # pylint: disable=invalid-name
api = Namespace('analysis', description="analysis")  # pylint: disable=invalid-name

# bussmodel = api.model('BussModel', {
#     'source_id': fields.Integer(required=True, description='The task unique identifier'),
#     'request_type': fields.String(required=True, description='The task details'),
#     'image': fields.String(required=True, description='The task details'),
#     'sub_type': fields.String(required=True, description='The task details'),
#     'content_id': fields.String(required=True, description='The task details'),
#     'content_type': fields.String(required=True, description='The task details'),
#     'content_text': fields.String(required=True, description='The task details'),
# })

login_dict = {

}

Fields_mapping = {
    10: {
        'source_id': 'cuid',
        'sub_type' : 'married',
        'content_id': 'age',
        'content_type': 'gender',
        'content_text': 'address',
    },
    11: {
        'source_id': 'cuid',
        'sub_type' : 'married',
        'content_id': 'age',
        'content_type': 'gender',
        'content_text': 'address',
    },
    12:{
        'content_id': 'log_id',
        'source_id': '',
        'sub_type' : '',
        'content_type': '',
        'content_text': '',
        'image': '',
    },
    20:{
        "sub_type": "match_type",
        "content_id": "log_id",
        "content_type": "gender",
        'content_text': 'address',
        'source_id': '',
        'image': '',
    },
    21:{
        'source_id': '',
        'image': '',
        "sub_type": "match_type",
        "content_id": "log_id",
        'content_type': '',
        'content_text': 'other_id',
    }

}

def transfer_fields(args):
    fmap = Fields_mapping[args['request_type']]
    for org_key in fmap.keys():
        tgt_key = fmap[org_key]
        if tgt_key:
            args[tgt_key] = args[org_key]
    for org_key in fmap.keys():
        args.pop(org_key, None)
    
    return args

def create_cuid(key=None):
    return str(uuid.uuid1())

def handle_image(image_id):
    cv_img = img_store_manager.get_OCV_image(image_id)
    if cv_img is None:
        log.error(f"can not find the image with id: {image_id} in backend.")
        abort(
            code=HTTPStatus.NOT_FOUND,
            message=f"can not find the image with id: {image_id} in backend."
        )

    log.debug(f"after decoding, the size of the image is {cv_img.shape[:2]}")
    w, h = cv_img.shape[0], cv_img.shape[1]
    align_img = warpImage(cv_img, get_points(cv_img))
    if align_img.shape[:2] != (w, h):
        align_img = resize(align_img, w, h)
#   save the aligned image. 
    uid = img_store_manager.save_opencv_img(align_img, img_id=image_id)
    return uid

def handle_user():
    user_info = jwt.decode()['identity']
    phone = user_info['identifier']
    # need to get user data from DB by phone, 
    # if not exist, create new one with cuid.
    user = User_Model.query.filter_by(phone=phone).first()
    if not user:
        # add one 
        with api.commit_or_abort(
                db.session,
                default_error_message="Failed to create a user"
            ):
            user = User_Model(cuid=tgt_args['cuid'], 
                            age=tgt_args['age'], 
                            gender=tgt_args['gender'],
                            phone=phone,
                            region=tgt_args['address'],
                            client_id=user_info['req_source'],
                            reg_time=datetime.utcnow()
                          )
        
            db.session.add(user)

    else:
        print("get user info ....")
        print(user)
        # update the other field? 
    
    return user


def check_args(args):
    married = args['sub_type']
    age_str = args['content_id']
    gender = args['content_type']
    try:
        age = int(age_str)
    except:
        age = 0

    if married not in (0, 1) or gender not in (0, 1) or 0 >= age or age > 150:
        abort(
            code=HTTPStatus.BAD_REQUEST,
            message=f"some fields have wrong values. [sub_type: (0, 1), content_type: (0, 1), content_id: (0 ~ 150)]"
        )


def do_character_analysis(args):
    check_args(args)
    tgt_args = transfer_fields(args)
    image_id = args.get('image', None)
    assert image_id is not None, "receive none image_id in character analysis"
    uid = handle_image(image_id)

    assert jwt.decode().get('identity', None) is not None
    user = handle_user()

    # real analysis logic here
    # 1. call AI API to get objects in image
    # 2. calculate the locations and area info
    # 3. evaluation the mind scores based on the above info
    # 4. genearte the result
    log_id = create_cuid()
    login_dict[log_id] = user

    c_analysis = CA_Model(**tgt_args)
    c_analysis.log_id = log_id
    c_analysis.result = mock_obj.emit_ca_result()

    return c_analysis

def do_analysis_report(args):
    tgt_args = transfer_fields(args)
    a_report = AR_Model(**tgt_args)
    a_report.result = mock_obj.emit_report()
        
    return a_report

def do_teen_analysis(args):
    check_args(args)
    tgt_args = transfer_fields(args)

    image_id = args.get('image', None)
    assert image_id is not None, "receive none image_id in character analysis"
    uid = handle_image(image_id)

    assert jwt.decode().get('identity', None) is not None
    user = handle_user()

    # real analysis logic here
    # 1. call AI API to get objects in image
    # 2. calculate the locations and area info
    # 3. evaluation the mind scores based on the above info
    # 4. genearte the result
    log_id = create_cuid()
    login_dict[log_id] = user

    c_analysis = CA_Model(**tgt_args)
    c_analysis.log_id = log_id
    c_analysis.result = mock_obj.emit_ca_result()

    ar_args = {'request_type': 12, 'log_id': c_analysis.log_id}
    a_report = AR_Model(**ar_args)
    a_report.result = mock_obj.emit_report()

    c_analysis.result += a_report.result
    return c_analysis

def do_mind_common_match(args):
    if args['sub_type'] not in (100, 101):
        abort(
                code=HTTPStatus.BAD_REQUEST,
                message="sub_type value is not in (100, 101)"
            )
    tgt_args = transfer_fields(args)
    log_id = tgt_args['log_id']
    user = login_dict.get(log_id, None)
    assert user is not None
    mc_match = MCM_Model(**tgt_args)
    mc_match.result = [{
        "matched_id": "235705179287220",
        "score": 0.75
        },
        {
        "matched_id": "8897623456098702",
        "score": 0.63
        }]
    return mc_match

def do_mind_spec_match(args):

    sub_type = args['sub_type']
    if sub_type not in (100, 101, 102):
        abort(
                code=HTTPStatus.BAD_REQUEST,
                message="sub_type value is not in (100, 101, 102)"
            )
    
    tgt_args = transfer_fields(args)

    ms_match = MSM_Model(**tgt_args)

    if sub_type == 100:
        ms_match.result = mock_obj.emit_100_result()
        
    elif sub_type == 101:
        ms_match.result = mock_obj.emit_101_result()
        
    else:
        ms_match.result = mock_obj.emit_102_result()

    return ms_match


@api.route('/')
class Business(Resource):

    @api.parameters(parameters.CreateBussParameters())
    @api.response(schemas.BussResultSchema())
    @jwt.required
    def post(self, args):
        print(f"-------------------request type is {args['request_type']}")
        if args['request_type'] == 10:
            return do_character_analysis(args)
        elif args['request_type'] == 12:
            return do_analysis_report(args)
        elif args['request_type'] == 11:
            return do_teen_analysis(args)
        elif args['request_type'] == 20:
            return do_mind_common_match(args)
        elif args['request_type'] == 21:
            return do_mind_spec_match(args)


