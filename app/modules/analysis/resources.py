# encoding: utf-8
# pylint: disable=bad-continuation
"""
RESTful API Team resources
--------------------------
"""

import logging

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



from mind.manager import img_store_manager
from cv.align import get_points, warpImage
from cv.utils import *


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


def do_character_analysis(args):
    tgt_args = transfer_fields(args)
    img_data = args.get('image', None)
    assert img_data is not None, "receive none image in character analysis"
    cv_img = b64toOCVImg(img_data)
    log.debug(f"received b64img. lenght: {len(img_data)}.")
    log.debug(f"after decoding, the size of the image is {cv_img.shape[:2]}")
    w, h = cv_img.shape[0], cv_img.shape[1]
    align_img = warpImage(cv_img, get_points(cv_img))
    if align_img.shape[:2] != (w, h):
        align_img = resize(align_img, w, h)
#   save the aligned image. 
    uid = img_store_manager.save_opencv_img(align_img)
    tgt_args['image'] = uid
    c_analysis = CA_Model(**tgt_args)
    c_analysis.log_id = "test_LOG_ID_0000001"
    c_analysis.result = [{
        "elements":"思维成熟度",
        "value":81,
        "average":62,
        "range":"58-85"
        },
        {
        "elements":"心智成熟度",
        "value": 71,
        "average": 59.3,
        "range":"50-80"
        }]
    return c_analysis

def do_analysis_report(args):
    tgt_args = transfer_fields(args)
    a_report = AR_Model(**tgt_args)
    a_report.result =  [
        {
        "elements": "自我认知",
        "summary": "思维成熟度，指的是看待事物有自己独立的思考，不会人云亦云。或者说拥有自己稳定的价值体系，不会轻易受他人和周围环境影响。",
        "description": "你的思维体系比较成熟. 在生活中你对人对事有自己独立的想法和判断，能够形成自己的意见，从而在生活中掌握主动权。从深层意义而言，这种思维的成熟意味着整合的价值体系。"
        },
        {
        "elements": "理性与感性",
        "summary": "一个人的心智成熟度，表现在他是如何看待自我的。他表现出来的状态与内在世界你保持真实的一致性，他们心智就达到成熟的状态。",
        "description": "你的心智比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        }]
        
    return a_report

def do_teen_analysis(args):
    tgt_args = transfer_fields(args)
    c_analysis = CA_Model(**tgt_args)
    c_analysis.log_id = "test_LOG_ID_0000001"
    c_analysis.result = [{
        "elements":"思维成熟度",
        "value":81,
        "average":62,
        "range":"58-85"
        },
        {
        "elements":"心智成熟度",
        "value": 71,
        "average": 59.3,
        "range":"50-80"
        }]
    ar_args = {'request_type': 12, 'log_id': c_analysis.log_id}
    a_report = AR_Model(**ar_args)
    a_report.result =  [
        {
        "elements": "自我认知",
        "summary": "思维成熟度，指的是看待事物有自己独立的思考，不会人云亦云。或者说拥有自己稳定的价值体系，不会轻易受他人和周围环境影响。",
        "description": "你的思维体系比较成熟. 在生活中你对人对事有自己独立的想法和判断，能够形成自己的意见，从而在生活中掌握主动权。从深层意义而言，这种思维的成熟意味着整合的价值体系。"
        },
        {
        "elements": "理性与感性",
        "summary": "一个人的心智成熟度，表现在他是如何看待自我的。他表现出来的状态与内在世界你保持真实的一致性，他们心智就达到成熟的状态。",
        "description": "你的心智比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        }]
    
    c_analysis.result += a_report.result
    return c_analysis

def do_mind_common_match(args):
    tgt_args = transfer_fields(args)
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
    tgt_args = transfer_fields(args)
    ms_match = MSM_Model(**tgt_args)
    ms_match.result = [{
        "elements": "性格匹配度",
        "source_value": 81,
        "target_value": 62
        },
        {
        "elements": "思想匹配度",
        "source_value": 55,
        "targer_value": 30
        },
        {
        "elements": "情感度",
        "source_value": 71,
        "targer_value": 59.3
        }]

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

"""
class Analysis(Resource):

    @api.parameters(parameters.CreateAnalysisParameters())
    @api.response(schemas.CharacterAnalysisResultSchema())
    def post(self, args):
        print(args)
        request_type = (args['request_type'])

        c_analysis = CA_Model(**args)
        c_analysis.log_id = "test_LOG_ID_0000001"
        c_analysis.result = [{
            "elements":"思维成熟度",
            "value":81,
            "average":62,
            "range":"58-85"
            },
            {
            "elements":"心智成熟度",
            "value": 71,
            "average": 59.3,
            "range":"50-80"
            }]
        return c_analysis
        

class Report(Resource):

    @api.parameters(parameters.CreateReportParameters())
    @api.response(schemas.AnalysisReportResultSchema())
    def post(self, args):
        print(args)
        a_report = AR_Model(**args)
        a_report.result =  [
            {
            "elements": "自我认知",
            "summary": "思维成熟度，指的是看待事物有自己独立的思考，不会人云亦云。或者说拥有自己稳定的价值体系，不会轻易受他人和周围环境影响。",
            "description": "你的思维体系比较成熟. 在生活中你对人对事有自己独立的想法和判断，能够形成自己的意见，从而在生活中掌握主动权。从深层意义而言，这种思维的成熟意味着整合的价值体系。"
            },
            {
            "elements": "理性与感性",
            "summary": "一个人的心智成熟度，表现在他是如何看待自我的。他表现出来的状态与内在世界你保持真实的一致性，他们心智就达到成熟的状态。",
            "description": "你的心智比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
            }]
            
        return a_report
"""