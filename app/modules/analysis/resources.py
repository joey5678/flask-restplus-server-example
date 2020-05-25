# encoding: utf-8
# pylint: disable=bad-continuation
"""
RESTful API Team resources
--------------------------
"""

import logging

from flask import redirect, url_for
from flask_login import current_user
from flask_restplus_patched import Resource
from flask_restplus import fields
from flask_restplus._http import HTTPStatus

from app.extensions import db
from app.extensions.api import Namespace, abort
from app.modules.users import permissions
from app.modules.users.models import User


from . import parameters, schemas
from .models import CharacterAnalysis as CA_Model
from .models import AnalysisReport as AR_Model


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
    12:{
        'content_id': 'log_id',
        'source_id': '',
        'sub_type' : '',
        'content_type': '',
        'content_text': '',
        'image': '',
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

@api.route('/')
class Business(Resource):

    @api.parameters(parameters.CreateBussParameters())
    @api.response(schemas.BussResultSchema())
    def post(self, args):

        if args['request_type'] == 10:
            print("go to character analysis api..")
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
            return c_analysis
        elif args['request_type'] == 12:
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

            


@api.route('/analy')
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
        
@api.route('/report')
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
