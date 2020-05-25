# encoding: utf-8
"""
Serialization schemas for Team resources RESTful API
----------------------------------------------------
"""

from flask_marshmallow import base_fields
from flask_restplus_patched import ModelSchema

# from app.modules.users.schemas import BaseUserSchema

from .models import CharacterAnalysis, AnalysisReport, BussModel


class BussSchema(ModelSchema):
    class Meta:
        # pylint: disable=missing-docstring
        model = BussModel
        fields = (
            BussModel.source_id.key,
            BussModel.request_type.key,
            BussModel.sub_type.key,
            BussModel.content_id.key,
            BussModel.content_type.key,
            BussModel.content_text.key,
            BussModel.image.key,
        )

class BussResultSchema(ModelSchema):

    class Meta:
        model = BussModel
        fields = (
            BussModel.log_id.key,
            BussModel.result.key,
        )


class BaseAnalysisSchema(ModelSchema):
    """
    Base team schema exposes only the most general fields.
    """

    class Meta:
        # pylint: disable=missing-docstring
        model = CharacterAnalysis
        fields = (
            CharacterAnalysis.id.key,
            CharacterAnalysis.cuid.key,
            CharacterAnalysis.request_type.key,
            CharacterAnalysis.married.key,
            CharacterAnalysis.age.key,
            CharacterAnalysis.gender.key,
            CharacterAnalysis.image.key,
        )
        dump_only = (
            CharacterAnalysis.id.key,
        )

class CharacterAnalysisResultSchema(ModelSchema):

    class Meta:
        model = CharacterAnalysis
        fields = (
            CharacterAnalysis.log_id.key,
            CharacterAnalysis.result.key,
 
        )


class BaseReportSchema(ModelSchema):
    """
    Base team schema exposes only the most general fields.
    """

    class Meta:
        # pylint: disable=missing-docstring
        model = AnalysisReport
        fields = (
            AnalysisReport.id.key,
            AnalysisReport.request_type.key,
            AnalysisReport.log_id.key,
        )
        dump_only = (
            AnalysisReport.id.key,
        )

class AnalysisReportResultSchema(ModelSchema):

    class Meta:
        model = AnalysisReport
        fields = (
            AnalysisReport.log_id.key,
            AnalysisReport.result.key,
        )
