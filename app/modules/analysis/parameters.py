# encoding: utf-8
"""
Input arguments (Parameters) for Team resources RESTful API
-----------------------------------------------------------
"""

from flask_marshmallow import base_fields
from flask_restplus_patched import Parameters, PostFormParameters, PatchJSONParameters

from . import schemas


"""
class CreateAnalysisParameters(PostFormParameters, schemas.BaseAnalysisSchema):

    class Meta(schemas.BaseAnalysisSchema.Meta):
        pass


class CreateReportParameters(PostFormParameters, schemas.BaseReportSchema):

    class Meta(schemas.BaseReportSchema.Meta):
        pass

"""

class CreateBussParameters(Parameters, schemas.BussSchema):

    class Meta(schemas.BussSchema.Meta):
        pass

