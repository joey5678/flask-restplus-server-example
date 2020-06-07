# encoding: utf-8
"""
API extension
=============
"""

from copy import deepcopy

from flask import current_app

from .api import Api
from .namespace import Namespace
from .http_exceptions import abort


api_v1 = Api( # pylint: disable=invalid-name
    version='1.0',
    title="Mind Map Web API",
    description=(
        "It is a [real-life example RESTful API server implementation using Flask-RESTplus]"
        "APIs: \n\n"
        "* /auth/ : \n"
        "-\t method: POST \n"
        '-\t params(json):{"phone":"xxxx", "app_id":"xxx"} \n'
        '-\t response": "message": "",  "values": {"access_token": "xxxx", "refresh_token": "xxxxxx"} \n\n'

        "* /auth/refresh : \n"
        "-\t method: POST \n"
        '-\t Authorization: Bearer Token \n'
        '-\t response": "message": "",  "values": {"access_token": "xxxx"} \n\n'

        "* /analysis/ : \n"
        "-\t method: POST \n"
        '-\t Authorization: Bearer Token \n'
        '-\t params(json):{"request_type": int, "source_id":"str", '
        '"sub_type": int, "content_id":"str", "content_type":int,"content_text":"str","image":"str:base64"} \n'
        '-\t response": "message": "",  "values": {"log_id": "xxxx", "result": "xxxxxx"} \n\n'
    ),
)


def serve_swaggerui_assets(path):
    """
    Swagger-UI assets serving route.
    """
    if not current_app.debug:
        import warnings
        warnings.warn(
            "/swaggerui/ is recommended to be served by public-facing server (e.g. NGINX)"
        )
    from flask import send_from_directory
    return send_from_directory('../static/', path)


def init_app(app, **kwargs):
    # pylint: disable=unused-argument
    """
    API extension initialization point.
    """
    app.route('/swaggerui/<path:path>')(serve_swaggerui_assets)

    # Prevent config variable modification with runtime changes
    api_v1.authorizations = deepcopy(app.config['AUTHORIZATIONS'])
