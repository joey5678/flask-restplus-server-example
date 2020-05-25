from app.extensions.api import api_v1


def init_app(app, **kwargs):
    # pylint: disable=unused-argument,unused-import
    """
    Init Analysis module.
    """
    api_v1.add_oauth_scope('analysis:read', "Provide access to analysis result details")
    api_v1.add_oauth_scope('analysis:write', "Provide write access to analysis result details")

    # Touch underlying modules
    from . import models, resources

    api_v1.add_namespace(resources.api)