from app.extensions.api import api_v1


def init_app(app, **kwargs):
    # pylint: disable=unused-argument,unused-import
    """
    Init Image module.
    """
    api_v1.add_oauth_scope('image:read', "Provide access to get image data")
    api_v1.add_oauth_scope('image:write', "Provide write access to upload image")

    # Touch underlying modules
    from . import resources

    api_v1.add_namespace(resources.api)