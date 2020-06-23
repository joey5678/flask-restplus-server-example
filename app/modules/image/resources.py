
import logging
import io

from flask import request, jsonify, make_response, Response
from flask_restplus_patched import Resource
from app.extensions.api import Namespace, abort

from app.library import jwt
from mind.manager import img_store_manager

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)  # pylint: disable=invalid-name
api = Namespace('image', description="Image")  # pylint: disable=invalid-name


@api.route("/upload")
class ImgUpload(Resource):

    @jwt.required
    def post(self):
        try:
            file = request.files['image']
            img_bytes = file.read()
            image_pil = Image.open(io.BytesIO(img_bytes))
            uid = img_store_manager.save_image(image_pil)
        except Exception as e:
            res = {
                "message": str(e),
                "image_name": ""
            }
            scode = 500
        else:
            res = {
                "message": "upload success.",
                "image_name": uid
            }
            scode = 200

        response = jsonify(res)
        response.status_code = scode
        return response


@api.route("/<id>")
class ImageStore(Resource):
    def get(self, id):
        img_data = img_store_manager.get_image(id)
        if img_data:
            resp = Response(img_data, mimetype="image/png")
            return resp
        else:
            res = {
                "retcode": 503,
                "message": "can not find the image"
            }
            response = jsonify(res)
            response.status_code = 503
            return response

