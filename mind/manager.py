import sys
import os
import base64
import json
import uuid

from cv.utils import b64toPILImg, save_ocv_image, rawtoOCVImg, rawtoPILImg


Storage_Dir = "img_data"


"""
Save the image and meta-data
"""
class ImageStoreManager():

    def __init__(self, saved_dir=Storage_Dir):
        self.saved_dir = saved_dir
        self.support_formats = ('jpg', 'jpeg', 'png')

    def check_format(self, feature_str):
        return "jpg"
        # return None
    
    def save_opencv_img(self, image, img_id=None, name_prefix=None, format='png'):
        uid = img_id if img_id is not None else str(uuid.uuid1())
        aligned_tag = "" if img_id is None else "aligned_"
        tag = name_prefix.strip() if name_prefix is not None else aligned_tag
        tag = f"{tag}_" if not tag.endswith("_") else tag
        img_name = f"{tag}sv_image_{uid}.{format}"
        save_ocv_image(image, os.path.join(self.saved_dir, img_name))
        return uid


    def saveb64(self, base64_img=None, img_metadata=None):
        if base64_img:
            uid = self.save_b64image(base64_img)
            if img_metadata:
                self.save_metadata(uid, img_metadata)


    def save(self, img=None, img_metadata=None):
        if img:
            uid = self.save_image(base64_img)
            if img_metadata:
                self.save_metadata(uid, img_metadata)

    
    def save_b64image(self, base64_img):

        img_format = self.check_format(base64_img[:20])
        if not img_format:
            return False
        
        uid = str(uuid.uuid1())
        # save image...

        image = b64toPILImg(base64_img)
        img_name = f"sv_image_{uid}.{img_format}"
        image.save(os.path.join(self.saved_dir, img_name))

        return uid

    def save_image(self, image, img_format='png'):

        uid = str(uuid.uuid1())
        # save image...
        img_name = f"sv_image_{uid}.{img_format}"
        image.save(os.path.join(self.saved_dir, img_name))

        return uid
    
    def save_metadata(self, uid, img_metadata):

        js_file = f"sv_meta_{uid}.json"
        try:
            with open(os.path.join(self.saved_dir, js_file), 'w') as f:
                json.dump(img_metadata, f)
        except:
            print("Fail to write metadata into json file ")

    def get_image(self, image_id, img_format='png'):
        data = None
        img_name = f"sv_image_{image_id}.{img_format}"
        if os.path.isfile(os.path.join(self.saved_dir, img_name)):
            with open(os.path.join(self.saved_dir, img_name), 'rb') as f:
                data = f.read()
        
        return data

    def get_OCV_image(self, image_id, img_format='png'):
        data = self.get_image(image_id, img_format)

        return None if data is None else rawtoOCVImg(data)

    def get_PIL_image(self, image_id, img_format='png'):
        data = self.get_image(image_id, img_format)

        return None if data is None else rawtoPILImg(data)

img_store_manager = ImageStoreManager()
