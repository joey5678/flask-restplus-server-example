import sys
import os
import base64
import json
import uuid

from cv.utils import b64toPILImg


Storage_Dir = "../img_data"


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

    def save(self, base64_img=None, img_metadata=None):
        if base64_img:
            uid = self.save_image(base64_img)
            if img_metadata:
                self.save_metadata(uid, img_metadata)
        
    
    def save_image(self, base64_img):

        img_format = self.check_format(base64_img[:20])
        if not img_format:
            return False
        
        uid = str(uuid.uuid1())
        # save image...

        image = b64toPILImg(base64_img)
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



img_store_manager = ImageStoreManager()
