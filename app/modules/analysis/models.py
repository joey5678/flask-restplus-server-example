# encoding: utf-8
"""
Team database models
--------------------
"""
from datetime import datetime
from sqlalchemy_utils import Timestamp

from app.extensions import db


class BussModel(db.Model, Timestamp):
    id = db.Column(db.Integer, primary_key=True) 
    source_id = db.Column(db.String(length=64), nullable=False)
    request_type = db.Column(db.Integer, nullable=False)
    sub_type = db.Column(db.Integer, nullable=False)
    content_id = db.Column(db.String(length=64), nullable=False)
    content_type = db.Column(db.Integer, nullable=False)
    content_text = db.Column(db.String(length=256), nullable=False)
    log_id = db.Column(db.String(length=64), nullable=False)
    result = db.Column(db.String(length=2000), nullable=False) #TODO replace with sub tables
    image = db.Column(db.String(length=10000), nullable=False) #TODO replace with image path


class LoginModel(db.Model, Timestamp):
    id = db.Column(db.Integer, primary_key=True) 
    source_str = db.Column(db.String(length=128), nullable=False) # openid
    indntifier = db.Column(db.String(length=128), nullable=False) # phone no
    token = db.Column(db.String(length=128), nullable=True)

class AI_UPIDModel(db.Model):
    __tablename__ = 'AI_UPID'

    id = db.Column(db.Integer, primary_key=True) 
    upid_int = db.Column(db.Integer, nullable=False)
    label = db.Column(db.String(length=128), nullable=False)
    pag = db.Column(db.Integer, nullable=False)
    des_basic = db.Column(db.String(length=1024), nullable=False)
    des_positive = db.Column(db.String(length=1024), nullable=False)
    des_negative = db.Column(db.String(length=1024), nullable=False)
    des_suggest = db.Column(db.String(length=1024), nullable=False)
    des_child_16 = db.Column(db.String(length=1024), nullable=False)


class AI_ConstantModel(db.Model):
    __tablename__ = 'AI_Constant'

    id = db.Column(db.Integer, primary_key=True) 
    const_name = db.Column(db.String(length=64), nullable=False)
    const_value_male = db.Column(db.DECIMAL(10,3), nullable=False)
    const_value_female = db.Column(db.DECIMAL(10,3), nullable=False)
    const_child_16 = db.Column(db.DECIMAL(10,3), nullable=False)
    sigma = db.Column(db.DECIMAL(10,3), nullable=False)


class AI_Constant_APPModel(db.Model):
    __tablename__ = 'AI_Constant_APP'

    id = db.Column(db.Integer, primary_key=True) 
    const_id = db.Column(db.Integer, nullable=False) 
    const_name = db.Column(db.String(length=64), nullable=False)
    const_value_male = db.Column(db.DECIMAL(10,3), nullable=False)
    const_value_female = db.Column(db.DECIMAL(10,3), nullable=False)
    const_child_16 = db.Column(db.DECIMAL(10,3), nullable=False)
    const_min = db.Column(db.DECIMAL(10,3), nullable=False)
    const_max = db.Column(db.DECIMAL(10,3), nullable=False)
    range_min = db.Column(db.DECIMAL(10,3), nullable=False)
    range_max = db.Column(db.DECIMAL(10,3), nullable=False)
    const_sigma = db.Column(db.DECIMAL(10,3), nullable=False)
    const_description = db.Column(db.String(length=1024), nullable=False)


class AI_USER_INFOModel(db.Model, Timestamp):
    __tablename__ = 'AI_USER_INFO'

    id = db.Column(db.Integer, primary_key=True) 
    cuid = db.Column(db.Integer, nullable=False)		
    age = db.Column(db.Integer, nullable=False)        
    gender = db.Column(db.Integer, nullable=False)
    phone =	 db.Column(db.String(length=64), nullable=False)
    region =  db.Column(db.String(length=128), nullable=False)    
    orignal_cuid = db.Column(db.String(length=128), nullable=True)    
    client_id = db.Column(db.String(length=128), nullable=False)    
    reg_time = db.Column(db.DateTime, nullable=False)


class AI_USER_REP(db.Model, Timestamp):
    __tablename__ = 'AI_USER_REP'

    id = db.Column(db.Integer, primary_key=True) 
    log_id = db.Column(db.String(length=128), nullable=False)
    cuid = db.Column(db.Integer, nullable=False)
    analysis_rep = db.Column(db.Text, nullable=False)
    log_time = db.Column(db.DateTime, nullable=True)


class AI_USER_APPKEYModel(db.Model, Timestamp):
    __tablename__ = 'AI_USER_APPKEY'

    id = db.Column(db.Integer, primary_key=True) 
    log_id = db.Column(db.String(length=128), nullable=False)
    cuid = db.Column(db.Integer, nullable=False)
    page_type = db.Column(db.Integer, nullable=False)
    tree_height = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_width = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_area = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_on_left = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_on_right = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_on_top = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_on_bottom  = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_on_page = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_height = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_width = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area_1 = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area_2 = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area_3 = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area_4 = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area_5 = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_crown_area_6 = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_trunk_height = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_trunk_width = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_trunk_area = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_root_height = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_root_width = db.Column(db.DECIMAL(10,3), nullable=False)
    tree_root_area = db.Column(db.DECIMAL(10,3), nullable=False)
    xinzhi_value = db.Column(db.DECIMAL(10,3), nullable=False)
    siwei_value = db.Column(db.DECIMAL(10,3), nullable=False)
    qingxu_value = db.Column(db.DECIMAL(10,3), nullable=False)
    juese_value = db.Column(db.DECIMAL(10,3), nullable=False)
    nengli_value = db.Column(db.DECIMAL(10,3), nullable=False)
    qianyishi_value = db.Column(db.DECIMAL(10,3), nullable=False)
    log_time = db.Column(db.DateTime, nullable=True)	



class CharacterAnalysis(db.Model, Timestamp):


    __tablename__ = 'character_analysis'

    id = db.Column(db.Integer, primary_key=True) 
    cuid = db.Column(db.String(length=64), nullable=False)
    log_id = db.Column(db.String(length=64), nullable=False)
    image = db.Column(db.String(length=9000000), nullable=False) #TODO replace with image path
    married = db.Column(db.Integer, nullable=False)
    request_type = db.Column(db.Integer, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    address = db.Column(db.String(length=256), nullable=False)
    result = db.Column(db.String(length=2000), nullable=False) #TODO replace with sub tables

    def __repr__(self):
        return (
            "<{class_name}("
            "log_id={self.log_id}, "
            "result=\"{self.result}\", "
            ")>".format(
                class_name=self.__class__.__name__,
                self=self
            )
        )

    @db.validates
    def validate_age(self, key, age):
        if age > 120 or age <= 0:
            raise ValueError("Age has to be no less than 0 and no more than 120.")
        return age


class AnalysisReport(db.Model, Timestamp):

    __tablename__ = 'analysis_report'

    id = db.Column(db.Integer, primary_key=True) 
    cuid = db.Column(db.String(length=64), nullable=True)
    log_id = db.Column(db.String(length=64), nullable=False)
    request_type = db.Column(db.Integer, nullable=False)
    result = db.Column(db.String(length=2000), nullable=False) #TODO replace with sub tables


    def __repr__(self):
        return (
            "<{class_name}("
            "log_id={self.log_id}, "
            "result=\"{self.result}\", "
            ")>".format(
                class_name=self.__class__.__name__,
                self=self
            )
        )


class MindMatch(db.Model, Timestamp):
    
    __tablename__ = "mind_match"

    id = db.Column(db.Integer, primary_key=True) 
    cuid = db.Column(db.String(length=64), nullable=True)
    log_id = db.Column(db.String(length=64), nullable=False)
    request_type = db.Column(db.Integer, nullable=False)
    match_type = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    address = db.Column(db.String(length=128), nullable=False)
    result = db.Column(db.String(length=1024), nullable=False)

    def __repr__(self):
        return (
            "<{class_name}("
            "log_id={self.log_id}, "
            "result=\"{self.result}\", "
            ")>".format(
                class_name=self.__class__.__name__,
                self=self
            )
        )
    
class MindSpecMatch(db.Model, Timestamp):

    __tablename__ = "mind_spec_match"

    id = db.Column(db.Integer, primary_key=True) 
    cuid = db.Column(db.String(length=64), nullable=True)
    log_id = db.Column(db.String(length=64), nullable=False)
    request_type = db.Column(db.Integer, nullable=False)
    match_type = db.Column(db.Integer, nullable=False)
    other_id = db.Column(db.String(length=64), nullable=True)
    result = db.Column(db.String(length=1024), nullable=False)

    def __repr__(self):
        return (
            "<{class_name}("
            "log_id={self.log_id}, "
            "result=\"{self.result}\", "
            ")>".format(
                class_name=self.__class__.__name__,
                self=self
            )
        )


