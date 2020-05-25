# encoding: utf-8
"""
Team database models
--------------------
"""

from sqlalchemy_utils import Timestamp

from app.extensions import db


class BussModel(db.Model, Timestamp):
    id = db.Column(db.Integer, primary_key=True) 
    source_id = db.Column(db.String(length=64), nullable=False)
    request_type = db.Column(db.Integer, nullable=False)
    sub_type = db.Column(db.Integer, nullable=False)
    content_id = db.Column(db.String(length=64), nullable=False)
    content_type = sub_type = db.Column(db.Integer, nullable=False)
    content_text = db.Column(db.String(length=256), nullable=False)
    log_id = db.Column(db.String(length=64), nullable=False)
    result = db.Column(db.String(length=2000), nullable=False) #TODO replace with sub tables
    image = db.Column(db.String(length=2000), nullable=False) #TODO replace with image path


class CharacterAnalysis(db.Model, Timestamp):
    """
    个性分析 db model
    """

    __tablename__ = 'character_analysis'

    id = db.Column(db.Integer, primary_key=True) 
    cuid = db.Column(db.String(length=64), nullable=False)
    log_id = db.Column(db.String(length=64), nullable=False)
    image = db.Column(db.String(length=2000), nullable=False) #TODO replace with image path
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
    """
    分析报告 db model
    """
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
    
