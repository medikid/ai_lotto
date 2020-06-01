
from sqlalchemy import Column, Integer, Numeric, String, Text, VARCHAR, DECIMAL, DateTime, Float, Boolean, LargeBinary, Binary, SmallInteger, BigInteger
from sqlalchemy import select, delete, update, insert
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy import Integer, DateTime
from wsgiref.handlers import format_date_time

# from db import db_init
# from base import Base
# from db_base import DBBase
# from utils import Utils

from db import db_init
from db.base import Base
from db.db_base import DBBase
from db.utils import Utils

import numpy as np
from datetime import datetime

class TrainingLog(Base, DBBase):
    __tablename__ = 'train_logs'
    __table_args__ = {'extend_existing': True}
    db = db_init('SQLITE')
    
    train_log_id=Column(Integer, primary_key=True, autoincrement=True)
    train_sess_id=Column(BigInteger)
    modset_id = Column(VARCHAR(50))
    datetime=Column(DateTime)
    epoch =Column(Integer)
    loss = Column(DECIMAL(35,30))
    metric_name = Column(VARCHAR(50))
    metric_value = Column(DECIMAL(35,30))
    
    
    def __init__(self, modsetID, trainSessID=0, trainLogID=0):
        self.modset_id = modsetID;
        self.train_sess_id = trainSessID;        
        if(trainLogID>0):
            self.train_log_id = trainLogID;

        super().setupDBBase(TrainingLog, TrainingLog.train_log_id, self.train_log_id)
        
    def reset(self):
        self.train_log_id=0
        self.epoch=0;
        self.loss=0;
        self.metric_name = '';
        self.metric_value=0;
        
    def create_table(self):
        self.metadata.create_all(self.db.db_engine)
        
    def set_log(self, Epoch, Loss, metricName, metricValue):
        self.epoch = Epoch;
        self.loss = Loss;
        self.metric_name = metricName;
        self.metric_value = metricValue;

        self.datetime = Utils.getTimeStamp();
        
        #lets save it to db
        self.db_save();
        
