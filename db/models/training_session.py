
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

class TrainingSession(Base, DBBase):
    __tablename__ = 'train_sessions'
    __table_args__ = {'extend_existing': True}
    db = db_init('SQLITE')
    

    train_sess_id=Column(BigInteger, primary_key=True, autoincrement=False)
    modset_id = Column(VARCHAR(50))
    start_datetime=Column(DateTime)
    end_datetime=Column(DateTime)
    epochs_total=Column(Integer)
    batch_size=Column(Integer)
    avg_time_secs=Column(Integer)
    initial_checkpoint=Column(VARCHAR(50))
    
    
    def __init__(self, modsetID, trainSessID=0):
        self.reset();
        self.modset_id = modsetID
        if(trainSessID>0):
            self.train_sess_id = trainSessID;

        super().setupDBBase(TrainingSession, TrainingSession.train_sess_id, self.train_sess_id)
        
    def create_table(self):
        self.metadata.create_all(self.db.db_engine)
        
    def reset(self):
        self.epochs_total=0;
        self.batch_size=0;
        avg_time_secs = 0;
        
    def start_session(self, initialVersion=1, epochsTotal=1, batchSize=32):
        self.initial_checkpoint = initialVersion;
        self.epochs_total = epochsTotal;
        self.batch_size = batchSize;
        
        self.generate_sess_id();
        self.set_start_time();
        
        self.db_save();
        
        return self.train_sess_id;
        
    def end_session(self):
        self.set_end_time();
        self.avg_time_secs = int(self.get_time_elapsed()/self.epochs_total);
        
        self.db_save(); #until we fix db_update func
        #self.db_update({'end_datetime':self.end_datetime, 'avg_time_secs':self.avg_time_secs}, {'train_sess_id':self.train_sess_id})
        
        
    def generate_sess_id(self):
        self.set_session_id(Utils.getTimeStampID();)
        print("New Session ID: ", self.train_sess_id)
        return self.train_sess_id;
    
    def set_session_id(self, sess_id):
        self.train_sess_id = sess_id;
        
    def get_session_id(self):
        return self.train_sess_id;
        
    def set_start_time(self):
        self.start_datetime = Utils.getLocalTimeStamp()
        
    def set_end_time(self):
        self.end_datetime = Utils.getLocalTimeStamp()
        
    def get_time_elapsed(self):
        time_elapsed = Utils.getLocalTimeStamp() - self.start_datetime;
        return time_elapsed.total_seconds();
        
