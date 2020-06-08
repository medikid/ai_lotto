

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


class iPred(Base, DBBase):
    __tablename__ = 'preds'
    __table_args__ = {'extend_existing': True}
    db = db_init('SQLITE')
    
    pred_id=Column(Integer, primary_key=True, autoincrement=True)    
    modset_id = Column(VARCHAR(50))
    checkpoint=Column(VARCHAR(50))
    draw_id = Column(Integer)
    p1 = Column(DECIMAL(35,30))
    p2 = Column(DECIMAL(35,30))
    p3 = Column(DECIMAL(35,30))
    p4 = Column(DECIMAL(35,30))
    p5= Column(DECIMAL(35,30))
    p6= Column(DECIMAL(35,30))
    p7= Column(DECIMAL(35,30))
    p8= Column(DECIMAL(35,30))
    p9= Column(DECIMAL(35,30))
    p10 = Column(DECIMAL(35,30))
    p11= Column(DECIMAL(35,30))
    p12= Column(DECIMAL(35,30))
    p13= Column(DECIMAL(35,30))
    p14= Column(DECIMAL(35,30))
    p15= Column(DECIMAL(35,30))
    p16= Column(DECIMAL(35,30))
    p17= Column(DECIMAL(35,30))
    p18= Column(DECIMAL(35,30))
    p19= Column(DECIMAL(35,30))
    p20= Column(DECIMAL(35,30))
    p21= Column(DECIMAL(35,30))
    p22= Column(DECIMAL(35,30))
    p23= Column(DECIMAL(35,30))
    p24= Column(DECIMAL(35,30))
    p25= Column(DECIMAL(35,30))
    p26= Column(DECIMAL(35,30))
    p27= Column(DECIMAL(35,30))
    p28= Column(DECIMAL(35,30))
    p29= Column(DECIMAL(35,30))
    p30= Column(DECIMAL(35,30))
    p31= Column(DECIMAL(35,30))
    p32= Column(DECIMAL(35,30))
    p33= Column(DECIMAL(35,30))
    p34= Column(DECIMAL(35,30))
    p35= Column(DECIMAL(35,30))
    p36= Column(DECIMAL(35,30))
    p37= Column(DECIMAL(35,30))
    p38= Column(DECIMAL(35,30))
    p39= Column(DECIMAL(35,30))
    p40= Column(DECIMAL(35,30))
    p41= Column(DECIMAL(35,30))
    p42= Column(DECIMAL(35,30))
    p43= Column(DECIMAL(35,30))
    p44= Column(DECIMAL(35,30))
    p45= Column(DECIMAL(35,30))
    p46= Column(DECIMAL(35,30))
    p47= Column(DECIMAL(35,30))
    p48= Column(DECIMAL(35,30))
    p49= Column(DECIMAL(35,30))
    p50= Column(DECIMAL(35,30))
    p51= Column(DECIMAL(35,30))
    p52= Column(DECIMAL(35,30))
    p53= Column(DECIMAL(35,30))
    p54= Column(DECIMAL(35,30))
    p55= Column(DECIMAL(35,30))
    p56= Column(DECIMAL(35,30))
    p57= Column(DECIMAL(35,30))
    p58= Column(DECIMAL(35,30))
    p59= Column(DECIMAL(35,30))
    p60= Column(DECIMAL(35,30))
    p61= Column(DECIMAL(35,30))
    p62= Column(DECIMAL(35,30))
    p63= Column(DECIMAL(35,30))
    p64= Column(DECIMAL(35,30))
    p65= Column(DECIMAL(35,30))
    p66= Column(DECIMAL(35,30))
    p67= Column(DECIMAL(35,30))
    p68= Column(DECIMAL(35,30))
    p69= Column(DECIMAL(35,30))
    p70= Column(DECIMAL(35,30))
    p71= Column(DECIMAL(35,30))
    p72= Column(DECIMAL(35,30))
    p73= Column(DECIMAL(35,30))
    p74= Column(DECIMAL(35,30))
    p75= Column(DECIMAL(35,30))
    p76= Column(DECIMAL(35,30))
    p77= Column(DECIMAL(35,30))
    p78= Column(DECIMAL(35,30))
    p79= Column(DECIMAL(35,30))
    p80= Column(DECIMAL(35,30))
    wins_in_top5= Column(Integer)
    wins_in_top10 = Column(Integer)
    wins_in_top15= Column(Integer)
    wins_in_top20= Column(Integer)
    wins_in_bottom10= Column(Integer)
    first_10_wins= Column(Integer)
    
    def __init__(self, modsetID, Checkpoint, drawID):
        self.reset();
        self.modset_id = modsetID;
        self.checkpoint = Checkpoint;        
        self.draw_id = int(drawID);

        super().setupDBBase(iPred, iPred.pred_id, self.pred_id)
        
    def reset(self):
        self.pred_id = None;
        self.draw_id = 0;
        self.wins_in_top5 = 0;
        self.wins_in_top5=0
        self.wins_in_top10=0
        self.wins_in_top15=0
        self.wins_in_top20=0
        self.wins_in_bottom10=0
        self.first_10_wins=0
    
    def set_pred(self, ball_n, pred):
        setattr(self, "p"+str(ball_n), pred)
        
    def get_pred(self, ball_n):
        return getattr(self, "p"+ball_n)
    
    def from_pred_array(self, p_array):
        if (p_array.ndim == 2):
            p_array = p_array[0]
            
        for (idx,), value in np.ndenumerate(p_array):
            self.set_pred(idx+1, value); #ball = index+1
            
    def __get_dict__(self):
        dic = {};
        dic[self.pred_id] = {'modset_id': self.modset_id, 'checkpoint': self.checkpoint, 'draw_id': self.draw_id};

        i = 1;
        while (i <=80):
            key = 'p'+str(i);
            dic[self.pred_id][key] = getattr(self, key);
            i += 1;
        
        return dic;
    
