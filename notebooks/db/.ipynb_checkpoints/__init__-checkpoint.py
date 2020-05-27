
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
#from _overlapped import NULL

#import all DB scripts
# from base import Base
# from db_base import DBBase
# from utils import Utils



class db_init():
    db_engine='';
    session='';
 
    
    #db_host='localhost'
    db_host='192.168.0.111'
    
    db_user='db_user'
    db_pword='db_pwd'

    # db_user='root'
    # # db_pword=''
    # db_pword = 'lubdub'
    
    db_port= 3306
    db_name='lotto_hotspot'
    

    #charset='UTF8MB4'

    def __init__(self, db_type="MYSQL"):
        self.init_db_engine(db_type);
        self.init_session();
        
    def init_db_engine(self, db_type):
        if (db_type == 'SQLITE'):
            db_folder = os.getcwd();
            engine = 'sqlite:///'+ db_folder +'/db/ai_lotto.db';
            self.db_engine = create_engine(engine);
            print("Setting up sqlite ", engine)
        else:
            engine = 'mysql+mysqldb://'+self.db_user+':'+self.db_pword+'@'+self.db_host+':'+str(self.db_port)+'/'+self.db_name
            #print('mysql+mysqldb://'+self.db_user+':'+self.db_pword+'@'+self.db_host+':'+str(self.db_port)+'/'+self.db_name)
            
            #self.db_engine = create_engine('mysql+mysqldb://db_user:db_pwd@192.168.0.111:3306/lotto_hotspot');
            
            self.db_engine = create_engine(engine);
            print("Setting up MySQL ", engine)
    
    def init_session(self):
        Session = sessionmaker(bind=self.db_engine);
        self.session = Session();       
