
import os, shutil

class Archiver:
    srcFolder = "../data"
    dstFolder = "../archive"
    srcFiles = []
    dstFiles = []
    
    def __init__(self, src='../data', dst='../archive'):
        self.setup(src, dst);
        self.process_files();
        
        self.setup('../g_drive', dst);
        self.process_files();
        
    def setup(self, src, dst):
        self.srcFolder = src;
        self.dstFolder = dst;
        self.srcFiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.srcFolder)) for f in fn];
        self.dstFiles = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.dstFolder)) for f in fn];
        print('Source:{0} Files'.format(len(self.srcFiles)))
        print('Dest: {0} Files'.format(len(self.dstFiles)))
        
        
    def process_files(self):        
        for srcFilePath in self.srcFiles:
            srcFolderPath = os.path.dirname(srcFilePath)
            srcFileFormat = srcFilePath.split('.')[-1]; # 'h5'
            srcFileName = os.path.basename(srcFilePath);
            game = srcFilePath.split('/')[2]
            if (srcFileFormat == 'h5'): 
                pass;
                self.archive_h5(srcFilePath);
            elif (srcFileFormat == 'pkl'):
                pass;
                self.archive_pkl(srcFilePath);
            elif (srcFileFormat == 'npz'):
                self.archive_npz(srcFilePath);
                print('FileName:{0}[{1} MB]'.format(srcFilePath,round(os.path.getsize(srcFilePath)/(1024*1024)),2) )
            else:
                print("Unknown format: {0}".format(srcFilePath))
                    
    def archive_h5(self, srcFilePath):
        srcFolderPath = os.path.dirname(srcFilePath)
        srcFileName = os.path.basename(srcFilePath);
        srcFileFormat = srcFilePath.split('.')[-1];
        folders = srcFolderPath.split('/');
        try:
            game = folders[2];
            xnInputs = folders[3];
            xnDraws = folders[4]
            srcDatasetID = '{0}_{1}_{2}_dr'.format(game,xnInputs,xnDraws)
            dstFileName = '[{0}]{1}'.format(srcDatasetID, srcFileName);
            dstFilePath = '{0}/{1}/models/{2}'.format(self.dstFolder, game, dstFileName)
            #print('{0}=>{1}'.format(srcFilePath, dstFilePath))
            if(os.path.exists(dstFilePath)):
                pass;
            else:
                self.archive(srcFilePath, dstFilePath);
            
        except IndexError:
            if (folders[3] == 'untrained_models'):
                game = folders[2]
                dstFileName = '[untrained]{0}'.format(srcFileName);
                dstFilePath = '{0}/{1}/models/{2}'.format(self.dstFolder, game, dstFileName)
                if(os.path.exists(dstFilePath)):
                    pass;
                else:
                    self.archive(srcFilePath, dstFilePath);
            else:
                print("exception",srcFilePath)
    
    def archive_pkl(self, srcFilePath):
        game = srcFilePath.split('/')[2]
        isMaster = srcFilePath.split('/')[3]
        srcFileName = os.path.basename(srcFilePath);
        dstFileName = srcFileName;
        if (isMaster == 'master'):
            dstFilePath = '{0}/{1}/masters/{2}'.format(self.dstFolder, game, dstFileName)
        
        if(os.path.exists(dstFilePath)):
                pass;
        else:
            self.archive(srcFilePath, dstFilePath);
            
    def archive_npz(self, srcFilePath):
        game = srcFilePath.split('/')[2]
        srcFileName = os.path.basename(srcFilePath);
        dstFileName = srcFileName;
        dstFilePath = '{0}/{1}/datasets/{2}'.format(self.dstFolder, game, dstFileName)
        #print(srcFilePath)
        if(os.path.exists(dstFilePath)):
                pass;
        else:
            self.archive(srcFilePath, dstFilePath);
            
    def archive(self, srcFilePath, dstFilePath):
        shutil.copy(srcFilePath, dstFilePath);
        print("Copying {0}".format(os.path.basename(dstFilePath)));
    
