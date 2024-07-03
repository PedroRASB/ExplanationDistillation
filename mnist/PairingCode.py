import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.utils.data as Tdata
from argparse import ArgumentParser
import os
from pytorch_lightning import loggers as pl_loggers
import SingleLabelEval as SLE
import ast
import h5py

import sys
sys.path.append('../DistillationCode')
import PairedISNetLightningZeMultiStep as Teachers
import OfflineTeacherLightningTrainer as Teachers2
import OfflineStudentLightningTrainer as ISNetLightning
import ISNetFunctionsZe as IsNet
import OfflineStudentZVariableEpsTorch as EDNet


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
torch.autograd.set_detect_anomaly(True)

TrainedPath='../Trained/LRPBlockFasterISNet/MNIST/Trained'
os.makedirs(TrainedPath,exist_ok=True)


class MNISTDatsetColor(Dataset):
    def __init__(self, mode,bias,privileged=False,confounding=False,
                 source='../data/mnistColor/'):
        self.images=[]
        if (mode=='train'):
            if (not bias or confounding):
                raise ValueError ('Use bias=True and confounding=False for training')
            self.dataset=h5py.File(source+'train.h5py', 'r')
        if (mode=='val'):
            if (not bias or confounding):
                raise ValueError ('Use bias=True and confounding=False for training')
            self.dataset=h5py.File(source+'val.h5py', 'r')
        if (mode=='test'):
            if not bias:
                self.dataset=h5py.File(source+'unbiasedTest.h5py', 'r')
            else:
                if confounding:
                    self.dataset=h5py.File(source+'confoundingTest.h5py', 'r')
                else:
                    self.dataset=h5py.File(source+'biasedTest.h5py', 'r')
        self.privileged=privileged
    def __len__(self):
        return (len(self.dataset['images']))
    def __getitem__(self,idx):
        image=torch.from_numpy(self.dataset['images'][idx])
        if self.privileged:
            privileged=torch.from_numpy(self.dataset['privileged'][idx])
        label=torch.tensor(self.dataset['labels'][idx]).long()
        
        if self.privileged:
            return image,privileged,label
        else:
            return image,label

        
    
    
def Test(net,ckpt,mode,device,precision,hparams,mask=False):
    print('Model:',ckpt)
    print('Mode:',mode)
    
    trainer=pl.Trainer(precision=precision,accelerator=hparams.accelerator,devices=device,
                       strategy=hparams.strategy,num_nodes=1,
                       auto_select_gpus=True)
    try:
        net=ISNetLightning.ISNetStudentLgt.load_from_checkpoint(ckpt,Zb=(hparams.Zb=='1'))
    except:
        net=ISNetLightning.ISNetStudentLgt.load_from_checkpoint(ckpt,backwardCompat=True,
                                                                Zb=(hparams.Zb=='1'))
    net.eval()
    if mode=='idtest':
        testSet=MNISTDatsetColor(mode='test',
                                        bias=True,confounding=False)
    elif mode=='oodtest':
        testSet=MNISTDatsetColor(mode='test',
                                        bias=False,confounding=False)
    elif mode=='sgtest':
        testSet=MNISTDatsetColor(mode='test',
                                        bias=True,confounding=True)
        
        
    testingLoader=Tdata.DataLoader(testSet,batch_size=int(hparams.batch),shuffle=False,
                                   num_workers=1)
    if mask:
        net.maskedTest=True
    trainer.test(net,dataloaders=testingLoader)
    pred,labels=net.TestResults
    net.maskedTest=False
    C=SLE.ConfusionMatrix(pred,labels)
    print('Confusion Matrix:')
    print(C)
    performance=SLE.PerformanceMatrix(C)
    print('Performance:')
    print(performance)
    acc=SLE.Acc(pred,labels)
    print('Accuracy:',acc)
    for i in list(range(10)):
        print(' ')

def getModel(hparams,changeLR=None):
    
    teacher=torch.load(hparams.teacher)
                
    if '[' in hparams.stepLr:
        stepLr=ast.literal_eval(hparams.stepLr)
    elif hparams.stepLr=='1':
        stepLr=[120,160,180,190]
    elif hparams.stepLr=='early':
        stepLr=[25,130,180,190]
    elif hparams.stepLr=='single':
        stepLr=[30]
    elif hparams.stepLr=='twice':
        stepLr=[25,130]
    else:
        stepLr=None
        
    if '[' in hparams.secondOptimLayers:
        secondOptimLayers=ast.literal_eval(hparams.secondOptimLayers)
    else:
        secondOptimLayers=hparams.secondOptimLayers
        
    if hparams.sepLastBlock=='0':
        multiLossLayers=[]
    else:
        multiLossLayers='lastResLayer'
    
    if(hparams.continuing=='1' and hparams.train=='1'):
        #checkpoint=TrainedPath+hparams.name+'/'+'last.ckpt'
        lst=[TrainedPath+hparams.name+'/'+item for item in os.listdir(TrainedPath+hparams.name) \
        if ('last' in item)]   
        checkpoint=max(lst, key=os.path.getctime)
    else:
        checkpoint=hparams.checkpoint
        
    if changeLR is not None:
        LR=changeLR
    else:
        LR=float(hparams.lr)
        
    
    text_features=None
    clipPreprocess=None
            
    if (checkpoint is None):
            
        pretrainedOnline=None
        if ('online' in hparams.mode and hparams.teacher is not None):
            try:
                pretrainedOnline=Teachers.ISNetLgt.load_from_checkpoint(hparams.teacher)\
                .returnBackbone()
            except:
                #teacher=Teachers.ISNetLgt.load_from_checkpoint(
                #    hparams.teacher,backwardCompat=True,mask=hparams.mask).returnBackbone()
                pretrainedOnline=Teachers2.ISNetFlexLgt.load_from_checkpoint(hparams.teacher)\
                .returnBackbone()
            
        net=ISNetLightning.ISNetStudentLgt(teacher=teacher,
                                           beginAsTeacher=(hparams.beginAsTeacher=='1'),
                                           #true makes the MNIST and dogs problem trivial
                                           multiLabel=False,
                                           e=float(hparams.e),
                                           LR=LR,optim='SGD',
                                           alpha=float(hparams.alpha),
                                           beta=float(hparams.beta),gamma=float(hparams.gamma),
                                           delta=float(hparams.delta),
                                           zeta=float(hparams.zeta),
                                           eta=float(hparams.eta),
                                           selective=(hparams.selective=='1'),
                                           epsSelective=float(hparams.epsSelective),
                                           highest=(hparams.highest=='1'),
                                           rule=hparams.rule,
                                           multiple=(hparams.multiple=='1'),
                                           momentum=float(hparams.momentum),
                                           dropLr=stepLr,WD=float(hparams.WD),
                                           mask=hparams.mask,
                                           detach=(hparams.detach=='1'),
                                           huber=(hparams.huber=='1'),
                                           L1=(hparams.L1=='1'),
                                           norm=hparams.norm,
                                           CELRP=(hparams.CELRP=='1'),
                                           reduction=hparams.reduction,
                                           clip=float(hparams.clip),
                                           zeroBias=(hparams.zeroBias=='1'),
                                           randomLogit=(hparams.randomLogit=='1'),
                                           freezeLastLayer=(hparams.freezeLastLayer=='1'),
                                           epsLow=-float(hparams.epsLow),
                                           epsHigh=-float(hparams.epsHigh),
                                           pencentageZero=float(hparams.pencentageEpsZero),
                                           architecture=hparams.architecture,
                                           classes=10,dropout=False,
                                           mode=hparams.mode,
                                           normPerBatch=(hparams.normPerBatch=='1'),
                                           detachTeacher=(hparams.detachTeacher=='1'),
                                           CELogits=(hparams.CELogits=='1'),
                                           normLogits=(hparams.normLogits=='1'),
                                           KLDivLRP=(hparams.KLDivLRP=='1'),
                                           normFeatures=(hparams.normFeatures=='1'),
                                           maskedStudentPercent=float(hparams.maskedStudentPercent),
                                           pyramidLoss=(hparams.pyramidLoss=='1'),
                                           minSize=int(hparams.minSize),
                                           ratio=float(hparams.ratio),
                                           HiddenLayerPenalization=(hparams.HLP=='1'),
                                           curriculumMode=hparams.curriculumMode,
                                           extremeD=float(hparams.extremeD),
                                           curriculumEpochs=int(hparams.curriculumEpochs),
                                           scale=(hparams.scale=='1'),
                                           maskTargetLRP=(hparams.maskTargetLRP=='1'),
                                           loss=hparams.loss,
                                           secondOptimLayers=secondOptimLayers,
                                           scheduler=hparams.scheduler,
                                           LRPlossOnFeatures=(hparams.LRPlossOnFeatures=='1'),
                                           P=float(hparams.P),
                                           Zb=(hparams.Zb=='1'),
                                           dPyramid=float(hparams.dPyramid),
                                           crossLoss=(hparams.crossLoss=='1'),
                                           LRPBlock=(hparams.LRPBlock=='1'),
                                           pretrainedOnline=pretrainedOnline,
                                           alternateTrain=(hparams.alternateTrain=='1'),
                                           alternateCycle=int(hparams.alternateCycle),
                                           matchArchitecture=False,
                                           clipPreprocess=clipPreprocess,
                                           textFeatures=text_features,
                                           trainableT=(hparams.trainableT=='1'),
                                           teachereT=float(hparams.teachereT),
                                           multiLossLayers=multiLossLayers,
                                           sepAlpha=float(hparams.sepAlpha),
                                           sepBeta=float(hparams.sepBeta),
                                           temperature=float(hparams.temperature),
                                           CLIPLikeLast=(hparams.CLIPLikeLast=='1'),
                                           sepZero=float(hparams.sepZero),
                                           CLIPLikeLastFreeze=(hparams.CLIPLikeLastFreeze=='1'),
                                           inputGradAblation=(hparams.inputGradAblation=='1'),
                                           attentionAblation=(hparams.attentionAblation=='1'),
                                           GradCAMAblation=(hparams.GradCAMAblation=='1'),
                                                               MNISTSpecial=True
                                          )
        
        del(teacher)
        del(pretrainedOnline)
    else:
        print('Restoring from:',checkpoint)   
        print(hparams.architecture)
        net=ISNetLightning.ISNetStudentLgt.load_from_checkpoint(checkpoint,
                                           beginAsTeacher=(hparams.beginAsTeacher=='1'),
                                           #true makes the MNIST problem trivial
                                           multiLabel=False,
                                           e=float(hparams.e),
                                           LR=LR,optim='SGD',
                                           alpha=float(hparams.alpha),
                                           beta=float(hparams.beta),gamma=float(hparams.gamma),
                                           delta=float(hparams.delta),
                                           zeta=float(hparams.zeta),
                                           eta=float(hparams.eta),
                                           epsSelective=float(hparams.epsSelective),
                                           highest=(hparams.highest=='1'),
                                           rule=hparams.rule,
                                           multiple=(hparams.multiple=='1'),
                                           momentum=float(hparams.momentum),
                                           dropLr=stepLr,WD=float(hparams.WD),
                                           mask=hparams.mask,
                                           detach=(hparams.detach=='1'),
                                           huber=(hparams.huber=='1'),
                                           L1=(hparams.L1=='1'),
                                           norm=hparams.norm,
                                           CELRP=(hparams.CELRP=='1'),
                                           reduction=hparams.reduction,
                                           clip=float(hparams.clip),
                                           zeroBias=(hparams.zeroBias=='1'),
                                           randomLogit=(hparams.randomLogit=='1'),
                                           freezeLastLayer=(hparams.freezeLastLayer=='1'),
                                           epsLow=-float(hparams.epsLow),
                                           epsHigh=-float(hparams.epsHigh),
                                           pencentageZero=float(hparams.pencentageEpsZero),
                                           architecture=hparams.architecture,
                                           classes=10,dropout=False,
                                           mode=hparams.mode,
                                           normPerBatch=(hparams.normPerBatch=='1'),
                                           detachTeacher=(hparams.detachTeacher=='1'),
                                           CELogits=(hparams.CELogits=='1'),
                                           normLogits=(hparams.normLogits=='1'),
                                           KLDivLRP=(hparams.KLDivLRP=='1'),
                                           normFeatures=(hparams.normFeatures=='1'),
                                           maskedStudentPercent=float(hparams.maskedStudentPercent),
                                           pyramidLoss=(hparams.pyramidLoss=='1'),
                                           minSize=int(hparams.minSize),
                                           ratio=float(hparams.ratio),
                                           HiddenLayerPenalization=(hparams.HLP=='1'),
                                           curriculumMode=hparams.curriculumMode,
                                           extremeD=float(hparams.extremeD),
                                           curriculumEpochs=int(hparams.curriculumEpochs),
                                           scale=(hparams.scale=='1'),
                                           maskTargetLRP=(hparams.maskTargetLRP=='1'),
                                           loss=hparams.loss,
                                           secondOptimLayers=secondOptimLayers,
                                           scheduler=hparams.scheduler,
                                           LRPlossOnFeatures=(hparams.LRPlossOnFeatures=='1'),
                                           P=float(hparams.P),
                                           Zb=(hparams.Zb=='1'),
                                           dPyramid=float(hparams.dPyramid),
                                           crossLoss=(hparams.crossLoss=='1'),
                                           LRPBlock=(hparams.LRPBlock=='1'),
                                           alternateTrain=(hparams.alternateTrain=='1'),
                                           alternateCycle=int(hparams.alternateCycle),
                                           matchArchitecture=False,
                                           clipPreprocess=clipPreprocess,
                                           textFeatures=text_features,
                                           trainableT=(hparams.trainableT=='1'),
                                           teachereT=float(hparams.teachereT),
                                           multiLossLayers=multiLossLayers,
                                           sepAlpha=float(hparams.sepAlpha),
                                           sepBeta=float(hparams.sepBeta),
                                           temperature=float(hparams.temperature),
                                           CLIPLikeLast=(hparams.CLIPLikeLast=='1'),
                                           sepZero=float(hparams.sepZero),
                                           CLIPLikeLastFreeze=(hparams.CLIPLikeLastFreeze=='1'),
                                           inputGradAblation=(hparams.inputGradAblation=='1'),
                                           attentionAblation=(hparams.attentionAblation=='1'),
                                           GradCAMAblation=(hparams.GradCAMAblation=='1'),
                                                               MNISTSpecial=True)
        
        
        
        if net.model.selective!=(hparams.selective=='1'):
            if net.gamma!=0:
                raise ValueError('Changing selective in LRP block not implemented')
            net.model.selective=(hparams.selective=='1')

            
    return net
        
def main(hparams):
    #unbiased ISNet
    
    #if ((hparams.checkpoint is not None) and (hparams.name not in hparams.checkpoint)):
    #    NetName=hparams.checkpoint[hparams.checkpoint[:-1].rfind('/')+1:-1]
    #else:
    NetName=hparams.name   
    print('Net name:', NetName)
    
    batch=int(hparams.batch)
    if ('[' in hparams.devices):#call for specific gpu: use [x]
        device=[int(hparams.devices[1])]
    else:
        device=int(hparams.devices)
        
        
        
    trainSet=MNISTDatsetColor(mode='train',
                                        bias=True,confounding=False)
    valiidSet=MNISTDatsetColor(mode='val',
                                        bias=True,confounding=False)
        
    
    precision=32
    

    
    trainingLoader=Tdata.DataLoader(trainSet,batch_size=batch,shuffle=True,
                                    num_workers=int(hparams.workers))
    iidValLoader=Tdata.DataLoader(valiidSet,batch_size=batch,shuffle=False,
                                      num_workers=int(hparams.workers))

    if (not os. path. exists(TrainedPath+'/'+NetName)):
        os.makedirs(TrainedPath+'/'+NetName, exist_ok=True)
        
    #checkpoint callbacks for dual validation (iid and ood)
    checkpoint_callback_iid = ModelCheckpoint(dirpath=TrainedPath+NetName+'/',
                                          filename=NetName+'IID'+'{epoch}-{step}',
                                          monitor='val_iidLoss',
                                          verbose=True,
                                          save_top_k=1,
                                          mode='min',
                                          every_n_epochs=1,
                                          save_on_train_epoch_end=False,
                                          auto_insert_metric_name=False,
                                          save_weights_only=False,
                                          save_last=True
                                          )
    
    tb_logger=pl_loggers.TensorBoardLogger(save_dir='Logs/'+NetName+'/')
    if not os.path.exists('Logs/'+NetName+'/'):
        os.makedirs('Logs/'+NetName+'/', exist_ok=True)
        
    
    if(hparams.continuing=='1' and hparams.train=='1'):
        checkpoint=TrainedPath+NetName+'/'+'last.ckpt'
    else:
        checkpoint=hparams.checkpoint
        
    if(hparams.train=='1'):
        net=getModel(hparams)
        trainer=pl.Trainer(precision=precision,
                           callbacks=[checkpoint_callback_iid,],
                           accelerator=hparams.accelerator,devices=device,
                           max_epochs=int(hparams.epochs),strategy=hparams.strategy,
                           num_nodes=int(hparams.nodes),
                           auto_select_gpus=True,
                           logger=tb_logger,
                           deterministic=True
                          )
        
        #trainer.fit(net,trainingLoader,val_dataloaders=[iidValLoader,oodValLoader])
        if (hparams.continuing=='0'):
            trainer.fit(net,trainingLoader,val_dataloaders=[iidValLoader])
        else:
            trainer.fit(net,trainingLoader,val_dataloaders=[iidValLoader],
                        ckpt_path=checkpoint)
        cis=[TrainedPath+NetName+'/'+item for item in os.listdir(TrainedPath+NetName) \
        if (NetName[NetName.find('/')+1:]+'IID' in item)]
        #select lastest
        checkpointIID=max(cis, key=os.path.getctime) 
    else:
        trainer=pl.Trainer(precision=precision,accelerator=hparams.accelerator,devices=device,
                           strategy=hparams.strategy,num_nodes=1,
                           auto_select_gpus=True)

        print(os.listdir(checkpoint))
        cis=[checkpoint+item for item in os.listdir(checkpoint) \
        if ('IID' in item)]
        
        checkpointIID=max(cis, key=os.path.getctime)  
        hparams.checkpoint=checkpointIID
        net=getModel(hparams)
        
    #test:
    print('iid validation, iid test:')
    Test(net,checkpointIID,'idtest',device,precision,hparams)
    print('iid validation, ood test:')
    Test(net,checkpointIID,'oodtest',device,precision,hparams)
    print('iid validation, systematic shift test:')
    Test(net,checkpointIID,'sgtest',device,precision,hparams)
    
    print(' ')
    print(' ')
    print(' ')
    print(' ')
    
    
    
if __name__ == '__main__':    
    parser=ArgumentParser()
    
    parser.add_argument('--typeBias', default='color')
    parser.add_argument('--train', default='1')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--local_rank', default=None)
    parser.add_argument('--devices', default='1')
    parser.add_argument('--nodes', default=1)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--batch', default=128)
    parser.add_argument('--name', default='EDNetMNIST')
    parser.add_argument('--continuing', default='0')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--teacher', default=None)
    parser.add_argument('--workers', default='5')
    parser.add_argument('--lr', default='1e-3')
    parser.add_argument('--stepLr', default='0')
    parser.add_argument('--bias', default='1')
    
    parser.add_argument('--momentum', default='0.99')
    parser.add_argument('--WD', default='0')
    
    parser.add_argument('--alpha', default='0')
    parser.add_argument('--beta', default='0')
    parser.add_argument('--gamma', default='0')
    parser.add_argument('--delta', default='1')
    parser.add_argument('--zeta', default='0')
    parser.add_argument('--eta', default='0')
    parser.add_argument('--e', default='1e-2')
    parser.add_argument('--clip', default='1.0')
    parser.add_argument('--sepAlpha', default='1')
    parser.add_argument('--sepBeta', default='0')
    
    parser.add_argument('--selective', default='0')
    parser.add_argument('--epsSelective', default='0.01')
    parser.add_argument('--highest', default='0')
    parser.add_argument('--rule', default='z+')
    parser.add_argument('--randomLogit', default='1')
    parser.add_argument('--multiple', default='0')
    
    parser.add_argument('--mask', default='none')
    parser.add_argument('--detach', default='1')
    #parser.add_argument('--accumulate', default='1') removed due to lightning bug with gradient clip
    
    
    parser.add_argument('--loss', default='L1')
    parser.add_argument('--huber', default='0')
    parser.add_argument('--L1', default='0')
    parser.add_argument('--norm', default='geoL1')
    parser.add_argument('--CELRP', default='0')
    parser.add_argument('--reduction', default='mean')
    
    
    
    parser.add_argument('--zeroBias', default='0')
    parser.add_argument('--freezeLastLayer', default='0')
    parser.add_argument('--epsLow', default='3')
    parser.add_argument('--epsHigh', default='2')
    parser.add_argument('--pencentageEpsZero', default='0.0')
    parser.add_argument('--architecture', default='resnet18')
    parser.add_argument('--normPerBatch', default='0')
    
    parser.add_argument('--mode', default='offline')
    parser.add_argument('--detachTeacher', default='1')
    parser.add_argument('--CELogits', default='0')
    parser.add_argument('--normLogits', default='0')
    parser.add_argument('--KLDivLRP', default='0')
    parser.add_argument('--normFeatures', default='0')
    parser.add_argument('--maskedStudentPercent', default='0.0')
    parser.add_argument('--pyramidLoss', default='1')
    parser.add_argument('--minSize', default='8')
    parser.add_argument('--ratio', default='2')
    parser.add_argument('--curriculumMode', default='None')
    parser.add_argument('--extremeD', default='0.2')
    parser.add_argument('--curriculumEpochs', default='100')
    parser.add_argument('--scale', default='0')
    parser.add_argument('--maskTargetLRP', default='0')
    parser.add_argument('--secondOptimLayers', default='[]')
    parser.add_argument('--scheduler', default='drop')
    parser.add_argument('--LRPlossOnFeatures', default='1')
    parser.add_argument('--P', default='1.0')
    parser.add_argument('--Zb', default='0')
    parser.add_argument('--dPyramid', default='1.0')
    parser.add_argument('--crossLoss', default='0')
    parser.add_argument('--LRPBlock', default='0')
    parser.add_argument('--alternateTrain', default='0')
    parser.add_argument('--alternateCycle', default='200')
    parser.add_argument('--trainableT', default='0')
    parser.add_argument('--teachereT', default='10')
    parser.add_argument('--sepLastBlock', default='0')
    parser.add_argument('--temperature', default='3')
    parser.add_argument('--CLIPLikeLast', default='0')
    parser.add_argument('--sepZero', default='0')
    parser.add_argument('--CLIPLikeLastFreeze', default='0')
    parser.add_argument('--inputGradAblation', default='0')
    parser.add_argument('--attentionAblation', default='0')
    parser.add_argument('--GradCAMAblation', default='0')
    parser.add_argument('--beginAsTeacher', default='0')# DO NOT SET TO 1, MAKES PROBLEM TRIVIAL
    
    
    
    
    
    
    parser.add_argument('--HLP', default='0')
    
    args=parser.parse_args()
    main(args)