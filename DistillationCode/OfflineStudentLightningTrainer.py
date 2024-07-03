import torch
import torch.nn.functional as F
import torch.nn as nn
import ISNetLayersZe as ISNetLayers
import ISNetFunctionsZe as ISNetFunctions
import LRPDenseNetZe as LRPDenseNet
import globalsZe as globals
import pytorch_lightning as pl
import warnings
from collections import OrderedDict
import numpy as np
import copy
from torch.autograd import Variable
import torch.autograd as ag
import random
import warnings
import OfflineStudentZVariableEpsTorch as TeacherStudentTorch

class ISNetStudentLgt(pl.LightningModule):
    def __init__(self,teacher,architecture,mask='noise',multiLabel=False,
                 LRPBlock=False,rule='e',multiple=True,selective=False,epsSelective=0.1,
                 highest=False,e=1e-2,Zb=False,detach=True,
                 randomLogit=False,
                 SequentialInputShape=None,SequentialPreFlattenShape=None,
                 alpha=1,beta=1,gamma=1,delta=1,zeta=0,eta=0,clss=0,
                 norm=True,optim='SGD',nesterov=False,
                 LR=1e-3,momentum=0.99,WD=0,dropLr=None,
                 maskedTest=False,
                 beginAsTeacher=True,clip=1,
                 huber=False,L1=False,
                 normTarget=False,stdTarget=False,CELRP=False,KLDivLRP=False,
                 geoStd=False,
                 reduction='mean',
                 zeroBias=False,freezeLastLayer=False,
                 epsLow=-4,epsHigh=-1,pencentageZero=0.2,
                 classes=2,dropout=False,normPerBatch=False,
                 detachTeacher=True,CELogits=False,
                 normLogits=False,normFeatures=False,
                 maskedStudentPercent=0.0,
                 pyramidLoss=False,minSize=8,ratio=2,
                 HiddenLayerPenalization=False,
                 curriculumMode='None',
                 curriculumEpochs=1000,extremeD=0.3,finalD=0.0,
                 scale=False,maskTargetLRP=False,
                 loss='MSE',secondOptimLayers=[],
                 scheduler='drop',LRPlossOnFeatures=False,
                 mode='offline',P=1,dPyramid=1,
                 crossLoss=False,pretrainedOnline=None,
                 alternateTrain=False,alternateCycle=100,
                 segmenter=None,th=0.5,
                 clsLossWeight=0,temperature=3,
                 matchArchitecture=True,
                 textFeatures=None,clipPreprocess=None,clipSavedMaps=None,
                 trainableT=False,teachereT=10.0,
                 multiLossLayers=[],
                 sepAlpha=1,sepBeta=1,sepEmb=0,sepEmbCos=0,
                 sepZero=0,
                 CLIPLikeLast=False,CLIPLikeLastFreeze=False,
                 inputGradAblation=False,attentionAblation=False,
                 GradCAMAblation=False,
                 frozenLayers='none',
                 basketEps=0.3,CLIPModLastLayer=False,
                 balancing=None,MNISTSpecial=False):
        
        super (ISNetStudentLgt,self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        self.model=TeacherStudentTorch.TeacherStudent(teacher=teacher,mask=mask,
                 beginAsTeacher=beginAsTeacher,freezeLastLayer=freezeLastLayer,zeroBias=zeroBias,
                 LRPBlock=LRPBlock,e=e,Zb=Zb,rule=rule,multiple=multiple,
                 randomLogit=randomLogit,selective=selective,epsSelective=epsSelective,
                 highest=highest,
                 detach=detach,SequentialInputShape=SequentialInputShape,
                 SequentialPreFlattenShape=SequentialPreFlattenShape,
                 epsLow=epsLow,epsHigh=epsHigh,pencentageEpsZero=pencentageZero,
                 architecture=architecture,classes=classes,dropout=dropout,
                 HiddenLayerPenalization=HiddenLayerPenalization,scale=scale,
                 mode=mode,pretrainedOnline=pretrainedOnline,
                 matchArchitecture=matchArchitecture,
                 textFeatures=textFeatures,clipPreprocess=clipPreprocess,clipSavedMaps=clipSavedMaps,
                 trainableT=trainableT,teachereT=teachereT,CLIPLikeLast=CLIPLikeLast,
                 inputGradAblation=inputGradAblation,attentionAblation=attentionAblation,
                 GradCAMAblation=GradCAMAblation,frozenLayers=frozenLayers,
                 CLIPModLastLayer=CLIPModLastLayer,MNISTSpecial=MNISTSpecial)
        
        
        self.lr=LR
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.delta=delta
        self.zeta=zeta
        self.eta=eta
        self.P=P
        self.optim=optim
        self.momentum=momentum
        self.WD=WD
        self.nesterov=nesterov
        self.clip=clip
        self.multiLabel=multiLabel
        self.reduction=reduction
        self.dropLr=dropLr
        self.mode=mode
        self.normPerBatch=normPerBatch
        self.CELogits=CELogits
        self.normLogits=normLogits
        self.normFeatures=normFeatures
        self.scale=scale
        
        self.curriculumMode=curriculumMode
        self.curriculumEpochs=curriculumEpochs
        self.extremeD=extremeD
        self.dPyramid=dPyramid
        self.dLayers=1
        self.finalD=finalD
        self.GWRPDescending=True
        self.GWRPRank=True
        self.simpleCurriculum=False
        
        self.pyramidLoss=pyramidLoss
        self.minSize=minSize
        self.ratio=ratio
        
        self.maskedTest=maskedTest
        self.maskedStudentPercent=maskedStudentPercent
        
        self.temperature=temperature
        self.clsLossWeight=clsLossWeight
        self.huber=huber
        self.L1=L1
        self.geoStd=geoStd
        self.norm=norm
        self.normTarget=normTarget
        self.CELRP=CELRP
        self.stdTarget=stdTarget
        self.detachTeacher=detachTeacher
        self.KLDivLRP=KLDivLRP
        self.maskTargetLRP=maskTargetLRP
        self.loss=loss
        self.scheduler=scheduler
        self.LRPlossOnFeatures=LRPlossOnFeatures
        self.crossLoss=crossLoss
        if crossLoss and 'online' not in self.mode:
            raise ValueError('Cross loss expects an online mode')
            
        self.alternateTrain=alternateTrain
        self.alternateCycle=alternateCycle
        self.alternateInitiated=False
        
        self.segmenter=segmenter
        self.th=th
        if self.segmenter is not None:
            for param in self.segmenter.parameters():
                param.requires_grad=False
        self.freezeLastLayer=freezeLastLayer
        self.penultimate=False
        if isinstance(secondOptimLayers, str):
            if secondOptimLayers=='last':
                if 'densenet' in architecture:
                    secondOptimLayers=['classifier']
                elif 'resnet' in architecture:
                    secondOptimLayers=['fc']
                elif 'vgg' in architecture:
                    secondOptimLayers=['classifier.6']
                else:
                    raise ValueError('Authomatic selection of secondOptimLayers only implemented for densenet, vgg and resnet. For other architectures, provide list with layer names')
            elif secondOptimLayers=='half':
                if 'densenet' in architecture:
                    secondOptimLayers=['classifier','norm5','features.denseblock4',
                                       'features.transition3','features.denseblock3']
                else:
                    raise ValueError('Not implemented')
            elif secondOptimLayers=='lastBlock':
                if 'densenet' in architecture:
                    secondOptimLayers=['classifier','norm5','features.denseblock4']
                else:
                    raise ValueError('Not implemented')
            elif secondOptimLayers=='penultimate':
                self.penultimate=True
                if textFeatures is None or not freezeLastLayer:
                    raise ValueError('secondOptimLayers=penultimate only implemented for clip with freezeLastLayer')
                secondOptimLayers=['CLIP_vision_encoder.layer4.1.intermediateExtra.weight',
                                   'CLIP_vision_encoder.layer4.1.intermediateExtra.bias']
            elif secondOptimLayers=='alternateBlock4':
                if 'resnet' in architecture:
                    secondOptimLayers=['CLIP_vision_encoder.layer4.1','fc']
                else:
                    raise ValueError('Authomatic selection of secondOptimLayers only implemented for densenet, vgg and resnet. For other architectures, provide list with layer names')
            elif secondOptimLayers=='alternateBlock4_0':
                if 'resnet' in architecture:
                    secondOptimLayers=['CLIP_vision_encoder.layer4.0','fc']
                else:
                    raise ValueError('Authomatic selection of secondOptimLayers only implemented for densenet, vgg and resnet. For other architectures, provide list with layer names')
            else:
                print(self.model)
                raise ValueError('Unrecognized secondOptimLayers')
                
        self.multiLossLayers=multiLossLayers        
        if isinstance(multiLossLayers, str):
            if multiLossLayers=='lastResLayer':
                if 'resnet' in architecture:
                    self.multiLossLayers=['CLIP_vision_encoder.layer4']
                else:
                    raise ValueError('Not implemented')
            elif multiLossLayers=='lastResLayerfc':
                if 'resnet' in architecture:
                    self.multiLossLayers=['CLIP_vision_encoder.layer4','fc']
                else:
                    raise ValueError('Not implemented')
            else:
                raise ValueError('Not implemented')
                
        
        self.secondOptimLayers=secondOptimLayers
        print(self.secondOptimLayers)
        #for i,name in enumerate(self.secondOptimLayers,0):
        #    self.secondOptimLayers
        if len(self.secondOptimLayers)!=0 and freezeLastLayer and textFeatures is None:
            raise ValueError('Using dual optimizers and freezing last layer')
            
        if (len(self.secondOptimLayers)+len(self.multiLossLayers))>0:
            self.SeparateParameters()
        self.sepAlpha=sepAlpha
        self.sepBeta=sepBeta
        self.sepAlphaInit=sepAlpha
        self.sepBetaInit=sepBeta
        self.sepEmb=sepEmb
        self.sepEmbCos=sepEmbCos
        self.sepZero=sepZero
        self.GradCAMAblation=GradCAMAblation
        self.clss=clss
        self.basketEps=basketEps
        
        self.CLIPLikeLastFreeze=CLIPLikeLastFreeze
        if CLIPLikeLastFreeze:
            print('careful, CLIPLikeLastFreeze should not be used when starting the student from scratch')
            for param in self.model.student.fc.parameters():
                param.requires_grad=False
                
                
        if balancing is None:
            self.balancing=None
        else:
            self.balancing=torch.tensor(balancing)
        
    def Alternate(self,init=False):
        if not self.alternateInitiated:
            print('Training teacher')
            self.alternateTurn='teacher'
            self.memoryTeacher={}
            for name,param in self.model.teacher.named_parameters():
                self.memoryTeacher[name]=param.requires_grad
            self.memoryStudent={}
            for name,param in self.model.student.named_parameters():
                self.memoryStudent[name]=param.requires_grad
            for name,param in self.model.student.named_parameters():
                    param.requires_grad=False
            self.lastEpoch=0
            self.alternateInitiated=True
        
        
        if (((self.current_epoch%self.alternateCycle)==0) and self.lastEpoch!=self.current_epoch):
            #reverts model being trained
            if self.alternateTurn=='teacher':
                print('Training student')
                self.alternateTurn='student'
                for name,param in self.model.teacher.named_parameters():
                    param.requires_grad=False
                for name,param in self.model.student.named_parameters():
                    param.requires_grad=self.memoryStudent[name]
            else:
                print('Training teacher')
                self.alternateTurn='teacher'
                for name,param in self.model.student.named_parameters():
                    param.requires_grad=False
                for name,param in self.model.teacher.named_parameters():
                    param.requires_grad=self.memoryTeacher[name]
        
        
        #if (self.lastEpoch!=self.current_epoch):
        #    print('Teacher param:', self.model.teacher.features.conv0.weight[0][0][0])
        #    print('Student param:', self.model.student.features.conv0.weight[0][0][0])
        self.lastEpoch=self.current_epoch
        
        
    def run_segmenter(self,x):
        with torch.no_grad():
            masks=self.segmenter(x)
        masks=torch.nn.functional.softmax(masks,dim=1)
        masks=masks[:,1,:,:].unsqueeze(1).repeat(1,3,1,1)
        masks=torch.where(masks<=self.th,
                     torch.zeros(masks.shape).type_as(masks),
                     torch.ones(masks.shape).type_as(masks))
        return masks
    
    def SeparateParameters(self):
        if self.mode=='online':
            model=self.model.model
        else:
            model=self.model.student
        self.paramsLogit=[]
        paramsLogitNames=[]
        self.paramsMultiLoss=[]
        for name,param in model.named_parameters():
            #print(name)
            if param.requires_grad:
                for layer in self.secondOptimLayers:
                    if name[:len(layer)]==layer:
                        print('entered second optim: ',name)
                        self.paramsLogit.append(param)
                        paramsLogitNames.append(name)
                        break
                for layer in self.multiLossLayers:
                    #if layer in name:
                    #    print(name)
                    if name[:len(layer)]==layer:
                        if name not in paramsLogitNames:
                            print('entered multiLossLayers optim: ',name)
                            self.paramsMultiLoss.append(param)
                            break
        print('Parameters in logit loss optimizer: ',len(self.paramsLogit))
        print('Parameters in multi loss optimizer: ',len(self.paramsMultiLoss))
        print('Total parameters: ',len([param for name, 
                    param in self.named_parameters() if \
                    param.requires_grad]))
        
    def forward(self,xS=None,xT=None,
                maskS=None,maskT=None,
                runLRPBlock=False,runLRPFlex=False,
                runFeatureGrad=False):
        #xT: input for teacher. If None, teacher is not run
        #xS: input for student. If None, student is not run
        #maskT: if provided, mask is applied to teacher input
        #maskS: if provided, mask is applied to student input
        #runLRPBlock: if True, will run the LRP block
        #runLRPFlex: if True, will run the LRP flex
        
        return self.model(xT=xT,xS=xS,maskT=maskT,maskS=maskS,
                          runLRPBlock=runLRPBlock,runLRPFlex=runLRPFlex,
                          runFeatureGrad=runFeatureGrad)

    def configure_optimizers(self):  
        if self.mode=='onlineSeparateWeights':
            #params=[{'params':self.model.teacher.parameters()},
            #        {'params':self.model.student.parameters(), 'lr': (self.lr/self.P)}]
            params=[{'params':filter(lambda p: p.requires_grad,self.model.teacher.parameters())},
                    {'params':filter(lambda p: p.requires_grad,self.model.student.parameters()),
                     'lr': (self.lr/self.P)}]
            #P will not affect the sudent training, only the teacher
        else:
            params=filter(lambda p: p.requires_grad,self.parameters())
        if (self.optim=='Adam'):
            from deepspeed.ops.adam import FusedAdam
            optimizer=FusedAdam(params,lr=self.lr)

        else:
            optimizer=torch.optim.SGD(params,
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.WD,
                                    nesterov=self.nesterov)
            
        
            
        if (self.dropLr is not None and self.scheduler=='drop'):
            #print('current epoch:',self.current_epoch)
            #drops=[]
            #for item in self.dropLr:
            #    x=item-self.current_epoch
            #    if x>=0:
            #        drops.append(x)
            #self.dropLr=drops
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.dropLr,
                                                           verbose=True)   
            return [optimizer],[scheduler]
        elif self.scheduler=='cyclic':
            scheduler=torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                        base_lr=5*1e-5,max_lr=0.1,
                                                        step_size_up=456,mode='triangular',
                                                        step_size_down=456,
                                                        base_momentum=0.9,max_momentum=0.99,
                                                        verbose=True)
            return [optimizer],[scheduler]
        else:
            return optimizer

    def training_step(self,train_batch,batch_idx):
        opt=self.optimizers()
        opt.zero_grad()
        
        
        if self.alternateTrain:
            self.Alternate()
        
        if (self.segmenter is not None):
            try:
                inputs,masks,labels=train_batch
                del masks
            except:
                inputs,labels=train_batch
            #run segmenter online, it should output logits, with 2 channels
            masks=self.run_segmenter(inputs)
        else:
            if self.model.maskMode=='none':
                inputs,labels=train_batch
                masks=None
            else:
                inputs,masks,labels=train_batch
            
        if self.maskedStudentPercent>0:
            ones=torch.ones((inputs.shape[0],1,1,1)).type_as(inputs)
            b=torch.bernoulli(ones*self.maskedStudentPercent)
            masksS=b*masks+(1-b)*torch.ones(masks.shape).type_as(masks) 
            #outputs masks of ones (all foreground) for unmasked samples
        else:
            masksS=None
        
        out=self.forward(xS=inputs,xT=inputs,maskS=masksS,maskT=masks,
                         runLRPBlock=(self.gamma>0),runLRPFlex=(self.delta>0),
                         runFeatureGrad=self.GradCAMAblation)
        
        
        #print('labels: ', labels[0])
        #print('Teacher Logits: ', out['outputTeacher'][0])
        #print('Student Logits: ', out['outputStudent'][0])
        #print('Teacher soft: ', torch.softmax(out['outputTeacher'][0],-1))
        #print('Student soft: ', torch.softmax(out['outputStudent'][0],-1))
        #print(self.model.teacher.fc.weight.data)
        #print('Teacher Flex: ', out['LRPFlexTeacher'])
        #print('Student Flex: ', out['LRPFlexStudent'])
        
        
        #if self.maskTargetLRP:
        #    maskLRP=masks
        #else:
        #    maskLRP=None
        losses=TeacherStudentTorch.studentLoss(out,huber=self.huber,L1=self.L1,
                                               norm=self.norm,
                                               normTarget=self.normTarget,CELRP=self.CELRP,
                                               reduction=self.reduction,stdNorm=self.stdTarget,
                                               online=('online' in self.mode),labels=labels,
                                               normPerBatch=self.normPerBatch,
                                               detachTeacher=self.detachTeacher,
                                               CELogits=self.CELogits,
                                               normLogits=self.normLogits,
                                               normFeatures=self.normFeatures,
                                               KLDivLRP=self.KLDivLRP,
                                               pyramidLoss=self.pyramidLoss,
                                               minSize=self.minSize,ratio=self.ratio,
                                               dPyramid=self.dPyramid,dLayers=self.dLayers,
                                               GWRPDescending=self.GWRPDescending,
                                               GWRPRank=self.GWRPRank,
                                               simpleCurriculum=self.simpleCurriculum,
                                               geoStd=self.geoStd,scale=self.model.scale,
                                               mask=masks,loss=self.loss,
                                               LRPlossOnFeatures=self.LRPlossOnFeatures,
                                               maskTargetLRP=self.maskTargetLRP,
                                               crossLoss=self.crossLoss,
                                               temperature=self.temperature,
                                               basketEps=self.basketEps)
        loss=0
        if losses['logits'] is not None:
            loss=self.alpha*losses['logits']
        if losses['features'] is not None:
            loss=loss+self.beta*losses['features']
        if losses['LRP'] is not None:
            loss=loss+self.gamma*losses['LRP']
        if losses['LRPFlex'] is not None:
            loss=loss+self.delta*losses['LRPFlex']
        if losses['HeatmapLossISNet'] is not None:
            loss=loss+self.zeta*losses['HeatmapLossISNet']
        if losses['HeatmapLossISNetFlex'] is not None:
            loss=loss+self.eta*losses['HeatmapLossISNetFlex']
        if losses['LRPTeacher'] is not None:
            loss=loss+self.gamma*losses['LRPTeacher']
        if losses['LRPFlexTeacher'] is not None:
            loss=loss+self.delta*losses['LRPFlexTeacher']    
        if losses['classificationStudent'] is not None:
            loss=loss+self.clss*losses['classificationStudent']    
        
        
        if 'online' in self.mode:#online teacher
            loss=self.P*loss+(1-self.P)*losses['classificationTeacher']
        
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')
        
        opt.zero_grad()
        
        
        #for param_group in opt.optimizer.param_groups:
        #    print(param_group['lr'])
        
        #self.lr_schedulers.print_lr()
        
        if len(self.secondOptimLayers)>0:
            #print('Second optim')
            if self.alternateTrain:
                raise ValueError('secondOptimLayers may conflict with alternateTrain')
            memory={}
            for name,param in self.named_parameters():
                memory[name]=param.requires_grad
                param.requires_grad=False
            for param in self.paramsLogit:
                param.requires_grad=True
                #print(name, param)
                
            if not self.freezeLastLayer:
                if self.clsLossWeight>0:
                    self.manual_backward(self.sepAlpha*losses['logits']+self.clsLossWeight*\
                                         losses['classificationStudent'],
                                         retain_graph=True)
                    
                else:
                    self.manual_backward(self.sepAlpha*losses['logits'],retain_graph=True)
                    #for name,param in self.named_parameters():
                    #    if param.requires_grad:
                    #        print(name)
            else:
                lSep=self.sepAlpha*losses['logits']+self.sepBeta*losses['features']
                if self.sepEmb>0 or self.sepEmbCos>0:
                    lSep=lSep+self.sepEmb*losses['logits']
                self.manual_backward(lSep,retain_graph=True)
            for name,param in self.named_parameters():
                param.requires_grad=memory[name]
            del memory
            
            #for param in self.paramsLRP:
            #    if param.grad is not None:
            #        if param.grad.sum().item()!=0:
            #            raise ValueError('we have a gradient for lrp')
            #tmp=[]
            #for param in self.paramsLogit:
            #    tmp.append(param.grad.clone())
        
            for param in self.paramsLogit:
                param.requires_grad=False
                
        
        if len(self.multiLossLayers)>0:
            if self.alternateTrain:
                raise ValueError('secondOptimLayers may conflict with alternateTrain')
            memory={}
            for name,param in self.named_parameters():
                memory[name]=param.requires_grad
                param.requires_grad=False
            for param in self.paramsMultiLoss:
                param.requires_grad=True
                
            if not self.freezeLastLayer:
                if self.clsLossWeight>0 and losses['classificationStudent'] is not None:
                    self.manual_backward(losses['LRPFlex']+self.sepAlpha*losses['logits']+\
                                         self.clsLossWeight*losses['classificationStudent'],
                                         retain_graph=True)
                else:
                    self.manual_backward(losses['LRPFlex']+self.sepAlpha*losses['logits'],
                                         retain_graph=True)
            else:
                self.manual_backward(losses['LRPFlex']+self.sepBeta*losses['features'],
                                     retain_graph=True)
            for name,param in self.named_parameters():
                param.requires_grad=memory[name]
            del memory
            
            for param in self.paramsMultiLoss:
                param.requires_grad=False
        
        
        self.manual_backward(loss)
        
        if len(self.secondOptimLayers)>0:
            for param in self.paramsLogit:
                param.requires_grad=True
                
        
        
        #for i,param in enumerate(self.paramsLogit,0):
        #    if not torch.equal(param.grad,tmp[i]):
        #        raise ValueError('Altered paramsLogit')
        
        if self.clip is not None:
            if self.clip!=0:
                self.clip_gradients(opt, gradient_clip_val=self.clip, gradient_clip_algorithm="norm")
        
        if self.alternateTrain:
            mt={}
            ms={}
            for name,param in self.model.teacher.named_parameters():
                if not param.requires_grad:
                    mt[name]=param.data.clone()
            for name,param in self.model.student.named_parameters():
                if not param.requires_grad:
                    ms[name]=param.data.clone()
        
        opt.step()
        
        if self.alternateTrain:
            #avoids changes to frozen parameters
            #weight decay and momentum may change parameters even when they do not require grad
            for name,param in self.model.teacher.named_parameters():
                if not param.requires_grad:
                    param.data=mt[name]
            for name,param in self.model.student.named_parameters():
                if not param.requires_grad:
                    param.data=ms[name]
            del mt
            del ms
        
        #log losses
        logs={}
        if loss.detach().item()>1e3:#reject outliers
            logs['sum']=torch.ones(loss.shape).type_as(loss)
        else:
            logs['sum']=loss.detach()
        for key in losses:
            if losses[key] is not None:
                if losses[key].detach().item()>1e3:
                    print('Large loss: ', key,', logging 1 instead')
                    logs[key]=torch.ones(losses[key].shape).type_as(losses[key])
                else:
                    logs[key]=losses[key].detach()

        self.log('train_loss', logs, on_epoch=True)#, on_epoch=True)
        
    def on_train_epoch_start(self, training_step_outputs=None):  
        #lr step
        #if self.global_rank == 0:
        if self.lr_schedulers() is not None:
            sch = self.lr_schedulers()
            sch.step()
            
        if self.curriculumMode!='None':
            self.Curriculum()
            
        if self.sepZero!=0:
            percent=self.current_epoch/(self.sepZero)
            if percent>1:
                percent=1
            self.sepAlpha=(1-percent)*self.sepAlphaInit
            self.sepBeta=(1-percent)*self.sepBetaInit
            print('sepAlpha:',self.sepAlpha)
            print('sepBeta:',self.sepBeta)
        
    def Curriculum(self):
        self.GWRPRank=True
        self.simpleCurriculum=False
        
        if self.curriculumMode=='None':
            self.dPyramid=1
            self.dLayers=1
            self.GWRPDescending=True
        elif self.curriculumMode=='GWRPPyramid':
            self.dLayers=1
            if self.current_epoch<=self.curriculumEpochs/2:
                self.GWRPDescending=False#focus on easiers losses (smallest)
                delta=self.current_epoch/(self.curriculumEpochs/2)
                self.dPyramid=self.extremeD+delta*(1-self.extremeD)
            elif ((self.current_epoch>self.curriculumEpochs/2) and \
                  (self.current_epoch<self.curriculumEpochs)):#end of training
                delta=self.current_epoch-(self.curriculumEpochs/2)
                delta=delta/(self.curriculumEpochs/2)
                self.GWRPDescending=True#focus on largest losses
                self.dPyramid=1-delta*(1-self.extremeD)
            else:
                self.GWRPDescending=True
                self.GWRPRank=False #prioritize bigger maps
                self.dPyramid=self.extremeD
                
        elif self.curriculumMode=='GWRPPyramidEasy':
            self.dLayers=1
            if self.current_epoch<=self.curriculumEpochs:
                self.GWRPDescending=False#focus on easiers losses (smallest)
                delta=self.current_epoch/(self.curriculumEpochs)
                self.dPyramid=self.extremeD+delta*(1-self.extremeD)
            else:
                self.GWRPDescending=False
                self.dPyramid=1
            
        elif self.curriculumMode=='GWRPLayers':
            self.dPyramid=1
            if self.current_epoch<=self.curriculumEpochs/2:
                self.GWRPDescending=False#focus on easiers losses (smallest)
                delta=self.current_epoch/(self.curriculumEpochs/2)
                self.dLayers=self.extremeD+delta*(1-self.extremeD)
            elif ((self.current_epoch>self.curriculumEpochs/2) and \
                  (self.current_epoch<self.curriculumEpochs)):#end of training
                delta=self.current_epoch-(self.curriculumEpochs/2)
                delta=delta/(self.curriculumEpochs/2)
                self.GWRPDescending=True#focus on largest losses
                self.dLayers=1-delta*(1-self.extremeD)
            else:
                self.GWRPDescending=True
                self.dLayers=self.extremeD
            
        elif self.curriculumMode=='Layers':
            #order losses from firts to last layer, with decreasing weighs
            self.GWRPRank=False
            self.dPyramid=1
            if self.current_epoch<=self.curriculumEpochs/2:
                self.dLayers=1
            elif ((self.current_epoch>self.curriculumEpochs/2) and \
                  (self.current_epoch<self.curriculumEpochs)):#end of training
                delta=self.current_epoch-(self.curriculumEpochs/2)
                delta=delta/(self.curriculumEpochs/2)
                self.dLayers=1-delta*(1-self.extremeD)
            else:
                self.dLayers=self.extremeD
                if self.extremeD==0:
                    self.model.HLP=False
        elif self.curriculumMode=='LayersEasyToHard':
            #order losses from firts to last layer, with decreasing weighs
            self.dPyramid=1
            if self.current_epoch<=self.curriculumEpochs/2:#GWRP ascending
                self.GWRPRank=True
                self.GWRPDescending=False#focus on easiers losses (smallest)
                delta=self.current_epoch/(self.curriculumEpochs/2)
                self.dLayers=self.extremeD+delta*(1-self.extremeD)
            elif ((self.current_epoch>self.curriculumEpochs/2) and \
                  (self.current_epoch<self.curriculumEpochs)):#Prioritize early layers
                self.GWRPRank=False
                delta=self.current_epoch-(self.curriculumEpochs/2)
                delta=delta/(self.curriculumEpochs/2)
                self.dLayers=1-delta#goes to 0
            else:#just input loss
                self.GWRPRank=False
                self.model.HLP=False
                
        elif self.curriculumMode=='LayersEasyToHardSimple':
            #order losses from firts to last layer, with decreasing weighs
            self.dPyramid=1
            if self.current_epoch<=self.curriculumEpochs/2:#GWRP ascending
                self.GWRPRank=True
                self.GWRPDescending=False#focus on easiers losses (smallest)
                delta=self.current_epoch/(self.curriculumEpochs/2)
                self.dLayers=self.extremeD+delta*(1-self.extremeD)
            elif ((self.current_epoch>self.curriculumEpochs/2) and \
                  (self.current_epoch<self.curriculumEpochs)):#Prioritize early layers
                self.simpleCurriculum=True#no GWRP over layers, just increase input lrp weight
                delta=self.current_epoch-(self.curriculumEpochs/2)
                delta=delta/(self.curriculumEpochs/2)
                self.dLayers=1-delta*(1-self.finalD)
            else:#just input loss
                self.dLayers=self.finalD
                if self.finalD==0:
                    self.model.HLP=False        
            
        else:
            raise ValueError('Not implemented curriculum mode')
        print('dPyramid,dLayers,GWRPDescending,GWRPRank:',
              self.dPyramid,self.dLayers,self.GWRPDescending,self.GWRPRank)        

    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        torch.set_grad_enabled(True)
        
        #data format: channel first
        if dataloader_idx==0:
            if (self.segmenter is not None):
                try:
                    inputs,masks,labels=val_batch
                    del masks
                except:
                    inputs,labels=val_batch
                masks=self.run_segmenter(inputs)
            else:
                if self.model.maskMode=='none':
                    inputs,labels=val_batch
                    masks=None
                else:
                    inputs,masks,labels=val_batch
        else:
            inputs,labels=val_batch
            
        
        if dataloader_idx==1:
            out=self.forward(xS=inputs,xT=None,maskS=None,maskT=None,
                         runLRPBlock=False,runLRPFlex=False,
                         runFeatureGrad=True)#only run student
            logits=out['outputStudent']
            #ood eval, take only classification loss, no mask
            if(self.multiLabel):
                loss=F.binary_cross_entropy_with_logits(logits,labels)
            else:
                #classifierLoss=F.cross_entropy(outputs,labels.squeeze(1))
                try:
                    loss=F.cross_entropy(logits,labels.squeeze(1))
                except:
                    loss=F.cross_entropy(logits,labels)
                    
            
            self.log('val_oodLoss', loss, on_step=False, on_epoch=True)
                    
        else:
            if self.maskedStudentPercent>0:
                ones=torch.ones((inputs.shape[0],1,1,1)).type_as(inputs)
                b=torch.bernoulli(ones*self.maskedStudentPercent)
                masksS=b*masks+(1-b)*torch.ones(masks.shape).type_as(masks) 
                #outputs masks of ones (all foreground) for unmasked samples
            else:
                masksS=None
                
            out=self.forward(xS=inputs,xT=inputs,maskS=masksS,maskT=masks,
                         runLRPBlock=(self.gamma>0),runLRPFlex=(self.delta>0),
                         runFeatureGrad=self.GradCAMAblation)
            #print(out)
            #print('Teacher Flex: ', out['LRPFlexTeacher'])
            #print('Student Flex: ', out['LRPFlexStudent'])
            
            #if self.maskTargetLRP:
            #    maskLRP=masks
            #else:
            #    maskLRP=None
            
            
            losses=TeacherStudentTorch.studentLoss(out,huber=self.huber,L1=self.L1,
                                                   norm=self.norm,online=('online' in self.mode),
                                                   normTarget=self.normTarget,CELRP=self.CELRP,
                                                   reduction=self.reduction,
                                                   stdNorm=self.stdTarget,labels=labels,
                                                   normPerBatch=self.normPerBatch,
                                                   detachTeacher=self.detachTeacher,
                                                   CELogits=self.CELogits,
                                                   normLogits=self.normLogits,
                                                   normFeatures=self.normFeatures,
                                                   KLDivLRP=self.KLDivLRP,
                                                   pyramidLoss=self.pyramidLoss,
                                                   minSize=self.minSize,ratio=self.ratio,
                                                   dPyramid=self.dPyramid,
                                                   dLayers=0,GWRPDescending=True,GWRPRank=False,
                                                   simpleCurriculum=self.simpleCurriculum,
                                                   geoStd=self.geoStd,
                                                   scale=self.model.scale,
                                                   mask=masks,loss=self.loss,
                                                   LRPlossOnFeatures=self.LRPlossOnFeatures,
                                                   maskTargetLRP=self.maskTargetLRP,
                                                   crossLoss=self.crossLoss,
                                                   temperature=self.temperature,
                                                   basketEps=self.basketEps)
            #HLP: for validation, we consider only the LRP losses at the input. Changing the weights
            #according to curriculum strategy could be problematic for validation, creating a
            #tendency for late losses to be higher

            loss=0
            if losses['logits'] is not None:
                loss=self.alpha*losses['logits']        
            if losses['features'] is not None:
                loss=loss+self.beta*losses['features']
            if losses['LRP'] is not None:
                loss=loss+self.gamma*losses['LRP']
            if losses['LRPFlex'] is not None:
                loss=loss+self.delta*losses['LRPFlex']
            if losses['LRPTeacher'] is not None:
                loss=loss+self.gamma*losses['LRPTeacher']
            if losses['LRPFlexTeacher'] is not None:
                loss=loss+self.delta*losses['LRPFlexTeacher']
            if 'online' in self.mode:#online teacher
                loss=self.P*loss+(1-self.P)*losses['classificationTeacher']
            #if self.clsLossWeight>0:
            #    loss=loss+losses['classificationStudent']
                
            logs={}
            if loss.detach().item()>1e3:#reject outliers
                print('Large LRP Sum loss')
                logs['sum']=torch.ones(loss.shape).type_as(loss)
            else:
                logs['sum']=loss.detach()
            for key in losses:
                if losses[key] is not None:
                    if losses[key].detach().item()>1e3:
                        print('Large loss: ', key,', logging 1 instead')
                        logs[key]=torch.ones(losses[key].shape).type_as(losses[key])
                    else:
                        logs[key]=losses[key].detach()

            self.log('val_loss_iid', logs, on_step=False, on_epoch=True)#, on_epoch=True)
            self.log('val_iidLoss', logs['sum'], on_step=False, on_epoch=True)

        opt=self.optimizers()
        opt.zero_grad()
        
        #loss=loss.detach()
        #if dataloader_idx==0:
        #    return {'iidLoss':loss}
        #if dataloader_idx==1:
        #    return {'oodLoss':loss}

    #def validation_step_end(self, batch_parts):
    #    
    #    if 'iidLoss' in list(batch_parts.keys()):
    #        lossType='iidLoss'
    #    elif 'oodLoss' in list(batch_parts.keys()):
    #        lossType='oodLoss'
    #    else:
    #        raise ValueError('Unrecognized loss')
    #        
    #    if(batch_parts[lossType].dim()>1):
    #        losses=batch_parts[lossType]
    #        return {lossType: torch.mean(losses,dim=0)}
    #    else:
    #        return batch_parts#

    #def validation_epoch_end(self, validation_step_outputs):
    #    for item in validation_step_outputs:
    #        try:
    #            lossType=list(item[0].keys())[0]
    #            loss=item[0][lossType].unsqueeze(0)
    #        except:
    #            lossType=list(item.keys())[0]
    #            loss=item[lossType].unsqueeze(0)
    #        for i,out in enumerate(item,0):
    #            if(i!=0):
    #                loss=torch.cat((loss,out[lossType].unsqueeze(0)),dim=0)
    #        self.log('val_'+lossType,torch.mean(loss,dim=0),
    #                 on_epoch=True,sync_dist=True)
        #torch.save(self.model.returnBackbone(),'modelsave.pt')
        #print(self.model.student)
    
    def test_step(self,test_batch,batch_idx):
        #data format: channel first
        try:
            inputs,labels=test_batch
            masks=None
        except:
            inputs,masks,labels=test_batch
        if not self.maskedTest:
            masks=None
            
        if self.maskedTest and (self.segmenter is not None):
            masks=self.run_segmenter(inputs)
            
        with torch.no_grad():
            out=self.forward(xS=inputs,xT=None,maskS=masks,maskT=None,
                         runLRPBlock=False,runLRPFlex=False,
                         runFeatureGrad=False)#only run student
        logits=out['outputStudent']
        return {'pred': logits.detach(), 'labels': labels}
            

    def test_step_end(self, batch_parts):
        if(batch_parts['pred'].dim()>2):
            logits=batch_parts['pred']
            labels=batch_parts['labels']
            return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                    'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1])}
        else:
            return batch_parts

    def test_epoch_end(self, test_step_outputs):
        pred=test_step_outputs[0]['pred']
        labels=test_step_outputs[0]['labels']
            
        for i,out in enumerate(test_step_outputs,0):
            if (i!=0):
                pred=torch.cat((pred,out['pred']),dim=0)
                labels=torch.cat((labels,out['labels']),dim=0)
                
        self.TestResults=pred,labels
            
    def returnBackbone(self,network='student'):
        return self.model.returnBackbone(network)