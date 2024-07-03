#new
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

class ISNetFlexLgt(pl.LightningModule):
    def __init__(self,architecture,pretrained=None,mask='noise',multiLabel=False,
                 heat=True,P=0.7,d=0,
                 optim='SGD',nesterov=False,
                 LR=1e-3,momentum=0.99,WD=0,dropLr=None,
                 maskedTest=False,clip=1,
                 zeroBias=False,
                 epsLow=-2,epsHigh=-1,pencentageZero=0,
                 classes=1,dropout=False,
                 pyramidLoss=False,minSize=8,ratio=2,
                 HiddenLayerPenalization=False,dLoss=1,
                 cut=1,cut2=25,sumMaps=None,selective=False,
                 alternativeForeground=False,
                 dPyramid=1,Zb=False,separate=False,
                 imageNetPretrained=False,
                 segmenter=None,th=0.5,
                 freezeBackbone=False):
        
        super (ISNetFlexLgt,self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        
        self.model=TeacherStudentTorch.TeacherStudent(mask=mask,
                 freezeLastLayer=False,zeroBias=zeroBias,
                 LRPBlock=False,epsLow=epsLow,epsHigh=epsHigh,
                 pencentageEpsZero=pencentageZero,
                 architecture=architecture,classes=classes,dropout=dropout,
                 HiddenLayerPenalization=HiddenLayerPenalization,
                 selective=selective,mode='online',pretrainedOnline=pretrained,
                 Zb=Zb,imageNetPretrained=imageNetPretrained)
        
        self.heat=heat
        self.lr=LR
        self.optim=optim
        self.momentum=momentum
        self.WD=WD
        self.nesterov=nesterov
        self.clip=clip
        self.multiLabel=multiLabel
        self.dropLr=dropLr
        self.penalizeAll=HiddenLayerPenalization
        self.dLoss=dLoss
        self.testLoss=False
        self.alternativeForeground=alternativeForeground
        self.dPyramid=dPyramid
        self.separate=separate
        
        self.segmenter=segmenter
        self.th=th
        if self.segmenter is not None:
            for param in self.segmenter.parameters():
                param.requires_grad=False
        
        if ((imageNetPretrained or pretrained is not None) and freezeBackbone):
            self.model.model=Freeze(self.model.model)

        if isinstance(P, dict):
            self.P=P[0]
            self.increaseP=P
            
        else:
            self.P=P
            self.increaseP=None
        self.d=d
        
        self.cut=cut
        self.cut2=cut2
        self.tuneCut=False
        
        self.pyramidLoss=pyramidLoss
        self.minSize=minSize
        self.ratio=ratio
        
        self.maskedTest=maskedTest
            
            
    def forward(self,xT,maskT=None):
        #xT: input for teacher. If None, teacher is not run
        #maskT: if provided, mask is applied to teacher input
        
        return self.model(xT=xT,xS=None,maskT=maskT,maskS=None,
                          runLRPBlock=False,runLRPFlex=self.heat)

    def configure_optimizers(self):        
        if (self.optim=='Adam'):
            from deepspeed.ops.adam import FusedAdam
            optimizer=FusedAdam(filter(
                    lambda p: p.requires_grad,
                                        self.parameters()),
                                        lr=self.lr)
            
        else:
            optimizer=torch.optim.SGD(filter(
                lambda p: p.requires_grad,
                                    self.parameters()),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.WD,
                                    nesterov=self.nesterov)

        if(self.dropLr is not None):
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
        else:
            return optimizer

    def compound_loss(self,out,labels,masks=None):
        
        loss= TeacherStudentTorch.TeacherLoss(out,labels,
                masks=masks,pyramidLoss=self.pyramidLoss,dPyramid=self.dPyramid,
                minSize=self.minSize,ratio=self.ratio,
                heat=self.heat,tuneCut=self.tuneCut,
                d=self.d,dLoss=self.dLoss,
                #cut=self.cut,cut2=self.cut2,
                cutFlex=self.cut,cut2Flex=self.cut2,
                alternativeForeground=self.alternativeForeground,
                separate=self.separate)

    
        if not self.heat or masks is None:
            return loss['classification']
        if not self.tuneCut:
            return loss['classification'],loss['LRPFlex']
        else:
            self.keys=list(loss['mapAbsFlex'].keys())
            return loss['classification'],loss['LRPFlex'],loss['mapAbsFlex']
        
    def run_segmenter(self,x):
        masks=self.segmenter(x)
        masks=torch.nn.functional.softmax(masks,dim=1)
        masks=masks[:,1,:,:].unsqueeze(1).repeat(1,3,1,1)
        masks=torch.where(masks<=self.th,
                     torch.zeros(masks.shape).type_as(masks),
                     torch.ones(masks.shape).type_as(masks))
        return masks
    
    def training_step(self,train_batch,batch_idx):
        opt=self.optimizers()
        opt.zero_grad()
        
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
        
        #for param_group in opt.optimizer.param_groups:
        #    print(param_group['lr'])
        
        if self.tuneCut:
            if(self.current_epoch!=self.cutEpochs):
                self.heat=False
                out=self.forward(inputs,masks)
                loss=self.compound_loss(out,labels=labels)
            if(self.current_epoch==self.cutEpochs):
                self.heat=True
                out=self.forward(inputs,masks)
                cLoss,hLoss,mapAbs=self.compound_loss(out,labels=labels,
                                                      masks=masks)
                #take only values from last tuning epoch
                self.updateCut(mapAbs)
                #use only for tuning cut value, ignores heatmap loss
                loss=cLoss
            
            self.log('train_loss',loss.detach(),                     
                     on_epoch=True)
        else:
            #update dinamic P
            if (self.increaseP is not None):
                epochs=list(self.increaseP.keys())
                epochs.sort()
                for epoch in epochs:
                    if (self.current_epoch>=epoch):
                        self.P=self.increaseP[epoch]

            #data format: channel first
            if (self.heat):#ISNet
                out=self.forward(inputs,masks)
                cLoss,hLoss=self.compound_loss(out,labels=labels,
                                               masks=masks)
                loss=(1-self.P)*cLoss+self.P*hLoss

                self.log('train_loss', {'Classifier':cLoss.detach(),
                                        'Heatmap':hLoss.detach(),
                                        'Sum':loss.detach()},                     
                         on_epoch=True)
                
            else:#Common DenseNet
                out=self.forward(inputs,masks)
                loss=self.compound_loss(out,labels=labels)
                self.log('train_loss',loss,                     
                         on_epoch=True)
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')
        
        opt.zero_grad()
        self.manual_backward(loss)
        
        if self.clip is not None:
            if self.clip!=0:
                self.clip_gradients(opt, gradient_clip_val=self.clip,
                                    gradient_clip_algorithm="norm")
        opt.step()
        
    def on_train_epoch_start(self, training_step_outputs=None):  
        #lr step
        #if self.global_rank == 0:
        if self.dropLr:
            sch = self.lr_schedulers()
            sch.step()

    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        torch.set_grad_enabled(True)
        
        if self.model.maskMode=='none':
            inputs,labels=val_batch
            masks=None
        else:
            try:
                inputs,masks,labels=val_batch
            except:
                inputs,labels=val_batch
                
        if (self.segmenter is not None):
            masks=self.run_segmenter(inputs)
            
        if self.tuneCut:
            self.heat=False
        if dataloader_idx==1:
            tmp=self.heat
            self.heat=False
            masks=None
        
        if (self.heat):#ISNet
            out=self.forward(inputs,masks)
            cLoss,hLoss=self.compound_loss(out,labels=labels,masks=masks)
            loss=(1-self.P)*cLoss+self.P*hLoss
        else:#Common DenseNet
            logits=self.forward(inputs,masks)
            loss=self.compound_loss(logits,labels=labels)
            
        if dataloader_idx==0:
            return {'iidLoss':loss.detach()}
        if dataloader_idx==1:
            self.heat=tmp
            return {'oodLoss':loss.detach()}

        opt=self.optimizers()
        opt.zero_grad()

    def validation_step_end(self, batch_parts):
        
        if 'iidLoss' in list(batch_parts.keys()):
            lossType='iidLoss'
        elif 'oodLoss' in list(batch_parts.keys()):
            lossType='oodLoss'
        else:
            raise ValueError('Unrecognized loss')
            
        if(batch_parts[lossType].dim()>1):
            losses=batch_parts[lossType]
            return {lossType: torch.mean(losses,dim=0)}
        else:
            return batch_parts

    def validation_epoch_end(self, validation_step_outputs):
        for item in validation_step_outputs:
            try:
                lossType=list(item[0].keys())[0]
                loss=item[0][lossType].unsqueeze(0)
            except:
                lossType=list(item.keys())[0]
                loss=item[lossType].unsqueeze(0)
            for i,out in enumerate(item,0):
                if(i!=0):
                    loss=torch.cat((loss,out[lossType].unsqueeze(0)),dim=0)
            self.log('val_'+lossType,torch.mean(loss,dim=0),
                     on_epoch=True,sync_dist=True)
    
    def test_step(self,test_batch,batch_idx):
        #data format: channel first
        if not self.maskedTest:
            inputs,labels=test_batch
        else:
            try:
                inputs,masks,labels=test_batch
            except:
                inputs,labels=test_batch
            if (self.segmenter is not None):
                masks=self.run_segmenter(inputs)
        
        if self.testLoss:
            if self.heat:
                if self.maskedTest:
                    out=self.forward(inputs,masks)
                else:
                    out=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(out,labels=labels,masks=masks)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(), 'hLoss': hLoss.detach()}
            else:
                if self.maskedTest:
                    out=self.forward(inputs,masks)
                else:
                    out=self.forward(inputs)
                cLoss=self.compound_loss(out,labels=labels)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(),
                        'hLoss': torch.zeros(cLoss.shape).type_as(cLoss)}
        elif (self.heat):#ISNet
            if self.maskedTest:
                out=self.forward(inputs,masks)
            else:
                out=self.forward(inputs)
            logits=out['outputTeacher']
            heatmaps=out['LRPFlexTeacher']['input']
            return {'pred': logits.detach(), 'labels': labels,
                    'images': inputs.cpu().float().detach(),
                    'heatmaps': heatmaps.cpu().float().detach()}

        else:#Common DenseNet
            if self.maskedTest:
                out=self.forward(inputs,masks)
            else:
                out=self.forward(inputs)
            logits=out['outputTeacher']
            return {'pred': logits.detach(), 'labels': labels}
            

    def test_step_end(self, batch_parts):
        if(batch_parts['pred'].dim()>2):
            logits=batch_parts['pred']
            labels=batch_parts['labels']
            if (not self.heat):
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1])}
            elif self.testLoss:
                cLoss=batch_parts['cLoss']
                hLoss=batch_parts['hLoss']
                #print(cLoss.shape)
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1]),
                        'cLoss': cLoss.view(cLoss.shape[0]*cLoss.shape[1],cLoss.shape[-1]), 
                        'hLoss': hLoss.view(hLoss.shape[0]*hLoss.shape[1],hLoss.shape[-1])}
            else:
                images=batch_parts['images']
                heatmaps=batch_parts['heatmaps']
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[2]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1]),
                        'images': images.view(images.shape[0]*images.shape[1],images.shape[2],
                                              images.shape[3],images.shape[4]),
                        'heatmaps': heatmaps.view(heatmaps.shape[0]*heatmaps.shape[1],
                                                  heatmaps.shape[2],
                                                  heatmaps.shape[3],heatmaps.shape[4],
                                                  heatmaps.shape[5])}
        else:
            return batch_parts

    def test_epoch_end(self, test_step_outputs):
        pred=test_step_outputs[0]['pred']
        labels=test_step_outputs[0]['labels']
        if self.testLoss:
            cLoss=test_step_outputs[0]['cLoss'].unsqueeze(0)
            hLoss=test_step_outputs[0]['hLoss'].unsqueeze(0)
        elif (self.heat):
            images=test_step_outputs[0]['images']
            heatmaps=test_step_outputs[0]['heatmaps']
            
        for i,out in enumerate(test_step_outputs,0):
            if (i!=0):
                pred=torch.cat((pred,out['pred']),dim=0)
                labels=torch.cat((labels,out['labels']),dim=0)
                if self.testLoss:
                    cLoss=torch.cat((cLoss,out['cLoss'].unsqueeze(0)),dim=0)
                    hLoss=torch.cat((hLoss,out['hLoss'].unsqueeze(0)),dim=0)
                elif (self.heat):
                    images=torch.cat((images,out['images']),dim=0)
                    heatmaps=torch.cat((heatmaps,out['heatmaps']),dim=0)
                
        if self.testLoss:
            self.TestResults=pred,labels,cLoss.mean().item(),hLoss.mean().item()
        elif (self.heat):
            self.TestResults=pred,labels,images,heatmaps
        else:
            self.TestResults=pred,labels
            
    def returnBackbone(self,network='student'):
        return self.model.returnBackbone(network)
    
    def initTuneCut(self,epochs):
        #train for self.cutEpochs to find cut values, do not use heatmap loss
        self.tuneCut=True
        self.cutEpochs=epochs-1
            
    def resetCut(self):
        self.aggregateE={}
        for name in self.keys:
            self.aggregateE[name]=[0,0,0]

    def updateWelford(self,existingAggregate,newValue):
        #https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)
    
    
    def updateCut(self,maps):
        if not hasattr(self, 'aggregateE'):
            self.resetCut()
        #print(maps)
        for layer in self.keys:
            mapAbs=maps[layer]
            
            mapAbsZ=mapAbs[:,:int(mapAbs.shape[1]/2)]
            mapAbsE=mapAbs[:,int(mapAbs.shape[1]/2):]


            for i,_ in enumerate(mapAbsE,0):#batch iteration
                valueE=torch.mean(mapAbsE[i].detach().float()).item()
                self.aggregateE[layer]=self.updateWelford(self.aggregateE[layer],valueE)



    def finalizeWelford(self,existingAggregate):
        # Retrieve the mean, variance and sample variance from an aggregate
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float("nan")
        else:
            mean, sampleVariance = mean, M2 / (count - 1)
            std=sampleVariance**(0.5)
            return mean, std
        
    def returnCut(self):
        self.tuneCut=False
        
        cut0={}
        cut1={}
        means={}
        stds={}
            
        for layer in self.keys:
            means[layer],stds[layer],cut0[layer],cut1[layer]=[],[],[],[]
            #order: Z, E, En
            
            
            mean,std=self.finalizeWelford(self.aggregateE[layer])
            means[layer].append(mean)
            stds[layer].append(std)
            c0=np.maximum(mean/5,mean-3*std)
            c1=np.minimum(c0*25,mean+3*std)
            cut0[layer].append(c0)
            cut1[layer].append(c1)

        return cut0,cut1,means,stds
    
def Freeze(classifierDNN):
    history={}
    if ('ResNet' in classifierDNN.__class__.__name__):
        for name,param in classifierDNN.fc.named_parameters():
            history[name]=param.requires_grad
        for param in classifierDNN.parameters():
            param.requires_grad=False
        for name,param in classifierDNN.fc.named_parameters():
            param.requires_grad=history[name]
    elif ('DenseNet' in classifierDNN.__class__.__name__):
        for name,param in classifierDNN.classifier.named_parameters():
            history[name]=param.requires_grad
        for param in classifierDNN.parameters():
            param.requires_grad=False
        for name,param in classifierDNN.classifier.named_parameters():
            param.requires_grad=history[name]
    elif ('VGG' in classifierDNN.__class__.__name__):
        for name,param in classifierDNN.classifier[-1].named_parameters():
            history[name]=param.requires_grad
        for param in classifierDNN.parameters():
            param.requires_grad=False
        for name,param in classifierDNN.classifier[-1].named_parameters():
            param.requires_grad=history[name]
    else:
        raise ValueError('backbone not implemented for ImageNet pretraining')
    return classifierDNN
