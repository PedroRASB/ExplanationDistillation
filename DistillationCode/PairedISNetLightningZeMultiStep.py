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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class ISNetLgt(pl.LightningModule):
    def __init__(self,mode='conventional',
                 multiLabel=False,multiMask=False,
                 classes=1000,architecture='densenet121',
                 e=1e-2,heat=True,
                 Zb=True,pretrained=False,
                 LR=1e-3,P=0.7,E=10,
                 A=1,B=3,d=0.996,
                 cut=1,cut2=25,sumMaps=None,
                 cutF=1,cut2F=25,
                 norm=True,
                 saveMaps=False,
                 mapsLocation='',optim='SGD',
                 Cweight=None,rule='e',multiple=True,positive=False,
                 ignore=None,
                 dropLr=None, selective=False,
                 highest=False, baseModel=None,
                 dropout=False,
                 amplify=1,momentum=0.99,WD=0,
                 separate=False,
                 mask='none',
                 penalizeAll=False,detach=True,
                 FSP=False,dLoss=1,backwardCompat=False,
                 teacher=None,nonUniform=False,
                 curriculum=False,maxEpochs=50,curriculumEpochs=None,
                 squareCurriculum=False,noNoiseEpochs=10, zeroBias=False, randomLogit=False):
        #Creates ISNet based on DenseNet121, non instantiated
        
        #model parameters:
        #e: LRP-e term
        #Zb: allows Zb rule. If false, will use traditional LRP-e.
        #heat: allows relevance propagation and heatmap creation. If False,
        #no signal is propagated through LRP block.
        #pretrained: if a pretrained DenseNet121 shall be downloaded
        #classes: number of output classes
        #rule: LRP rule, choose e, z+, AB or z+e. For background relevance 
        #minimization use either e or z+e.
        #multiple: whether to produce a single heatmap or one per class
        #positive: whether to make output relevance positive or not
        #ignore: list with classes which will not suffer attention control
        #selective: uses class selective propagation. Only for explanation, not for ISNet training.
        #architecture: densenet,resnet
        #if not None: standard resnet or densenet to be converted
        #dropout:adds dropout before last layer
        #randomNoise: uses random noise instead of 0 as masks in Forge and Refine stages
        
        #training parameters:
        #LR:learning rate, list of tuples (epoch,lr) for scheduler
        #P: loss balancing hyperparameter. int or dictionbary, with epochs (int) and P (float)
        #E: heatmap loss hyperparameter
        #multiMask: for a segmentation mask per class
        #multiLabel: for multi-label problems
        #saveMaps: saves test hetamaps
        #optim: SGD or Adam
        #Cweight: BCE loss weights to deal with unbalanced datasets
        #validation loader should return dataset identifier (0=iid,1=ood) with each label
        #dropLr: if not None, list of tuples (epoch,lr) for scheduler
        #meanMaps: standard value for heatmap sums
        #separate: separatelly minimizes positive and negative relevance
        
        super (ISNetLgt,self).__init__()
        self.save_hyperparameters()
        
        if (ignore is not None and selective):
            raise ValueError('class ignore not implemented for selective output')
        
        self.getBackbone(architecture,baseModel,dropout,pretrained,classes)
        
        if ('DenseNet' in self.classifierDNN.__class__.__name__):
            self.LRPBlock=ISNetLayers._LRPDenseNet(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,positive=positive,
                                                   ignore=ignore,selective=selective,
                                                   highest=highest,detach=detach,
                                                   storeRelevance=penalizeAll,FSP=FSP,
                                                   backwardCompat=backwardCompat,
                                                   randomLogit=randomLogit)
        elif ('FixupResNet' in self.classifierDNN.__class__.__name__):                
            self.LRPBlock=ISNetLayers._LRPResNetFixUp(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,positive=positive,
                                                   ignore=ignore,selective=selective,
                                                   highest=highest,detach=detach,
                                                   amplify=amplify,
                                                   FSP=FSP,storeRelevance=penalizeAll,
                                                 randomLogit=randomLogit)
        elif ('ResNet' in self.classifierDNN.__class__.__name__):                
            self.LRPBlock=ISNetLayers._LRPResNet(self.classifierDNN,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,positive=positive,
                                                   ignore=ignore,selective=selective,
                                                   highest=highest,detach=detach,
                                                   amplify=amplify,
                                                   FSP=FSP,storeRelevance=penalizeAll,
                                                 randomLogit=randomLogit)
        else:
            raise ValueError('Unrecognized backbone')
            
        self.heat=heat
        self.lr=LR
        self.P=P
        self.E=E
        self.multiMask=multiMask
        self.multiLabel=multiLabel
        self.cut=cut
        self.cut2=cut2
        self.cutF=cutF
        self.cut2F=cut2F
        self.A=A
        self.B=B
        self.d=d
        self.Cweight=Cweight
        self.criterion=nn.CrossEntropyLoss()
        self.norm=norm
        self.sumMaps=sumMaps
        self.separate=separate
        self.penalizeAll=penalizeAll
        
        self.saveMaps=saveMaps
        self.mapsLocation=mapsLocation
        self.optim=optim
        self.rule=rule
        self.classes=classes
        self.dropLr=dropLr
        self.momentum=momentum
        self.WD=WD
        self.FSP=FSP
        self.dLoss=dLoss
        self.nonUniform=nonUniform
        self.curriculum=curriculum
        self.maxEpochs=maxEpochs
        self.curriculumEpochs=curriculumEpochs
        self.squareCurriculum=squareCurriculum
        self.noNoiseEpochs=noNoiseEpochs
        
        if isinstance(P, dict):
            self.P=P[0]
            self.increaseP=P
            
        else:
            self.P=P
            self.increaseP=None
            
        self.tuneCut=False
        
        self.mode=mode
        self.mask=mask
        
        if zeroBias:
            self.ZeroTheBias()
        
        #stage 1:
        if self.mode=='conventional':
            if mask !='none':
                print('Careful, using conventional and mask=',mask)
        elif self.mode=='forge':
            #must be zero, noise or recombine
            self.heat=False
            self.P=0
            if mask=='none':
                raise ValueError('Specify mask=zero, noise or recombine')
        elif self.mode=='refine':
            raise ValueError('Use student model')
            self.heat=True
            self.teacher=teacher
            if teacher is None:
                raise ValueError('Provide teacher network')
            if mask=='none':
                raise ValueError('Specify mask=zero, noise or recombine')
        else:
            raise ValueError('Unrecognized mode. Mode should be conventional for standard isnet training, forge for step 1 of multi-stage training, refine for step 2, harden for step 3 or polish for step 4')
            
    def ZeroTheBias(self):
        i=0
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = False
                param.data.fill_(0.0)
                i+=1
        print('Biases det to zero and frozen:', i)
        
    def forward(self,img,mask=None,img2=None,mask2=None):
        #img:input
        #if mask is not None, it will be used to erase background of img
        
        #mask image if requested
        if self.mask!='none':
            x=self.mask_image(img,mask,img2,mask2)
        else:
            x=img
        
        #forward
        y=self.classifierDNN(x)
        
        #LRP
        if(self.heat):#only run the LRP block if self.heat==True
            R=self.LRPBlock(x=x,y=y)
            return y,R
        else: 
            globals.X=[]
            globals.XI=[]
            return y

    def mask_image(self,img,mask,img2=None,mask2=None):
        #erase background (forge and refine stages)
        x=torch.mul(img,mask)
        if self.mask=='zero':
            pass
        elif self.mask=='mixed':
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            noise=torch.rand(x.shape).type_as(x)
            noise=torch.mul(Imask,noise)
            if self.training:
                percentNoise=0.5
            else:
                percentNoise=1
            ones=torch.ones((x.shape[0],1,1,1)).type_as(x)
            b=torch.bernoulli(ones*percentNoise)
            x=x+noise*b
        elif self.mask=='noise':
            #use uniform noise as masked images' backgrounds
            #Imask: 1 in background, 0 in foreground
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            noise=torch.rand(x.shape).type_as(x)
            noise=torch.mul(Imask,noise)
            #if (self.nonUniform and self.training):
            #    if self.curriculum:                  
            #        epoch=self.current_epoch-self.noNoiseEpochs#10 epochs with no noise
            #        beta=5
            #        alpha=5
            #        if epoch<=(self.curriculumEpochs/2):
            #            alpha=5*(epoch/((self.curriculumEpochs/2)+1e-10))
            #        else:
            #            count=epoch-(self.curriculumEpochs/2)
            #            beta=5-5*(count/(self.curriculumEpochs/2))
            #    else:
            #        alpha=1
            #        beta=3 
            #        
            #    if self.curriculum and (self.current_epoch<self.noNoiseEpochs):
            #        magnitude=0
            #    elif alpha<=0:
            #        magnitude=0
            #    elif beta<=0:
            #        magnitude=1
            #    else:
            #        size=img.shape[0]
            #        magnitude=torch.from_numpy(np.random.beta(alpha, beta, size))\
            #        .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).type_as(img)
                    
            ##    noise=noise*magnitude
            #if (self.squareCurriculum and self.training):
            #    magnitude=(self.current_epoch**2)/(self.curriculumEpochs**2)
            #    if magnitude>1:
            #        magnitude=1
            #    noise=noise*magnitude
            #x=x+noise
            
            if self.curriculum and self.training:
                if self.current_epoch<=self.noNoiseEpochs:
                    pass
                elif self.current_epoch>self.curriculumEpochs:
                    x=x+noise
                else:#add noise to part of the images
                    percentMasked=(self.current_epoch-self.noNoiseEpochs)/\
                    (self.curriculumEpochs-self.noNoiseEpochs)
                    ones=torch.ones((x.shape[0],1,1,1)).type_as(x)
                    b=torch.bernoulli(ones*percentMasked)
                    x=x+noise*b
            else:
                x=x+noise
                    
        elif self.mask=='recombine':
            #segment auxiliary (img2) image background
            Imask2=torch.ones(mask2.shape).type_as(mask2)-mask2
            aux=torch.mul(img2,Imask2)

            #paint the balck foreground in aux with the backgroud mean color
            mean=aux.sum(dim=(-1,-2),keepdim=True)/(Imask2.sum(dim=(-1,-2),keepdim=True)+1e-10)
            aux=aux+torch.mul(mask2,mean)

            #remove img foreground from aux
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            aux=torch.mul(aux,Imask)

            #put img object over aux
            x=x+aux
        else:
            raise ValueError('Unrecognized mask argument')
        return x
                
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
                                    weight_decay=self.WD)
        
        if(self.dropLr is not None):
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.dropLr,
                                                           verbose=True)            
            return [optimizer],[scheduler]
        else:
            return optimizer
    
    def compound_loss(self,outputs,labels,heatmaps=None,masks=None):
        if(self.multiLabel):
            classifierLoss=F.binary_cross_entropy_with_logits(outputs,labels,
                                                              pos_weight=self.Cweight)
        else:
            #classifierLoss=F.cross_entropy(outputs,labels.squeeze(1))
            #print(labels)
            #print(labels.shape)
            try:
                classifierLoss=self.criterion(outputs,labels.squeeze(1))
            except:
                classifierLoss=self.criterion(outputs,labels)
        if not self.heat or masks is None:
            return classifierLoss
        
        
        LRPs={'input':heatmaps}
        #penalize heatmaps for multiple layers:
        if self.penalizeAll and not self.FSP:
            ISNetFunctions.getRelevance(self.LRPBlock.storedRelevance,LRPs,'')
        elif self.FSP:
            for key in self.LRPBlock.storedRelevance:
                    LRPs[key]=self.LRPBlock.storedRelevance[key].output
        self.keys=list(LRPs.keys())
        losses=[]
        tune={}
        for key in LRPs:
            if not self.separate:
                if (self.heat and self.tuneCut):
                    heatmapLoss,foreg=ISNetFunctions.LRPLossCEValleysGWRP(LRPs[key],masks,
                                                                   A=self.A,B=self.B,d=self.d,
                                                                   E=self.E,
                                                                   rule=self.rule,
                                                                   tuneCut=self.tuneCut,
                                                                   norm=self.norm,
                                                                   sumMaps=self.sumMaps)
                    losses.append(heatmapLoss)
                    tune[key]=foreg

                if (self.heat and not self.tuneCut):
                    heatmapLoss=ISNetFunctions.LRPLossCEValleysGWRP(LRPs[key],masks,
                                                                   cut=self.cut[key],
                                                                   cut2=self.cut2[key],
                                                                   A=self.A,B=self.B,d=self.d,
                                                                   E=self.E,
                                                                   rule=self.rule,
                                                                   tuneCut=self.tuneCut)
                    losses.append(heatmapLoss)
            else:
                if (self.heat and self.tuneCut):
                    heatmapLoss,foreg=ISNetFunctions.LRPLossGWRPSeparate(LRPs[key],masks,
                                                                   A=self.A,B=self.B,d=self.d,
                                                                   E=self.E,
                                                                   rule=self.rule,
                                                                   tuneCut=self.tuneCut)
                    losses.append(heatmapLoss)
                    une[key]=foreg
                    

                if (self.heat and not self.tuneCut):
                    heatmapLoss=ISNetFunctions.LRPLossGWRPSeparate(LRPs[key],masks,
                                                                   cut=self.cut[key],
                                                                   cut2=self.cut2[key],
                                                                   A=self.A,B=self.B,d=self.d,
                                                                   E=self.E,
                                                                   rule=self.rule,
                                                                   tuneCut=self.tuneCut)
                    losses.append(heatmapLoss)
                    
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=ISNetFunctions.GlobalWeightedRankPooling(heatmapLoss,d=self.dLoss)
        #heatmapLoss=torch.mean(heatmapLoss,dim=-1)
        if not self.tuneCut:
            return classifierLoss,heatmapLoss
        else:
            return classifierLoss,heatmapLoss,tune
        
    def training_step(self,train_batch,batch_idx):
        
        #update dinamic P
        if (self.increaseP is not None):
            epochs=list(self.increaseP.keys())
            epochs.sort()
            for epoch in epochs:
                if (self.current_epoch>=epoch):
                    self.P=self.increaseP[epoch]
        #get inputs
        if self.mask=='recombine':
            inputs,masks,labels,inputs2,masks2,_=train_batch
        else:
            inputs2=None
            masks2=None
            try:
                inputs,masks,labels=train_batch
            except:
                inputs,labels=train_batch
                masks=None
            
        #set parameters
        oldHeat=None
        if self.tuneCut:
            if(self.current_epoch!=self.cutEpochs):
                oldHeat=self.heat
                self.heat=False
            else:
                oldHeat=self.heat
                self.heat=True

        #forward pass 
        if not self.heat:
            logits=self.forward(inputs,masks,inputs2,masks2)
            loss=self.compound_loss(logits,labels=labels)
            self.log('train_loss',loss, on_epoch=True)
        else:
            logits,heatmaps=self.forward(inputs,masks,inputs2,masks2)
            
            if self.tuneCut:
                cLoss,hLoss,mapAbs=self.compound_loss(logits,labels=labels,
                                           heatmaps=heatmaps,masks=masks)
                #take only values from last tuning epoch
                self.updateCut(mapAbs)
                #use only for tuning cut value, ignores heatmap loss
                loss=cLoss
                self.log('train_loss',loss, on_epoch=True)
            else:
                cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                               heatmaps=heatmaps,masks=masks)
                #hLoss and fLoss do not directly compete, unlike cLoss
                loss=(1-self.P)*cLoss+self.P*hLoss

                self.log('train_loss', {'Classifier':cLoss,
                                        'Heatmap':hLoss,
                                        'Sum':loss},                     
                         on_epoch=True)

        #restore parameters
        if oldHeat is not None:
            self.heat=oldHeat
                
        #avoid destroyning DNN by passing a nan value
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')
            
        return loss
    
    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        #set parameters
        oldHeat=None
        if (self.tuneCut or (dataloader_idx==1 and \
                             (self.mode=='conventional' or self.mode=='polish') \
                             and self.mask=='none')):
            oldHeat=self.heat
            self.heat=False
            
        #get inputs
        if self.mask=='recombine':
            inputs,masks,labels,inputs2,masks2,_=val_batch
        else:
            inputs2=None
            masks2=None
            try:
                inputs,masks,labels=val_batch
            except:
                inputs,labels=val_batch
                masks=None
            
        #forward pass
        if (self.heat):#ISNet
            logits,heatmaps=self.forward(inputs,masks,inputs2,masks2)
            cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                               heatmaps=heatmaps,masks=masks)
            #hLoss and fLoss do not directly compete, unlike cLoss
            loss=(1-self.P)*cLoss+self.P*hLoss
        else:#Common DenseNet
            logits=self.forward(inputs,masks,inputs2,masks2)
            loss=self.compound_loss(logits,labels=labels)
            
        if oldHeat is not None:
            self.heat=oldHeat
            
        if dataloader_idx==0:
            return {'iidLoss':loss}
        if dataloader_idx==1:
            return {'oodLoss':loss}

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
        #get inputs
        tmp=None
        if self.mask=='recombine':
            try:
                inputs,masks,labels,inputs2,masks2,_=test_batch
            except:
                #evaluation with standard images (no recombination)
                inputs,labels=test_batch
                masks=None
                inputs2=None
                masks2=None
                tmp=self.mask
                self.mask='none'
                
        else:
            inputs2=None
            masks2=None
            try:
                inputs,masks,labels=test_batch
            except:
                inputs,labels=test_batch
                tmp=self.mask
                self.mask='none'
                masks=None
        
        
        if (self.heat):
            logits,heatmaps=self.forward(inputs,masks,inputs2,masks2)
            if tmp is not None:
                self.mask=tmp
            return {'pred': logits, 'labels': labels, 'images': inputs.cpu().float(),
                    'heatmaps': heatmaps.cpu().float()}

        else:
            logits=self.forward(inputs,masks,inputs2,masks2)
            if tmp is not None:
                self.mask=tmp
            return {'pred': logits, 'labels': labels}
            

    def test_step_end(self, batch_parts):
        if(batch_parts['pred'].dim()>2):
            logits=batch_parts['pred']
            labels=batch_parts['labels']
            if (not self.heat):
                return {'pred': logits.view(logits.shape[0]*logits.shape[1],logits.shape[-1]),
                        'labels': labels.view(labels.shape[0]*labels.shape[1],labels.shape[-1])}
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
        if (self.heat):
            images=test_step_outputs[0]['images']
            heatmaps=test_step_outputs[0]['heatmaps']
            
        for i,out in enumerate(test_step_outputs,0):
            if (i!=0):
                pred=torch.cat((pred,out['pred']),dim=0)
                labels=torch.cat((labels,out['labels']),dim=0)
                if (self.heat):
                    images=torch.cat((images,out['images']),dim=0)
                    heatmaps=torch.cat((heatmaps,out['heatmaps']),dim=0)
                
        if (self.heat):
            self.TestResults=pred,labels,images,heatmaps
        else:
            self.TestResults=pred,labels
            
    def returnBackbone(self):
        model=self.classifierDNN
        ISNetFunctions.remove_all_forward_hooks(model)
        return model
    
    #Functions for automatic tunning of CUT values in the heatmap loss
    def initTuneCut(self,epochs):
        #train for self.cutEpochs to find cut values, do not use heatmap loss
        self.tuneCut=True
        self.cutEpochs=epochs-1
            
    def resetCut(self):
        self.aggregateE={}
        for name in self.keys:
            self.aggregateE[name]=[0,0,0]

        if self.rule=='z+e':
            self.aggregateZ={}
            for name in self.keys:
                self.aggregateZ[name]=[0,0,0]
                    
        if self.separate:
            self.aggregateEn={}
            for name in self.keys:
                self.aggregateEn[name]=[0,0,0]
            
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
            if self.separate:
                if self.rule=='e':
                    mapAbsE,mapAbsEn=mapAbs[:,0],mapAbs[:,1]
                elif self.rule=='z+e':
                    mapAbsZ,mapAbsE,mapAbsEn=mapAbs[:,0],mapAbs[:,1],mapAbs[:,2]
            else:
                if self.rule=='z+e':
                    mapAbsZ=mapAbs[:,:int(mapAbs.shape[1]/2)]
                    mapAbsE=mapAbs[:,int(mapAbs.shape[1]/2):]
                else:
                    mapAbsE=mapAbs

            for i,_ in enumerate(mapAbsE,0):#batch iteration
                valueE=mapAbsE[i].detach().float().item()
                self.aggregateE[layer]=self.updateWelford(self.aggregateE[layer],valueE)

                if self.rule=='z+e':
                    valueZ=mapAbsZ[i].detach().float().item()
                    self.aggregateZ[layer]=self.updateWelford(self.aggregateZ[layer],valueZ)

                if self.separate:
                    valueEn=mapAbsEn[i].detach().float().item()
                    self.aggregateEn[layer]=self.updateWelford(self.aggregateEn[layer],valueEn)


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
            
            if self.rule=='z+e':
                mean,std=self.finalizeWelford(self.aggregateZ[layer])
                means[layer].append(mean)
                stds[layer].append(std)
                c0=np.maximum(mean/5,mean-3*std)
                c1=np.minimum(c0*25,mean+3*std)
                cut0[layer].append(c0)
                cut1[layer].append(c1)
                
            
            mean,std=self.finalizeWelford(self.aggregateE[layer])
            means[layer].append(mean)
            stds[layer].append(std)
            c0=np.maximum(mean/5,mean-3*std)
            c1=np.minimum(c0*25,mean+3*std)
            cut0[layer].append(c0)
            cut1[layer].append(c1)

            if self.separate:
                mean,std=self.finalizeWelford(self.aggregateEn[layer])
                means[layer].append(mean)
                stds[layer].append(std)
                c0=np.maximum(mean/5,mean-3*std)
                c1=np.minimum(c0*25,mean+3*std)
                cut0[layer].append(c0)
                cut1[layer].append(c1)
        return cut0,cut1,means,stds

    def getBackbone(self,architecture,baseModel,dropout,pretrained,classes):
        if (baseModel==None):
            if (architecture=='densenet121'):
                self.classifierDNN=LRPDenseNet.densenet121(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet161'):
                self.classifierDNN=LRPDenseNet.densenet161(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet169'):
                self.classifierDNN=LRPDenseNet.densenet169(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet201'):
                self.classifierDNN=LRPDenseNet.densenet201(pretrained=pretrained,
                                                      num_classes=classes)
            elif (architecture=='densenet264'):
                if(pretrained):
                    raise ValueError('No available pretrained densenet264')
                self.classifierDNN=LRPDenseNet.densenet264(pretrained=False,
                                                      num_classes=classes)
            elif (architecture=='resnet18'):
                import resnet
                self.classifierDNN=resnet.resnet18(pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet34'):
                import resnet
                self.classifierDNN=resnet.resnet34(pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet50'):
                import resnet
                self.classifierDNN=resnet.resnet50(pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet101'):
                import resnet
                self.classifierDNN=resnet.resnet101(pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet152'):
                import resnet
                self.classifierDNN=resnet.resnet152(pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet18FixUpZeroBias'):
                import ZeroBiasResNetFixUp
                if pretrained:
                    raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
                self.classifierDNN=ZeroBiasResNetFixUp.fixup_resnet18(num_classes=classes)
            elif (architecture=='resnet34FixUpZeroBias'):
                import ZeroBiasResNetFixUp
                self.classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                             'resnet34', pretrained=pretrained,
                                             num_classes=classes)
            elif (architecture=='resnet50FixUpZeroBias'):
                import ZeroBiasResNetFixUp
                if pretrained:
                    raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
                self.classifierDNN=ZeroBiasResNetFixUp.fixup_resnet50(num_classes=classes)
            elif (architecture=='resnet101FixUpZeroBias'):
                import ZeroBiasResNetFixUp
                if pretrained:
                    raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
                self.classifierDNN=ZeroBiasResNetFixUp.fixup_resnet101(num_classes=classes)
            elif (architecture=='resnet152FixUpZeroBias'):
                import ZeroBiasResNetFixUp
                if pretrained:
                    raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
                self.classifierDNN=ZeroBiasResNetFixUp.fixup_resnet152(num_classes=classes)
            
            
            else:
                raise ValueError('Architecture must be densenet121, 161, 169, 201 or 204; or resnet18, 34, 50, 101 or 152; or  wideResnet28. User may also supply a DenseNet or ResNet as baseModel.')
        
        else:
            self.classifierDNN=baseModel
            
        if dropout:
            if ('DenseNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.classifier=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                       self.classifierDNN.classifier)
            elif ('ResNet' in self.classifierDNN.__class__.__name__):
                self.classifierDNN.fc=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                       self.classifierDNN.fc)
            else:
                raise ValueError('Unrecognized backbone')
                
        if (('ResNet' in self.classifierDNN.__class__.__name__) and
            hasattr(self.classifierDNN, 'maxpool')):
                self.classifierDNN.maxpool.return_indices=True
                self.classifierDNN.maxpool=nn.Sequential(
                    OrderedDict([('maxpool',self.classifierDNN.maxpool),
                                 ('special',ISNetLayers.IgnoreIndexes())]))


class ISNetStudentLgt(pl.LightningModule):
    def __init__(self,teacher,mask='noise',multiLabel=False,multiMask=False,
                 e=1e-2,heat=True,
                 Zb=True,
                 LR=1e-3,
                 alpha=1,beta=1,gamma=1,
                 norm=True,optim='SGD',
                 Cweight=None,rule='e',multiple=True,positive=False,
                 ignore=None,
                 dropLr=None, selective=False,
                 highest=False, 
                 amplify=1,momentum=0.99,WD=0,
                 detach=True,
                 penalizeAll=False,
                 FSP=False,dLoss=1,
                 randomLogit=False,
                 SequentialInputShape=None,SequentialPreFlattenShape=None,
                 channelGWRP=1.0,nesterov=False,
                 huber=False,maskedTest=False,
                 percentageMasked=0.5,beginAsTeacher=True,
                 zeroBias=False):
        
        super (ISNetStudentLgt,self).__init__()
        self.save_hyperparameters()
        
        if (ignore is not None and selective):
            raise ValueError('class ignore not implemented for selective output')
        
        try:
            self.teacher=teacher.returnBackbone()
        except:
            self.teacher=teacher
            

        #LRp block
        if ('DenseNet' in self.teacher.__class__.__name__):
            self.teacher.LRPBlock=ISNetLayers._LRPDenseNet(self.teacher,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,positive=positive,
                                                   ignore=ignore,selective=selective,
                                                   highest=highest,detach=detach,
                                                   storeRelevance=penalizeAll,
                                                   FSP=FSP,randomLogit=randomLogit)
        elif ('FixupResNet' in self.teacher.__class__.__name__):                
            self.LRPBlock=ISNetLayers._LRPResNetFixUp(self.teacher,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,positive=positive,
                                                   ignore=ignore,selective=selective,
                                                   highest=highest,detach=detach,
                                                   amplify=amplify,
                                                   FSP=FSP,storeRelevance=penalizeAll,
                                                 randomLogit=randomLogit)
        elif ('ResNet' in self.teacher.__class__.__name__):
            #if FSP:
            #    raise ValueError('not implemented')
            if (hasattr(self.teacher, 'maxpool') and \
            self.teacher.maxpool.__class__.__name__=='MaxPool2d'):
                self.teacher.maxpool.return_indices=True
                self.teacher.maxpool=nn.Sequential(OrderedDict([('maxpool',
                                                                 self.teacher.maxpool),
                                                    ('special',ISNetLayers.IgnoreIndexes())]))
            self.teacher.LRPBlock=ISNetLayers._LRPResNet(self.teacher,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,positive=positive,
                                                   ignore=ignore,selective=selective,
                                                   highest=highest,
                                                   amplify=amplify,detach=detach,
                                                   storeRelevance=penalizeAll,
                                                   FSP=FSP,randomLogit=randomLogit)
        elif ('Sequential' in self.teacher.__class__.__name__):
            self.teacher.LRPBlock=ISNetLayers._LRPSequential(self.teacher,e=e,Zb=Zb,rule=rule,
                                                   multiple=multiple,selective=selective,
                                                   highest=highest,
                                                   amplify=amplify,detach=detach,
                                                   storeRelevance=penalizeAll,
                                                   inputShape=SequentialInputShape,
                                                   preFlattenShape=SequentialPreFlattenShape,
                                                   randomLogit=randomLogit)

        elif('VGG' in self.teacher.__class__.__name__):
            if (positive or ignore):
                raise ValueError('not implemented')
            self.teacher.LRPBlock=ISNetLayers._LRPVGG(self.teacher,e=e,Zb=Zb,rule=rule,
                                              multiple=multiple,selective=selective,
                                              highest=highest,
                                              amplify=amplify,detach=detach,
                                              randomLogit=randomLogit,
                                              storeRelevance=penalizeAll)

        else:
            raise ValueError('Unrecognized backbone')
                
        
        self.student=copy.deepcopy(teacher)
        print('COPIED TEACHER AS STUDENT')
        if not beginAsTeacher:
            for layer in self.student.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            print('PARAMETERS RESET')
            
            
        #Forward hook to get last feature map
        #print(self.teacher)
        if ('DenseNet' in self.teacher.__class__.__name__):
            self.featureMapT=ISNetFunctions.HookFwd(self.teacher.features.fReLU,mode='output')
            self.featureMapS=ISNetFunctions.HookFwd(self.student.features.fReLU,mode='output')
        elif ('ResNet' in self.teacher.__class__.__name__):
            self.featureMapT=ISNetFunctions.HookFwd(self.teacher.layer4,mode='output')
            self.featureMapS=ISNetFunctions.HookFwd(self.student.layer4,mode='output')
        elif('VGG' in self.teacher.__class__.__name__):
            self.featureMapT=ISNetFunctions.HookFwd(self.teacher.features,mode='output')
            self.featureMapS=ISNetFunctions.HookFwd(self.student.features,mode='output')
        elif ('Sequential' in self.teacher.__class__.__name__):#get last convolutional map
            last_conv_layer = None
            for name, layer in self.teacher.named_children():
                if isinstance(layer, torch.nn.Conv2d):
                    last_conv_layer = name
            self.featureMapT=ISNetFunctions.HookFwd(getattr(self.teacher,
                                                           last_conv_layer),mode='output')
            self.featureMapS=ISNetFunctions.HookFwd(getattr(self.student,
                                                           last_conv_layer),mode='output')
            
        #freeze teacher:
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        for param in self.student.parameters():
            param.requires_grad = True
            
        #freeze student classifier layer
        #if ('DenseNet' in self.student.__class__.__name__):
        #    for param in self.student.classifier:
        #        param.requires_grad = False
        #elif ('ResNet' in self.student.__class__.__name__):
        #    for param in self.student.fc:
        #        param.requires_grad = False
        #elif ('vgg' in self.student.__class__.__name__):
        #    for param in self.student.classifier[-1]:
        #        param.requires_grad = False
        #elif ('Sequential' in self.student.__class__.__name__):
        #    for param in self.student[-1]:
        #        param.requires_grad = False
        #else:
        #    raise ValueError('Unrecognized backbone')
        if zeroBias:
            self.ZeroTheBias()
        
        self.heat=heat
        self.lr=LR
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.Cweight=Cweight
        self.norm=norm
        
        self.optim=optim
        self.rule=rule
        self.dropLr=dropLr
        self.momentum=momentum
        self.WD=WD
        self.penalizeAll=penalizeAll
        self.FSP=FSP
        self.dLoss=dLoss
        self.keys={}
        self.testLoss=False
        self.nesterov=nesterov
        self.huber=huber
        self.maskedTest=maskedTest
        self.mask=mask
        self.multiLabel=multiLabel
        self.multiMask=multiMask
        self.criterion=nn.CrossEntropyLoss()
        self.percentageMasked=percentageMasked
        self.lastEpoch=0
        
        #[drop (percentage), interval (epochs), minimum]
        if isinstance(self.percentageMasked, list):
            self.currentPercentage=1
            self.counterPercentage=0
            self.dropPercentage=self.percentageMasked[0]
            self.intervalPercentage=self.percentageMasked[1]
            self.minimumPercentage=self.percentageMasked[2]
        
    
    def ZeroTheBias(self):
        i=0
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = False
                param.data.fill_(0.0)
                i+=1
        print('Biases det to zero and frozen:', i)
    
    def mask_image(self,img,mask,img2=None,mask2=None):
        #erase background (forge and refine stages)
        x=torch.mul(img,mask)
        if self.mask=='zero':
            pass
        elif self.mask=='noise':
            #use uniform noise as masked images' backgrounds
            #Imask: 1 in background, 0 in foreground
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            noise=torch.rand(x.shape).type_as(x)
            noise=torch.mul(Imask,noise)
            x=x+noise
        elif self.mask=='recombine':
            #segment auxiliary (img2) image background
            Imask2=torch.ones(mask2.shape).type_as(mask2)-mask2
            aux=torch.mul(img2,Imask2)

            #paint the balck foreground in aux with the backgroud mean color
            mean=aux.sum(dim=(-1,-2),keepdim=True)/(Imask2.sum(dim=(-1,-2),keepdim=True)+1e-10)
            aux=aux+torch.mul(mask2,mean)

            #remove img foreground from aux
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            aux=torch.mul(aux,Imask)

            #put img object over aux
            x=x+aux
        else:
            raise ValueError('Unrecognized mask argument')
        return x
    
    def forward(self,x,mask=None):
        globals.X=[]
        globals.XI=[]
        #x:input
        
        y=self.student(x)
        if mask is not None:
            features=self.featureMapS.x.clone()
        
        if(self.heat):#only run the LRP block if self.heat==True
            R=self.student.LRPBlock(x=x,y=y)
            self.LRPs={'input':R}#.clone()}
            #penalize heatmaps for multiple layers:
            if self.penalizeAll and not self.FSP:
                ISNetFunctions.getRelevance(self.student.LRPBlock.storedRelevance,
                                            self.LRPs,'',clone=False)#True)
            elif self.FSP:
                for key in self.student.LRPBlock.storedRelevance:
                        self.LRPs[key]=self.student.LRPBlock.storedRelevance[key].output#.clone()
                        
        if mask is not None:
            #teacher
            with torch.no_grad():
                globals.X=[]
                globals.XI=[]
                masked=self.mask_image(x,mask)
                yMask=self.teacher(masked)
                featuresMask=self.featureMapT.x.detach().clone()

                if(self.heat):     
                    RMask=self.teacher.LRPBlock(x=masked,y=yMask)
                    self.LRPsMask={'input':RMask.detach()}#.clone()}
                    #penalize heatmaps for multiple layers:
                    if self.penalizeAll and not self.FSP:
                        ISNetFunctions.getRelevance(self.teacher.LRPBlock.storedRelevance,
                                                    self.LRPsMask,'',detach=False,clone=False)
                    elif self.FSP:
                        for key in self.teacher.LRPBlock.storedRelevance:
                                self.LRPsMask[key]=\
                                self.teacher.LRPBlock.storedRelevance[key].output.detach()#.clone()
            
        if mask is None:#cut tune, test, val
            if(self.heat):    
                return y,R
            else: 
                return y
        else:#teacher student
            if(self.heat):#only run the LRP block if self.heat==True
                return y,yMask,R,RMask,features,featuresMask
            else: 
                return y,yMask,features,featuresMask

    def configure_optimizers(self):
        #freeze teacher:
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.student.parameters():
            param.requires_grad = True
        #freeze student classifier layer
        #if ('DenseNet' in self.student.__class__.__name__):
        #    for param in self.student.classifier:
        #        param.requires_grad = False
        #elif ('ResNet' in self.student.__class__.__name__):
        #    for param in self.student.fc:
        #        param.requires_grad = False
        #elif ('vgg' in self.student.__class__.__name__):
        #    for param in self.student.classifier[-1]:
        #        param.requires_grad = False
        #elif ('Sequential' in self.student.__class__.__name__):
        #    for param in self.student[-1]:
        #        param.requires_grad = False
        #else:
        #    raise ValueError('Unrecognized backbone')
        
        if (self.optim=='Adam'):
            from deepspeed.ops.adam import FusedAdam
            optimizer=FusedAdam(filter(
                    lambda p: p.requires_grad,
                                        self.parameters()),
                                        lr=self.lr)
            
        else:
            #for param in self.parameters():
            #    if param.requires_grad:
            #        print('grad:',param)
            optimizer=torch.optim.SGD(filter(
                lambda p: p.requires_grad,
                                    self.parameters()),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.WD,
                                    nesterov=self.nesterov)

        if(self.dropLr is not None):
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.dropLr,
                                                           verbose=True)            
            return [optimizer],[scheduler]
        else:
            return optimizer
    
    def compound_loss(self,outputsMask,labels,featuresMask=None,heatmapsMask=None,
                      outputs=None,features=None,heatmaps=None,masks=None):
        
        #pair logits        
        logitsLoss=torch.nn.functional.mse_loss(outputs,outputsMask.detach())
        #if(self.multiLabel):
        #    classifierLossMask=F.binary_cross_entropy_with_logits(outputsMask,labels,
        #                                                      pos_weight=self.Cweight)
        #else:
            #classifierLoss=F.cross_entropy(outputs,labels.squeeze(1))
        #    try:
        #        classifierLossMask=self.criterion(outputsMask,labels.squeeze(1))
        #    except:
        #        classifierLossMask=self.criterion(outputsMask,labels)
                
        
        #if (features is None and (not self.heat or masks is None)):
        #    return classifierLossMask
                
        if features is not None:
            #feature map pairing loss
            distillLoss=torch.nn.functional.mse_loss(features,featuresMask.detach())
            if not self.heat:
                return logitsLoss,distillLoss
        
        #LRPs={'input':heatmaps}
        #penalize heatmaps for multiple layers:
        #if self.penalizeAll and not self.FSP:
        #    ISNetFunctions.getRelevance(self.LRPBlock.storedRelevance,LRPs,'')
        #elif self.FSP:
        #    for key in self.LRPBlock.storedRelevance:
        #            LRPs[key]=self.LRPBlock.storedRelevance[key].output
        
        self.keys=list(self.LRPs.keys())
        losses=[]
        tune={}
        for key in self.LRPs:
            heatmapLoss=ISNetFunctions.PairLRPLoss(heatmap=self.LRPs[key],
                                                   heatmapMask=self.LRPsMask[key],
                                                   huber=self.huber,normMask=self.norm)
            losses.append(heatmapLoss)
                    
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=ISNetFunctions.GlobalWeightedRankPooling(heatmapLoss,d=self.dLoss)
        #heatmapLoss=torch.mean(heatmapLoss,dim=-1)
        return logitsLoss,distillLoss,heatmapLoss

    
    def training_step(self,train_batch,batch_idx):
        inputs,masks,labels=train_batch
        
        
        #can be a contant number or dropping
        #constant: float
        if isinstance(self.percentageMasked, float):
            percentageMasked=self.percentageMasked
        else:
            percentageMasked=self.currentPercentage
            
            #check for epoch change:
            if self.lastEpoch!=self.current_epoch:
                #update percentage:
                self.counterPercentage+=1
                if self.counterPercentage>=self.intervalPercentage:
                    self.counterPercentage=0
                    if (self.currentPercentage-self.dropPercentage)>=self.minimumPercentage:
                        #print('hi')
                        self.currentPercentage=self.currentPercentage-self.dropPercentage
                #print(self.currentPercentage,percentageMasked,self.counterPercentage,
                #      self.intervalPercentage,self.minimumPercentage,
                #      self.dropPercentage)
                self.lastEpoch=self.current_epoch

        if percentageMasked>0:
            tmp=masks.clone()
            for i,_ in enumerate(tmp,0):
                if torch.rand(1).item()>percentageMasked:
                    tmp[i]=torch.ones(tmp[i].shape).type_as(tmp[i])#itens that will NOT be masked
            inputs=self.mask_image(inputs,tmp)#mask self.percentageMasked% of itens
            
        #data format: channel first
        if (self.heat):#ISNet
            logits,logitsMask,heatmaps,heatmapsMask,features,featuresMask=\
            self.forward(inputs,mask=masks)
            logitsLoss,feraturesLoss,heatmapLoss=self.compound_loss(outputsMask=logitsMask,
                                                  labels=labels,
                                                  featuresMask=featuresMask,
                                                  outputs=logits,features=features,
                                                  heatmaps=heatmaps,heatmapsMask=heatmapsMask)
            loss=self.alpha*logitsLoss+self.beta*feraturesLoss+self.gamma*heatmapLoss

            self.log('train_loss', {'Logit':logitsLoss,
                                    'Feature':feraturesLoss,
                                    'Heatmap':heatmapLoss,
                                    'Sum':loss},                     
                     on_epoch=True)

        else:#Common DenseNet
            logits,logitsMask,features,featuresMask=self.forward(inputs,mask=masks)
            logitsLoss,feraturesLoss=self.compound_loss(outputsMask=logitsMask,
                                           labels=labels,
                                           featuresMask=featuresMask,
                                           outputs=logits,features=features)
            loss=self.alpha*logitsLoss+self.beta*feraturesLoss
            self.log('train_loss', {'Logit':logitsLoss,
                                    'Feature':feraturesLoss,
                                    'Sum':loss},                     
                     on_epoch=True)
        if(torch.isnan(loss).any()):
            raise ValueError('NaN Training Loss')
        return loss
    
    def validation_step(self,val_batch,batch_idx,dataloader_idx=0):
        if dataloader_idx==1:
            tmp=self.heat
            self.heat=False
        #data format: channel first
        if dataloader_idx==0:
            inputs,masks,labels=val_batch
        else:
            inputs,labels=val_batch
            
        
        if dataloader_idx==1:
            logits=self.forward(inputs)
            #ood eval, take only classification loss, no mask
            if(self.multiLabel):
                loss=F.binary_cross_entropy_with_logits(logits,labels,pos_weight=self.Cweight)
            else:
                #classifierLoss=F.cross_entropy(outputs,labels.squeeze(1))
                try:
                    loss=self.criterion(logits,labels.squeeze(1))
                except:
                    loss=self.criterion(logits,labels)
        else:
            if (self.heat):#ISNet
                logits,logitsMask,heatmaps,heatmapsMask,features,featuresMask=\
                self.forward(inputs,mask=masks)
                logitsLoss,feraturesLoss,heatmapLoss=self.compound_loss(outputsMask=logitsMask,
                                                  labels=labels,
                                                  featuresMask=featuresMask,
                                                  outputs=logits,features=features,
                                                  heatmaps=heatmaps,heatmapsMask=heatmapsMask)
                loss=self.alpha*logitsLoss+self.beta*feraturesLoss+self.gamma*heatmapLoss
            else:
                logits,logitsMask,features,featuresMask=self.forward(inputs,mask=masks)
                logitsLoss,feraturesLoss=self.compound_loss(outputsMask=logitsMask,
                                           labels=labels,
                                           featuresMask=featuresMask,
                                           outputs=logits,features=features)
                loss=self.alpha*logitsLoss+self.beta*feraturesLoss

        if dataloader_idx==0:
            return {'iidLoss':loss}
        if dataloader_idx==1:
            self.heat=tmp
            return {'oodLoss':loss}

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
        
        if self.testLoss:
            raise ValueError('Not implemented')
            if self.heat:
                inputs,masks,labels=test_batch
                logits,heatmaps=self.forward(inputs)
                cLoss,hLoss=self.compound_loss(logits,labels=labels,
                                               heatmaps=heatmaps,masks=masks)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(), 'hLoss': hLoss.detach()}
            else:
                inputs,labels=test_batch
                logits=self.forward(inputs)
                cLoss=self.compound_loss(logits,labels=labels)
                return {'pred': logits.detach(), 'labels': labels,
                        'cLoss': cLoss.detach(),
                        'hLoss': torch.zeros(cLoss.shape).type_as(cLoss)}
        elif (self.heat):#ISNet
            inputs,masks,labels=test_batch
            if self.maskedTest:
                inputs=self.mask_image(inputs,masks)
            logits,heatmaps=self.forward(inputs)
            return {'pred': logits.detach(), 'labels': labels,
                    'images': inputs.cpu().float().detach(),
                    'heatmaps': heatmaps.cpu().float().detach()}

        else:#Common DenseNet
            if self.maskedTest:
                inputs,masks,labels=test_batch
                inputs=self.mask_image(inputs,masks)
            else:
                inputs,labels=test_batch
            logits=self.forward(inputs)
            return {'pred': logits, 'labels': labels}
            

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
            
    def returnBackbone(self):
        model=self.student
        delattr(model,'LRPBlock')
        ISNetFunctions.remove_all_forward_hooks(model)
        return model