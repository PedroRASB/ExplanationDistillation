#new
import torch
import torch.nn.functional as F
import torch.nn as nn
import ISNetLayersZe as ISNetLayers
import ISNetFunctionsZe as ISNetFunctions
import globalsZe as globals
import warnings
from collections import OrderedDict
import numpy as np
import copy
from torch.autograd import Variable
import torch.autograd as ag
import random
import warnings
import math
try:
    import clip
except:
    pass


def LRPPair(LRPS,LRPT,
            huber=False,L1=False,norm=False,
            normTarget=False,CELRP=False,KLDivLRP=False,
            reduction='mean',stdNorm=False,normPerBatch=False,
            detachTeacher=True,
            pyramidLoss=False,minSize=8,ratio=2,
            dPyramid=1,dLayers=1,GWRPDescending=True,GWRPRank=True,
            simpleCurriculum=False,geoStd=False,scale=1,
            mask=None,loss='MSE',LRPlossOnFeatures=False,
            maskTargetLRP=False, basketEps=0.3):
    
    if not isinstance(LRPS, dict):
        LRPS={'input':LRPS}
        LRPT={'input':LRPT}

    l=[]
    for key in LRPS:
        if not pyramidLoss:
            l.append(ISNetFunctions.PairLRPLoss(heatmap=LRPS[key]*scale,
                                                   heatmapTarget=LRPT[key],
                                                   huber=huber,L1=L1,norm=norm,
                                                   normTarget=normTarget,CE=CELRP,
                                                   reduction=reduction,
                                                   stdTarget=stdNorm,
                                                   normPerBatch=normPerBatch,
                                                   detachTeacher=detachTeacher,
                                                   KLDiv=KLDivLRP,geoStd=geoStd,
                                                   mask=mask,loss=loss,
                                                   maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps))
        else:
            ls=ApplyFilters(LRPS[key],minSize=minSize,ratio=ratio)
            lt=ApplyFilters(LRPT[key],minSize=minSize,ratio=ratio)
            pl=[]
            for i,_ in enumerate(ls,0):
                pl.append(ISNetFunctions.PairLRPLoss(heatmap=ls[i]*scale,
                                                    heatmapTarget=lt[i],
                                                    huber=huber,L1=L1,norm=norm,
                                                    normTarget=normTarget,CE=CELRP,
                                                    reduction=reduction,
                                                    stdTarget=stdNorm,
                                                    normPerBatch=normPerBatch,
                                                    detachTeacher=detachTeacher,
                                                    KLDiv=KLDivLRP,geoStd=geoStd,
                                                   mask=mask,loss=loss,
                                                   maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps))
            l.append(ISNetFunctions.GlobalWeightedRankPooling(torch.stack(pl,dim=-1),
                                                           d=dPyramid,oneD=True,
                                                           descending=GWRPDescending,
                                                           rank=GWRPRank))
    if simpleCurriculum:
        lossesLRP=dLayers*torch.stack(l,dim=0).mean(0)+\
                      (1-dLayers)*l[0]#input lrp loss
    else:
        lossesLRP=ISNetFunctions.GlobalWeightedRankPooling(torch.stack(l,dim=-1),
                                                           d=dLayers,oneD=True,
                                                           descending=GWRPDescending,
                                                           rank=GWRPRank)
    return lossesLRP
    


def studentLoss(out=None,yS=None,yT=None,featuresS=None,featuresT=None,
                LRPS=None,LRPT=None,LRPFlexS=None,LRPFlexT=None,
                huber=False,L1=False,norm=False,normTarget=False,CELRP=False,KLDivLRP=False,
                reduction='mean',stdNorm=False,
                online=False,labels=None,normPerBatch=False,
                detachTeacher=True,CELogits=False,
                normLogits=False,normFeatures=False,
                pyramidLoss=False,minSize=8,ratio=2,
                dPyramid=1,dLayers=1,GWRPDescending=True,GWRPRank=True,
                simpleCurriculum=False,geoStd=False,scale=1,
                mask=None,loss='MSE',LRPlossOnFeatures=False,
                maskTargetLRP=False,d=0.999,crossLoss=False,
                geoL1Weight=1,cosWeight=1,
                temperature=3,basketEps=0.3,
                balancing=None):
    #online: trains teacher and student simultaneously
    
    if out is not None:
        yS=out['outputStudent']
        featuresS=out['featuresStudent']
        LRPS=out['LRPStudent']
        LRPFlexS=out['LRPFlexStudent']
        yT=out['outputTeacher']
        featuresT=out['featuresTeacher']
        LRPT=out['LRPTeacher']
        LRPFlexT=out['LRPFlexTeacher']
        
    losses={'logits':None,
             'features':None,
             'LRP':None,
             'LRPFlex':None,
             'LRPTeacher':None,
             'LRPFlexTeacher':None,
             'HeatmapLossISNet':None,
             'HeatmapLossISNetFlex':None,
             'classificationStudent':None,
             'embeddingsStudent':None,
             'CosEmbeddingsStudent':None}
    
    scale=torch.exp(torch.ones(1).type_as(yS)*scale)
    
    if online:
        #classification loss over teacher
        if len(labels.shape)>1:
            labels=labels.squeeze(-1)
        if len(labels.shape)>1:#multi-task
            losses['classificationTeacher']=torch.nn.functional.binary_cross_entropy_with_logits(
                yT,labels,pos_weight=balancing)
        else:
            losses['classificationTeacher']=torch.nn.functional.cross_entropy(yT,labels,
                                                                              weight=balancing)
                

    if labels is not None:
        if len(labels.shape)>1:
            labels=labels.squeeze(-1)
        if len(labels.shape)>1:#multi-task
            losses['classificationStudent']=torch.nn.functional.binary_cross_entropy_with_logits(
                yS,labels,pos_weight=balancing)
        else:
            
            losses['classificationStudent']=torch.nn.functional.cross_entropy(yS,labels,
                                                                              weight=balancing)    
            
    if not CELogits:
        if normLogits:
            yS=yS/(torch.std(yT,dim=-1,keepdim=True)+1e-10)#Norm target
            yT=yT/(torch.std(yT,dim=-1,keepdim=True)+1e-10)
        if detachTeacher:
            losses['logits']=torch.nn.functional.mse_loss(yS,yT.detach(),reduction='mean')
        else:
            losses['logits']=torch.nn.functional.mse_loss(yS-yT,torch.zeros(yS.shape).type_as(yS),
                                                          reduction='mean')
        #print(losses['logits'])
        #print('student:',yS)
        #print('teacher:',yT)
    else:
        #print('cross entropy')
        #Soften
        yS=torch.nn.functional.log_softmax(yS/temperature,-1)
        yT=torch.nn.functional.softmax(yT/temperature,-1)
        
        if detachTeacher:
            yT=yT.detach()
        losses['logits']=(torch.sum(yT*(yT.log()-yS))/yS.size()[0])*(temperature**2)
        #yS=yS/temperature
        #yT=yT/temperature
        #yT=torch.nn.functional.softmax(yT,-1)
        
        #if detachTeacher:
        #    yT=yT.detach()
        #print(yT)
        #losses['logits']=torch.nn.functional.binary_cross_entropy_with_logits(yS,yT,
        #                 pos_weight=balancing)*(temperature**2)
        #print(losses['logits'])
        
        
    if out['embeddingsStudent'] is not None:
        losses['embeddingsStudent']=torch.nn.functional.binary_cross_entropy_with_logits(
                                                                      out['embeddingsStudent'],
                                                                      out['embeddingsTeacher'])
        es=out['embeddingsStudent']/out['embeddingsStudent'].norm(dim=-1, keepdim=True)
        et=out['embeddingsTeacher']/out['embeddingsTeacher'].norm(dim=-1, keepdim=True)
        cos=(et*es).sum(-1)
        losses['CosEmbeddingsStudent']=torch.nn.functional.binary_cross_entropy_with_logits(
            cos,torch.ones(cos.shape).type_as(cos))
        
        
        
    if featuresS is not None:
        #print(featuresS.shape,featuresT.shape)
        if featuresS.shape!=featuresT.shape:
            featuresS=featuresS.squeeze()
            featuresT=featuresT.squeeze()
        if featuresS.shape==featuresT.shape:
            if LRPlossOnFeatures:
                if loss!='geoL1Cos':
                    losses['features']=ISNetFunctions.PairLRPLoss(heatmap=featuresS.unsqueeze(1),
                                                               heatmapTarget=featuresT.unsqueeze(1),
                                                               huber=huber,L1=L1,norm=norm,
                                                               normTarget=normTarget,CE=CELRP,
                                                               reduction=reduction,
                                                               stdTarget=stdNorm,
                                                               normPerBatch=normPerBatch,
                                                               detachTeacher=detachTeacher,
                                                               KLDiv=KLDivLRP,geoStd=geoStd,
                                                               mask=mask,loss=loss,
                                                               maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
                else:
                    L1=ISNetFunctions.PairLRPLoss(heatmap=featuresS.unsqueeze(1),
                                                               heatmapTarget=featuresT.unsqueeze(1),
                                                               huber=huber,L1=L1,norm='geoL1',
                                                               normTarget=normTarget,CE=CELRP,
                                                               reduction=reduction,
                                                               stdTarget=stdNorm,
                                                               normPerBatch=normPerBatch,
                                                               detachTeacher=detachTeacher,
                                                               KLDiv=KLDivLRP,geoStd=geoStd,
                                                               mask=mask,loss='L1',
                                                               maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
                    cos=ISNetFunctions.PairLRPLoss(heatmap=featuresS.unsqueeze(1),
                                                               heatmapTarget=featuresT.unsqueeze(1),
                                                               huber=huber,L1=L1,norm='individualL2',
                                                               normTarget=normTarget,CE=CELRP,
                                                               reduction=reduction,
                                                               stdTarget=stdNorm,
                                                               normPerBatch=normPerBatch,
                                                               detachTeacher=detachTeacher,
                                                               KLDiv=KLDivLRP,geoStd=geoStd,
                                                               mask=mask,loss='cos',
                                                               maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
                    losses['features']=geoL1Weight*L1+cosWeight*cos
                    losses['featuresL1']=L1
                    losses['featuresCos']=cos
            else:
                if normFeatures:
                    featuresS=featuresS/(torch.std(featuresT,dim=(-1,-2,-3),keepdim=True)+1e-10)
                    featuresT=featuresT/(torch.std(featuresT,dim=(-1,-2,-3),keepdim=True)+1e-10)
                #feature map pairing loss
                if detachTeacher:
                    losses['features']=torch.nn.functional.mse_loss(featuresS,featuresT.detach(),
                                                                     reduction='mean')
                else:
                    losses['features']=torch.nn.functional.mse_loss(featuresS-featuresT,
                                        torch.zeros(featuresS.shape).type_as(featuresS),
                                                                     reduction='mean')

        

    if LRPS is not None:
        if loss!='geoL1Cos':
            losses['LRP']=LRPPair(LRPS=LRPS,LRPT=LRPT,
                                  huber=huber,L1=L1,norm=norm,
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss=loss,LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
        else:
            L1=LRPPair(LRPS=LRPS,LRPT=LRPT,
                                  huber=huber,L1=L1,norm='geoL1',
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss='L1',LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
            cos=LRPPair(LRPS=LRPS,LRPT=LRPT,
                                  huber=huber,L1=L1,norm='individualL2',
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss='cos',LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
            losses['LRP']=geoL1Weight*L1+cosWeight*cos
            losses['LRPL1']=L1
            losses['LRPCos']=cos
        
        if crossLoss and online:
            if not detachTeacher:
                raise ValueError('set detachTeacher to True for cross loss')
            if mask is None:
                raise ValueError('provide segmentation masks for cross loss')
                
            #invert teacher and student, mask target heatmap (student)
            losses['LRPTeacher']=LRPPair(LRPS=LRPT,LRPT=LRPS,
                                  huber=huber,L1=L1,norm=norm,
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss=loss,LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
        
        l=[]
        for key in list(LRPS.keys()):
            l.append(ISNetFunctions.LRPLossCEValleysGWRP(LRPS[key]*scale,
                                                         mask,
                                                         A=1,B=0,d=d,E=1,
                                                         channelGWRP=1,
                                                         norm=True,
                                                         alternativeForeground=False,
                                                         newSeparate=False,
                                           pyramidLoss=pyramidLoss,minSize=minSize,
                                           dPyramid=dPyramid,pyramidGWRP=GWRPRank))
        losses['HeatmapLossISNet']=ISNetFunctions.GlobalWeightedRankPooling(
                                                               torch.stack(l,dim=-1),
                                                               d=dLayers,oneD=True,
                                                               descending=GWRPDescending,
                                                               rank=GWRPRank)
            
    if LRPFlexS is not None:
        #print('mask teacher:',maskTargetLRP)
        #print('norm:',norm)
        
        if loss!='geoL1Cos':
            losses['LRPFlex']=LRPPair(LRPS=LRPFlexS,LRPT=LRPFlexT,
                                  huber=huber,L1=L1,norm=norm,
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss=loss,LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
        else:
            L1=LRPPair(LRPS=LRPFlexS,LRPT=LRPFlexT,
                                  huber=huber,L1=L1,norm='geoL1',
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss='L1',LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
            cos=LRPPair(LRPS=LRPFlexS,LRPT=LRPFlexT,
                                  huber=huber,L1=L1,norm='individualL2',
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss='cos',LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)
            losses['LRPFlex']=geoL1Weight*L1+cosWeight*cos
            losses['LRPFlexL1']=L1
            losses['LRPFlexCos']=cos
              
        if crossLoss and online:
            if not detachTeacher:
                raise ValueError('set detachTeacher to True for cross loss')
            if mask is None:
                raise ValueError('provide segmentation masks for cross loss')
                
            #invert teacher and student, mask target heatmap (student)
            losses['LRPFlexTeacher']=LRPPair(LRPS=LRPFlexT,LRPT=LRPFlexS,
                                  huber=huber,L1=L1,norm=norm,
                                  normTarget=normTarget,CELRP=CELRP,KLDivLRP=KLDivLRP,
                                  reduction=reduction,stdNorm=stdNorm,normPerBatch=normPerBatch,
                                  detachTeacher=detachTeacher,
                                  pyramidLoss=pyramidLoss,minSize=minSize,ratio=ratio,
                                  dPyramid=dPyramid,dLayers=dLayers,GWRPDescending=GWRPDescending,
                                  GWRPRank=GWRPRank,
                                  simpleCurriculum=simpleCurriculum,geoStd=geoStd,scale=scale,
                                  mask=mask,loss=loss,LRPlossOnFeatures=LRPlossOnFeatures,
                                  maskTargetLRP=maskTargetLRP,
                                                   basketEps=basketEps)      
        #if reduction=='mean':
        #    losses['LRPFlex']=losses['LRPFlex'].mean()
        #elif reduction=='sum':
        #    losses['LRPFlex']=losses['LRPFlex'].sum()
        if mask is not None:
            #ISNet loss
            l=[]
            for key in list(LRPFlexS.keys()):
                l.append(ISNetFunctions.LRPLossCEValleysGWRP(LRPFlexS[key].unsqueeze(1)*scale,
                                                             mask,
                                                             A=1,B=0,d=d,E=1,
                                                             channelGWRP=1,
                                                             norm=True,
                                                             alternativeForeground=False,
                                                             newSeparate=False,
                                               pyramidLoss=pyramidLoss,minSize=minSize,
                                               dPyramid=dPyramid,pyramidGWRP=GWRPRank))
            losses['HeatmapLossISNetFlex']=ISNetFunctions.GlobalWeightedRankPooling(
                                                                   torch.stack(l,dim=-1),
                                                                   d=dLayers,oneD=True,
                                                                   descending=GWRPDescending,
                                                                   rank=GWRPRank)
            
    return losses

def TeacherLoss(out,labels,masks=None,pyramidLoss=False,minSize=8,ratio=2,
                heat=False,tuneCut=False,d=1,dLoss=1,dPyramid=1,
                cut=None,cut2=None,cutFlex=None,cut2Flex=None,
                alternativeForeground=False,separate=False):
    
    L={'classification':None,
       'LRP':None,
       'LRPFlex':None,
       'mapAbs':None,
       'mapAbsFlex':None}
    
    outputs=out['outputTeacher']
    LRPT=out['LRPTeacher']
    LRPFlexT=out['LRPFlexTeacher']
    
    if len(labels.shape)>1:
        labels=labels.squeeze(-1)
    if len(labels.shape)>1:#multi-task
        L['classification']=F.binary_cross_entropy_with_logits(outputs,labels)
    else:
        L['classification']=torch.nn.functional.cross_entropy(outputs,labels)
    
    
    
    if LRPT is not None:
        if not isinstance(LRPT, dict):
            LRPT={'input':LRPT}
        if not isinstance(cut, dict):
            cut={'input':cut}
            cut2={'input':cut2}
        
        losses=[]
        tune={}
        for key in LRPT:
            if tuneCut:
                heatmapLoss,foreg=ISNetFunctions.LRPLossCEValleysGWRP(LRPT[key],masks,
                                                               A=1,B=1,d=d,
                                                               E=1,
                                                               rule='e',
                                                               tuneCut=tuneCut,
                                                               norm=True,
                                                               channelGWRP=1,
                                           alternativeForeground=alternativeForeground,
                                                            newSeparate=separate)
                losses.append(heatmapLoss)
                tune[key]=foreg

            else:
                heatmapLoss=ISNetFunctions.LRPLossCEValleysGWRP(LRPT[key],masks,
                                                           cut=cut[key],
                                                           cut2=cut2[key],
                                                           A=1,B=1,d=d,
                                                           E=1,
                                                           rule='e',
                                                           tuneCut=tuneCut,
                                                           channelGWRP=1,
                                                           norm=True,
                                       alternativeForeground=alternativeForeground,
                                                        newSeparate=separate,
                                       pyramidLoss=pyramidLoss,minSize=minSize,
                                       dPyramid=dPyramid,pyramidGWRP=False)
                losses.append(heatmapLoss)
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=ISNetFunctions.GlobalWeightedRankPooling(heatmapLoss,d=dLoss)
        if tuneCut:
            L['mapAbs']=tune
        L['LRP']=heatmapLoss
        
    if LRPFlexT is not None:
        if not isinstance(LRPFlexT, dict):
            LRPFlexT={'input':LRPFlexT}
        if not isinstance(cutFlex, dict):
            cutFlex={'input':cutFlex}
            cut2Flex={'input':cut2Flex}
            
        losses=[]
        tune={}
        for key in LRPFlexT:
            if tuneCut:
                heatmapLoss,foreg=ISNetFunctions.LRPLossCEValleysGWRP(LRPFlexT[key].unsqueeze(1),
                                                                      masks,
                                                                       A=1,B=1,d=d,
                                                                       E=1,
                                                                       rule='e',
                                                                       tuneCut=tuneCut,
                                                                       norm=True,
                                                                       channelGWRP=1,
                                           alternativeForeground=alternativeForeground,
                                                            newSeparate=separate)
                losses.append(heatmapLoss)
                tune[key]=foreg

            else:
                heatmapLoss=ISNetFunctions.LRPLossCEValleysGWRP(LRPFlexT[key].unsqueeze(1),
                                                                masks,
                                                           cut=cutFlex[key],
                                                           cut2=cut2Flex[key],
                                                           A=1,B=1,d=d,
                                                           E=1,
                                                           rule='e',
                                                           tuneCut=tuneCut,
                                                           channelGWRP=1,
                                                           norm=True,
                                       alternativeForeground=alternativeForeground,
                                                        newSeparate=separate,
                                       pyramidLoss=pyramidLoss,minSize=minSize,
                                       dPyramid=dPyramid,pyramidGWRP=False)
                losses.append(heatmapLoss)
                
        heatmapLoss=torch.stack(losses,dim=-1)
        heatmapLoss=ISNetFunctions.GlobalWeightedRankPooling(heatmapLoss,d=dLoss)
        if tuneCut:
            L['mapAbsFlex']=tune
        L['LRPFlex']=heatmapLoss
        
    return L
    


def ApplyFilters(heatmap,filters=4,ratio=2,minSize=None):
    if minSize is not None:
        if isinstance(minSize,int):
            minSize=torch.tensor([minSize,minSize])
        elif isinstance(minSize,list):
            minSize=torch.tensor(minSize)
        else:
            raise ValueError('minSize should be int or list')
            
        if heatmap.shape[-1]<=minSize[-1].item() or \
           heatmap.shape[-2]<=minSize[-2].item():
            return [heatmap]
        filters=1
        size=torch.tensor([heatmap.shape[-2],heatmap.shape[-1]])
        #print(filters,size)
        while size[0].item()>=minSize[0].item() and \
              size[1].item()>=minSize[1].item():
            size=size/ratio
            if size[0].item()>=minSize[0].item() and \
               size[1].item()>=minSize[1].item():
                filters+=1
                #print(filters,size)
                
        #print(filters)
        
    
    if len(heatmap.shape)==5:
        x=heatmap.view(heatmap.shape[0]*heatmap.shape[1],
                       heatmap.shape[2],heatmap.shape[3],heatmap.shape[4])
        reshaped=True
    else:
        x=heatmap
        reshaped=False
    
    y=[]
    y.append(x)#filter 0
    for i in list(range(1,filters)):
        kernel=ratio**i
        if ((kernel>x.shape[-1]) or (kernel>x.shape[-2])):
            raise ValueError('Filter cannot be bigger than image shape, reduce number of filters or ratio')
        pooled=torch.nn.functional.avg_pool2d(x,kernel_size=(int(kernel),int(kernel)))
        if reshaped:
            pooled=pooled.view(heatmap.shape[0],heatmap.shape[1],
                               pooled.shape[-3],pooled.shape[-2],pooled.shape[-1])
        y.append(pooled)
    return y

#def separateBackbone(model,frozenLayers):
#    first=copy.deepcopy(model)
#    last=model
#    print(frozenLayers)
#    if frozenLayers=='allBut4':
#        for name, _ in model.CLIP_vision_encoder.named_children():
#            if ('layer4' not in name and 'attnpool' not in name):
#                setattr(last.CLIP_vision_encoder, name, nn.Identity())
#            else:
#                setattr(first.CLIP_vision_encoder, name, nn.Identity())
#    else:
#        raise ValueError('Unrecognized frozenLayers')
#        
#        
#    for name, _ in model.named_children():
#        if ('CLIP_vision_encoder' not in name):
#            setattr(first, name, nn.Identity())
#        
#    for param in first.parameters():
#        param.requires_grad=False
#        
#    return first, last

def freezeBackbone(model,frozenLayers):
    if frozenLayers=='allBut4':
        for name, param in model.CLIP_vision_encoder.named_parameters():
            if ('layer4' not in name and 'attnpool' not in name):
                param.requires_grad=False
        return ISNetFunctions.HookFwd(model.CLIP_vision_encoder.layer3,mode='output')
    else:
        raise ValueError('Unrecognized frozenLayers')
    

class TeacherStudent(nn.Module):
    def __init__(self,architecture=None,teacher=None,
                 dropout=False,classes=1,
                 mask='noise',
                 beginAsTeacher=True,freezeLastLayer=False,zeroBias=False,
                 LRPBlock=True,e=1e-2,Zb=False,rule='e',multiple=True,
                 randomLogit=False,selective=False,epsSelective=0.01,highest=False,
                 HiddenLayerPenalization=False,
                 detach=True,
                 SequentialInputShape=None,SequentialPreFlattenShape=None,
                 epsLow=-2,epsHigh=-1,pencentageEpsZero=0,scale=False,
                 mode='offline',pretrainedOnline=None,
                 imageNetPretrained=False,matchArchitecture=True,
                 textFeatures=None,clipPreprocess=None,clipSavedMaps=None,
                 trainableT=False,teachereT=100, 
                 CLIP1024Feat=True, CLIPLikeLast=False,
                 inputGradAblation=False,attentionAblation=False,GradCAMAblation=False,
                 frozenLayers='none',CLIPModLastLayer=False,MNISTSpecial=False):
        
        super(TeacherStudent,self).__init__()
        
        if frozenLayers!='none' and (not beginAsTeacher or mode!='offline'):
            raise ValueError('Frozen backbone (filterBlock)requires beginning as teacher')
        
        self.mode=mode
        self.Zb=Zb
        print('Mode is: '+self.mode)
        if mode not in ['online','offline','onlineSeparateWeights','online']:
            raise ValueError('Unrecognized mode')
            
        if textFeatures is not None:
            if mode != 'offline':
                raise ValueError('offline is the only mode implemented for CLIP')
            import torchvision.transforms as transforms
            self.teacher=teacher
            self.clipTeacher=True
            self.preprocess=clipPreprocess
            self.clipSavedMaps=clipSavedMaps
        else:
            self.preprocess=None
        
        #create student
        if self.mode=='offline' or self.mode=='onlineSeparateWeights':
            if textFeatures is None:
                if self.mode=='offline':
                    self.teacher=teacher
                    tmp=0
                    num=0
                    for param in teacher.parameters():
                        tmp=tmp+torch.abs(param).sum().item()
                        #print(param)
                        num+=param.numel()
                    print('mean of parameters of teacher is:',tmp/num)
                    #raise ValueError('Stopping')
                else:
                    self.teacher=getBackbone(architecture,dropout,classes,
                                             pretrained=imageNetPretrained)
            self.teacher=CleanNet(self.teacher)
            if matchArchitecture:
                self.teacher=RemoveInplace(self.teacher,architecture)
            else:
                if MNISTSpecial:
                    self.teacher=RemoveInplace(self.teacher,'resnet34')
                else:
                    ChangeInplace(self.teacher)
                    print('Please make sure teacher has no in-place opeartions')
                
                
            if beginAsTeacher:
                self.student=copy.deepcopy(self.teacher)
                print('COPIED TEACHER AS STUDENT')
            else:
                if matchArchitecture:
                    for layer in self.student.children():
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
                    print('PARAMETERS RESET')
                else:
                    self.student=getBackbone(architecture,dropout,classes,
                                             pretrained=imageNetPretrained)                    
                
            if self.mode=='offline':
                #freze and zero necessary parameters
                self.teacher,self.student=freeze(self.teacher,self.student,
                                                 (freezeLastLayer and (textFeatures is None)),
                                                 zeroBias,
                                                 freezeTeacher=True)
                print('Created teacher and student')
                #print(self.teacher)
                    
                if (textFeatures is not None and freezeLastLayer and not beginAsTeacher):
                    print('Add extra layer')
                    intermediate=nn.Conv2d(
                                in_channels=self.student.layer4[-1].conv1.in_channels,                                               out_channels=self.teacher.layer4[-1].conv1.in_channels,
                                           kernel_size=(1,1))
                    pool=copy.deepcopy(self.teacher.attnpool)
                    for param in pool.parameters():
                        param.requires_grad=False
                    self.student.layer4[-1]=nn.Sequential(OrderedDict([
                                                ('a',self.student.layer4[-1]),
                                                ('intermediateExtra',intermediate),
                                                ('attnpool',pool)]))
                    self.student.avgpool=nn.Identity()
                        
                        
            else:
                #do not freeze teacher
                self.teacher,self.student=freeze(self.teacher,self.student,freezeLastLayer=False,
                                                 zeroBias=zeroBias,freezeTeacher=False)
            self.hooksTeacher=InsertReLUHooks(self.teacher)
            self.hooksStudent=InsertReLUHooks(self.student)
            print('Inserted hooks in teacher and student')
            if LRPBlock:
                #LRP block
                self.teacher=CreateLRPBlock(self.teacher,e=e,Zb=Zb,rule=rule,multiple=multiple,
                                            selective=selective,highest=highest,detach=detach,
                                            randomLogit=False,storeRelevance=HiddenLayerPenalization)
                #do not set random logit to true, not adequate for pairing
                self.student=CreateLRPBlock(self.student,e=e,Zb=Zb,rule=rule,multiple=multiple,
                                            selective=selective,highest=highest,detach=detach,
                                            randomLogit=False,storeRelevance=HiddenLayerPenalization)
                #do not set random logit to true, not adequate for pairing
            
            self.featureMapT,self.featureMapS=InsertFeaturesHook(self.teacher,self.student)
            
            if HiddenLayerPenalization:
                self.HLPHooksFlexT=LRPFlexStoreActivations(self.teacher,
                                                          ReLUHooks=self.hooksTeacher)
                self.HLPHooksFlexS=LRPFlexStoreActivations(self.student,
                                                          ReLUHooks=self.hooksStudent)
                
            if Zb:
                print('ZB IS TRUE')
                self.SetZbLayer(self.teacher,'teacher')
                self.SetZbLayer(self.student,'student')
            if textFeatures is not None:
                self.teacher=nn.Sequential(OrderedDict([
                                ('CLIP_vision_encoder',self.teacher),
                                ('CLIP_output_layer',CLIPOutput(textFeatures,
                                                                teachereT=teachereT,
                                                                 trainableW=False,
                                                                applyT=(not CLIPModLastLayer),
                                                                bias=False,
                                                norm=(not CLIPModLastLayer)
                                                               ))]))
                if beginAsTeacher:
                    self.student=nn.Sequential(OrderedDict([
                                ('CLIP_vision_encoder',self.student),
                                ('fc',CLIPOutput(textFeatures,
                                                 trainableT=trainableT,
                                                 trainableW=(not freezeLastLayer),
                                                 teachereT=teachereT,
                                                 applyT=(not CLIPModLastLayer),
                                                 bias=CLIPModLastLayer,
                                                norm=(not CLIPModLastLayer),
                                                ))]))
                    #print(self.student)
                if not beginAsTeacher and freezeLastLayer:
                    if not matchArchitecture:
                        self.student.fc=nn.Identity()
                    self.student=nn.Sequential(OrderedDict([
                                ('CLIP_vision_encoder',self.student),
                                ('fc',CLIPOutput(textFeatures,
                                                trainableT=trainableT,
                                                 trainableW=False,
                                                teachereT=teachereT,
                                                applyT=(not CLIPModLastLayer),
                                                bias=False,
                                                norm=(not CLIPModLastLayer)))]))
                if CLIPLikeLast:
                    if freezeLastLayer:
                        raise ValueError('CLIPLikeLast not compatible with freezeLast')
                    weightMatrix = torch.empty(classes, self.student.fc.in_features)
                    stdv = 1. / math.sqrt(weightMatrix.size(1))
                    weightMatrix.uniform_(-stdv, stdv)
                    self.student.fc=CLIPOutput(weightMatrix,
                                               trainableT=trainableT,
                                               trainableW=True,
                                               teachereT=teachereT)
                    #print(self.student)
                    
                if CLIPLikeLast:
                    if freezeLastLayer:
                        raise ValueError('CLIPLikeLast not compatible with freezeLast')
                    weightMatrix = torch.empty(classes, self.student.fc.in_features)
                    stdv = 1. / math.sqrt(weightMatrix.size(1))
                    weightMatrix.uniform_(-stdv, stdv)
                    self.student.fc=CLIPOutput(weightMatrix,
                                               trainableT=trainableT,
                                               trainableW=True,
                                               teachereT=teachereT)
                
        elif self.mode=='online':
            if pretrainedOnline is None:
                self.model=getBackbone(architecture,dropout,
                                       classes,pretrained=imageNetPretrained)
            else:
                self.model=pretrainedOnline
                print('Not starting from scratch')
                for param in self.model.parameters():
                    param.requires_grad=True
            self.model=CleanNet(self.model)
            self.model=RemoveInplace(self.model,architecture)
            self.hooksModel=InsertReLUHooks(self.model)
            if LRPBlock:
                #LRP block
                self.model=CreateLRPBlock(self.model,e=e,Zb=Zb,rule=rule,multiple=multiple,
                                            selective=selective,highest=highest,detach=detach,
                                            randomLogit=False,storeRelevance=HiddenLayerPenalization)
                #do not set random logit to true, not adequate for pairingadequate for pairing
            #Forward hook to get last feature map
            self.featureMap=InsertFeaturesHookSingle(self.model)
            if HiddenLayerPenalization:
                self.HLPHooksFlex=LRPFlexStoreActivations(self.model,
                                                          ReLUHooks=self.hooksModel)
            
            if Zb:
                self.SetZbLayer(self.model,'online')
        else:
            raise ValueError('Unrecognized mode')
        
        
        globals.LRP=False
        
        self.HLP=HiddenLayerPenalization
        
        #save parameters
        self.epsLow=epsLow
        self.epsHigh=epsHigh
        globals.pencentageEpsZero=pencentageEpsZero
        
        self.maskMode=mask
        self.randomLogitLRPBlock=randomLogit
        self.rule=rule
        self.selective=selective
        self.epsSelective=epsSelective
        self.sanityCheckImages=False
        self.sanityCount=0
        
        self.inputGradAblation=inputGradAblation
        if inputGradAblation:
            print('Running input gradients ablation study, output[LRPFlex] and LRPFlex loss now represent input gradients, not LRP')
        self.attentionAblation=attentionAblation
        if attentionAblation:
            print('Running attention ablation study, output[features] and features loss now represent attention')
        self.GradCAMAblation=GradCAMAblation
        if GradCAMAblation:
            print('Running Grad-CAM ablation study, output[features] and features loss now represent Grad-CAM')
        
        if scale:
            #trainable scale parameter
            self.scale=torch.nn.parameter.Parameter(torch.zeros(1),requires_grad=True)
        else:
            #constant scale
            self.scale=0
            
        if textFeatures is not None and self.clipSavedMaps:
            del self.teacher
            
            
        if frozenLayers!='none':
            #self.frozen,self.teacher=separateBackbone(self.teacher,frozenLayers)
           # _,self.student=separateBackbone(self.student,frozenLayers)
            #print(self.student)
            #print(self.frozen)
            self.frozenBackboneHookStudent=freezeBackbone(self.student,frozenLayers)
            self.frozenBackboneHookTeacher=freezeBackbone(self.teacher,frozenLayers)
        else:
            self.frozenBackboneHookStudent=None
            self.frozenBackboneHookTeacher=None
            
            
        #else:
        #    self.frozen=None
        
    def SetZbLayer(self,model,mode):
        #Zb: separate the DNN first layer
        
        firstLayer=[]
        flag=False
        for layer in model.children():
            if layer.__class__.__name__=='Sequential':
                for layer in layer.children():
                    firstLayer.append(layer)
                    if isinstance(layer, torch.nn.ReLU):
                        flag=True
                        break
            else:
                firstLayer.append(layer)
            if flag:
                break
            if isinstance(layer, torch.nn.ReLU):
                break
        
        chain=[]
        for module in firstLayer:
            chain.append(module.__class__.__name__)
            
        #LRP block for first layer:
        if (chain==['Conv2d','ReLU'] or chain==['Conv2d']):
            model.ZbModule=ZbRuleConvInput(firstLayer[0],l=0,h=1)
        elif chain==['Conv2d','BatchNorm2d','ReLU']:
            model.ZbModule=ZbRuleConvBNInput(firstLayer[0],
                                             firstLayer[1],
                                             l=0,h=1)
        elif chain==['Linear','ReLU']:
            model.ZbModule=ZbRuleDenseInput(firstLayer[0],l=0,h=1)
        else:
            #print(model)
            print(chain)
            raise ValueError ('Unrecognized first layer sequence for Zb rule')
        model.chain=chain
        #add hook to save first ReLU output, to get relevance after first layer:
        model.firstLayerOutput=SaveOutput(firstLayer[-1])
        model.firstLayerLinearOutput=SaveOutput(firstLayer[0])
        if 'BatchNorm2d' in model.chain:
            model.firstLayerBNOutput=SaveOutput(firstLayer[1])
        print(firstLayer)
        
        #set first ReLU to save its output LRP:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                model.firstReLUName=name
                break
        print('first relu:', model.firstReLUName)
        if mode=='online':
            self.hooksModel[model.firstReLUName][1].saveLRPout=True
        elif mode=='teacher':
            self.hooksTeacher[model.firstReLUName][1].saveLRPout=True
        elif mode=='student':
            self.hooksStudent[model.firstReLUName][1].saveLRPout=True
        else:
            raise ValueError('Unrecognized mode')
        
    def forward(self,xS,xT=None,
                maskS=None,maskT=None,
                runLRPBlock=False,runLRPFlex=False,
                runFeatureGrad=False,
                indices=None,names=None):
        #xT: input for teacher. If None, teacher is not run
        #xS: input for student. If None, student is not run
        #maskT: if provided, mask is applied to teacher input
        #maskS: if provided, mask is applied to student input
        #runLRPBlock: if True, will run the LRP block
        #runLRPFlex: if True, will run the LRP flex
        
        if (xT is None) and (xS is None):
            raise ValueError('Both teacher and student inputs are None')
        
        #set all outputs to none, they will be overwritten
        outputs={'outputStudent':None,
                 'featuresStudent':None,
                 'LRPStudent':None,
                 'LRPFlexStudent':None,
                 'featureGradientsStudent':None,
                 'embeddingsStudent':None,
                 'outputTeacher':None,
                 'featuresTeacher':None,
                 'LRPTeacher':None,
                 'LRPFlexTeacher':None,
                 'featureGradientsTeacher':None,
                 'embeddingsTeacher':None
                 }
        
        if runLRPFlex:
            #random selection of e in LRP-e
            if xS is not None:
                self.DefineEps(xS)
            else:
                self.DefineEps(xT)
                
        #lrp block memory    
        if self.preprocess is not None:
            if (xS is not None) and (xT is not None):
                if not torch.equal(xS,xT):
                    raise ValueError('xS and xT are expected to match for clip teacher')
                xS=self.preprocess(xS).float()
                xT=xS.clone()
            elif (xS is not None) and (xT is None):
                xS=self.preprocess(xS).float()
            elif (xS is None) and (xT is not None):
                xT=self.preprocess(xT).float()
            else:
                raise ValueError('wrong code')
                
        #if self.frozen is not None:
            #pass trough frozen common backbone
        #    with torch.no_grad():
        #        if (xS is not None) and (xT is not None):
         #           if not torch.equal(xS,xT):
        #                raise ValueError('xS and xT are expected to match for clip teacher')
        #            xS=self.frozen(xS)
        #            xT=xS.clone()
        #        elif (xS is not None) and (xT is None):
        #            xS=self.frozen(xS)
        #        elif (xS is None) and (xT is not None):
        #            xT=self.frozen(xT)
        #        else:
        #            raise ValueError('wrong code')
        
        if xT is not None:
            #define LRPFlex indices on teacher
            outputs,indices=self.RunModule(outputs=outputs,x=xT,mask=maskT,
                                           runLRPFlex=runLRPFlex,
                                           runLRPBlock=runLRPBlock,
                                           module='Teacher',
                                           indices=indices,
                                           runFeatureGrad=runFeatureGrad)
        
        if xS is not None:
            outputs,_=self.RunModule(outputs=outputs,x=xS,mask=maskS,
                                     runLRPFlex=runLRPFlex,
                                     runLRPBlock=runLRPBlock,
                                     module='Student',
                                     indices=indices,
                                     runFeatureGrad=runFeatureGrad)
            
              
        return outputs
    
    #def ReadDict(self,teacherDict,outputs,x,mask,runLRPFlex,runLRPBlock,module,indices=None,
    #              runFeatureGrad=False):
        
        
    def RunModule(self,outputs,x,mask,runLRPFlex,runLRPBlock,module,indices=None,
                  runFeatureGrad=False):
        
        if self.mode=='online':
            net=self.model
            featureMapHook=self.featureMap
            ReLUHooks=self.hooksModel
            if self.HLP:
                HLPHooksFlex=self.HLPHooksFlex
            #Lrp flex parameters
            retain_graph=self.training
            create_graph=self.training
        elif module=='Teacher':
            net=self.teacher
            featureMapHook=self.featureMapT
            ReLUHooks=self.hooksTeacher
            if self.HLP:
                HLPHooksFlex=self.HLPHooksFlexT
            #Lrp flex parameters
            if self.mode=='offline':
                retain_graph=False
                create_graph=False
            else:
                retain_graph=self.training
                create_graph=self.training
            frozenHook=self.frozenBackboneHookTeacher
        elif module=='Student':
            net=self.student
            featureMapHook=self.featureMapS
            ReLUHooks=self.hooksStudent
            #Lrp flex parameters
            retain_graph=self.training
            create_graph=self.training
            if self.HLP:
                HLPHooksFlex=self.HLPHooksFlexS
            frozenHook=self.frozenBackboneHookStudent
        else:
            raise ValueError('Unrecognized module')
        
        #lrp block memory clean
        globals.X=[]
        globals.XI=[]   
        
        if x is not None:
            if mask is not None:
                #tmp=x.clone()
                x=mask_image(x,mask,mode=self.maskMode)
                #print('masked ',module)
            if self.sanityCheckImages and self.sanityCount<10:
                self.SaveImage(x[0],module)
                self.sanityCount+=1
            if runLRPFlex or runFeatureGrad:
                x=Variable(x, requires_grad=True)
            #run
            globals.LRP=runLRPFlex
            #print(globals.LRP,globals.e)
            outputs['output'+module]=net(x)
            globals.LRP=False
            if (self.training and outputs['output'+module].dtype!=torch.float32):
                raise ValueError('Support for half-precision not implemented, notice we manipulate the gradients and generate them inside forward')
            
            #get features (last convolutional output)
            outputs['features'+module]=featureMapHook.x.clone()
            #print(module+' features:',outputs['features'+module])
            
            if self.attentionAblation:
                #print('feature map shape:',featureMapHook.x.shape)
                outputs['features'+module]=(featureMapHook.x.clone()**2).sum(dim=1,keepdim=True)
                #print('attention map shape:',outputs['features'+module].shape)
            
            
            if ((indices is None) and (runLRPFlex or (runLRPBlock and self.randomLogitLRPBlock) \
               or runFeatureGrad)):
                    indices=RandomLogitIndices(outputs['output'+module])
                    
            if runFeatureGrad:# or self.GradCAMAblation:
                quantity=[]
                for i,val in enumerate(indices,0):
                    quantity.append(outputs['output'+module][i,val])
                quantity=torch.stack(quantity,0).sum(0)
                globals.LRP=False
                G=ag.grad(quantity, [featureMapHook.x], retain_graph=True, 
                          create_graph=create_graph)
                outputs['featureGradients'+module]=G[0]
                
                #GradCAM
                AVG=outputs['featureGradients'+module].mean(dim=(-2,-1),keepdim=True)
                GCAM=(outputs['features'+module]*AVG).sum(dim=1,keepdim=True)
                GCAM=torch.nn.functional.relu(GCAM)
                if self.GradCAMAblation:
                    outputs['features'+module]=GCAM
                    #print(GCAM.shape)
                    #print('running grad-cam')
                
                
            if hasattr(net,'CLIP_output_layer'):
                outputs['embeddings'+module]=net.CLIP_output_layer.embeddings
                
            #get LRP Flex output
            if runLRPFlex:
                quantity=[]
                for i,val in enumerate(indices,0):
                    if self.selective:
                        quantity.append(torch.softmax(outputs['output'+module],dim=-1)[i,val])
                    else:
                        quantity.append(outputs['output'+module][i,val])
                quantity=torch.stack(quantity,0)
                if self.selective:
                    with torch.cuda.amp.autocast(enabled=False):
                        quantity=quantity.float()
                        #quantity=torch.clamp(quantity,min=1e-7,max=1-1e-7)
                        #quantity=torch.log(quantity)-torch.log(1-quantity)
                        quantity=torch.log(quantity+self.epsSelective)\
                        -torch.log(1-quantity+self.epsSelective)
                        #print('running selective')
                quantity=quantity.sum(0)
                #sum because batch size should not change LRP maps
                
                globals.LRP=True
                if frozenHook is None:
                    G=ag.grad(quantity, [x], retain_graph=retain_graph, 
                              create_graph=create_graph)
                else:
                    G=ag.grad(quantity, [frozenHook.x], retain_graph=retain_graph, 
                              create_graph=create_graph)
                    
                globals.LRP=False

                GX={}
                if self.Zb:
                    R=torch.nan_to_num(ReLUHooks[self.firstReLUName][1].outLRP).unsqueeze(1)
                    if 'BatchNorm2d' not in net.chain:
                        GX['input']=net.ZbModule(rK=R,
                                    aJ=x,aK=net.firstLayerLinearOutput.output).squeeze(1)
                    else:
                        GX['input']=net.ZbModule(rK=R,aJ=x,
                                                 aKConv=net.firstLayerLinearOutput.output,
                                        aK=net.firstLayerBNOutput.output).squeeze(1)
                else:
                    if frozenHook is None:
                        GX['input']=(torch.nan_to_num(G[0])*x)
                    else:
                        GX['input']=(torch.nan_to_num(G[0])*frozenHook.x)
                        


                if self.HLP:
                    for key in HLPHooksFlex:
                        GX[key]=torch.nan_to_num(self.HLPHooksFlex[key].outLRP)
                
                outputs['LRPFlex'+module]=GX
                
                if self.inputGradAblation:
                    if self.Zb or self.HLP:
                        raise ValueError('Zb and HLP should be false for input gradients ablation')
                    if globals.e.sum()!=0:
                        raise ValueError('epsilon should be zero for input gradients ablation')
                    GX['input']=torch.nan_to_num(G[0])
                    #print('optimizing input gradients')
                    # do not multiply by inputs
                    
                
                
                
                
                                              
                
            if runLRPBlock:
                if (module=='Teacher' and self.mode=='offline'):
                    with torch.no_grad():#reduce memory consumption, as you do not optimize LRP
                        if self.randomLogitLRPBlock:
                             outputs['LRP'+module]={'input': net.LRPBlock(x=x,
                                                                 y=outputs['output'+module],
                                                                 label=indices)}
                        else:
                             outputs['LRP'+module]={'input': net.LRPBlock(x=x,
                                                                 y=outputs['output'+module],
                                                                 label=None)}
                else:
                    if self.randomLogitLRPBlock:
                         outputs['LRP'+module]={'input': net.LRPBlock(x=x,
                                                             y=outputs['output'+module],
                                                             label=indices)}
                    else:
                         outputs['LRP'+module]={'input': net.LRPBlock(x=x,
                                                             y=outputs['output'+module],
                                                             label=None)}
                
                if self.HLP:
                    outputs['LRP'+module]={'input':outputs['LRP'+module]}
                    for key in reversed(net.LRPBlock.storedRelevance):
                        outputs['LRP'+module][key]=net.LRPBlock.storedRelevance[key].output
        #release memory by cleaning all values saved in hooks for LRP
        try:
            ISNetFunctions.CleanHooks(net.LRPBlock.model) 
        except:
            pass
        ISNetFunctions.CleanHooks(ReLUHooks) 
        
        
        #print(module,net.classifier.bias.data)
        #try:
        #    for i,img in enumerate((outputs['LRPFlexTeacher']['input'].detach()),0):
        #        print(img.shape)
        #        save(img.squeeze(),Name=str(i))
        #        saveImg(mask[i].squeeze(),Name='Mask'+str(i))
        #        saveImg(x[i].squeeze(),Name='x'+str(i))
        #        saveImg(tmp[i].squeeze(),Name='xOri'+str(i))
        #        saveImg(tmp[i].squeeze(),Name='xOri'+str(i))
        #except:
        #    pass
        
        return outputs,indices
    
    def SaveImage(self,image,module):
        image=image.permute(1,2,0).cpu().numpy()*255
        import os
        import cv2
        os.makedirs('SanityModel',exist_ok=True)
        cv2.imwrite('SanityModel/'+module+str(self.sanityCount)+'.png',image)
    
    def DefineEps(self,x):
        
        if self.epsLow==self.epsHigh and globals.pencentageEpsZero==0:
            globals.e=10**self.epsLow
        else:
            #random choice of epsilon (per element)
            #globals.e should be a list with the minimum and maximum eps
            expoents=torch.FloatTensor(x.shape[0]).\
            uniform_(self.epsLow,self.epsHigh).type_as(x)
            ones=torch.ones(expoents.shape).type_as(x)
            globals.e=torch.pow(ones*10,expoents)
            if globals.pencentageEpsZero>0:
                LRP0=torch.bernoulli(ones*globals.pencentageEpsZero)
                globals.e=globals.e*(1-LRP0)
    
    def returnBackbone(self,network='student'):
        #returns student
        if self.mode=='online':
            model=self.model
        else:
            if network=='teacher':
                model=self.teacher
            else:
                model=self.student
            
        if hasattr(model,'LRPBlock'):
            delattr(model,'LRPBlock')
        if hasattr(model,'ZbModule'):
            delattr(model,'ZbModule')  
        if hasattr(model,'chain'):
            delattr(model,'chain')  
        if hasattr(model,'firstLayerOutput'):
            delattr(model,'firstLayerOutput')  
        if hasattr(model,'firstLayerLinearOutput'):
            delattr(model,'firstLayerLinearOutput')  
        if hasattr(model,'firstLayerBNOutput'):
            delattr(model,'firstLayerBNOutput')  
        
        ISNetFunctions.remove_all_forward_hooks(model)
        return model

                
def InsertFeaturesHook(teacher,student):
    if ('ModifiedResNet' in teacher.__class__.__name__):#CLIP
        try:
            featureMapStudent=ISNetFunctions.HookFwd(student.layer4[-1].intermediateExtra,
                                                     mode='output')
        except:
            featureMapStudent=ISNetFunctions.HookFwd(student.layer4,mode='output')
        featureMapTeacher=ISNetFunctions.HookFwd(teacher.layer4,mode='output')
    elif ('DenseNet' in teacher.__class__.__name__):
        featureMapTeacher=ISNetFunctions.HookFwd(teacher.features.fReLU,mode='output')
        featureMapStudent=ISNetFunctions.HookFwd(student.features.fReLU,mode='output')
    elif ('ResNet' in teacher.__class__.__name__):
        featureMapTeacher=ISNetFunctions.HookFwd(teacher.layer4,mode='output')
        featureMapStudent=ISNetFunctions.HookFwd(student.layer4,mode='output')
    elif('VGG' in teacher.__class__.__name__ or 'Sequential' in teacher.__class__.__name__):
        featureMapTeacher=InsertFeaturesHookSingle(teacher)
        featureMapStudent=InsertFeaturesHookSingle(student)
    return featureMapTeacher,featureMapStudent

def InsertFeaturesHookSingle(model):
    if ('DenseNet' in model.__class__.__name__):
        featureMap=ISNetFunctions.HookFwd(model.features.fReLU,mode='output')
    elif ('ResNet' in model.__class__.__name__):
        featureMap=ISNetFunctions.HookFwd(model.layer4,mode='output')
    elif('VGG' in model.__class__.__name__):
        featureMap=ISNetFunctions.HookFwd(model.features,mode='output')
    elif ('Sequential' in model.__class__.__name__):#get last convolutional map
        last_conv_layer = None
        for name, layer in model.named_children():
            if isinstance(layer, torch.nn.Conv2d):
                last_conv_layer = name
        featureMap=ISNetFunctions.HookFwd(getattr(model,last_conv_layer),mode='output')
    return featureMap



def find_last_relu_layers(model):
    last_relu_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            parent_module_name = name.rsplit(".", 2)[0]  # Get the parent module name
            if parent_module_name.startswith("layer"):
                last_relu_layers[parent_module_name] = name
    return last_relu_layers

def FindLastResNetReLUs(model):
    # Get the names of the last ReLU layers in each residual layer
    last_relu_layers = find_last_relu_layers(model)
    tmp=[]
    for key in last_relu_layers:
        tmp.append(last_relu_layers[key])
    return tmp

def GetReLUBeforePoolVGG(model):
    layers=[]
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MaxPool2d):
            number=int(name[name.rfind('.')+1:])-1
            layers.append('features.'+str(number))
    return layers

def LRPFlexStoreActivations(model,ReLUHooks,layers=None):
    """
    Insert forward hooks, which store the activations for supervised layers in LRP deep supervision.
    
    Args:
        model: PyTorch model
        ReLUHooks: Dictionary of denoise ReLU hooks
        layers: list of names of supervised layers, optional
        
    Return Value:
        dictionary with layer names as keys and instantiated hooks
    """
    if ('ResNet' in model.__class__.__name__):
        chosen=FindLastResNetReLUs(model)
    if ('VGG' in model.__class__.__name__):
        chosen=GetReLUBeforePoolVGG(model)
    
    hooks={}
    for name in ReLUHooks:
        if layers is not None:
            if name in layers:
                ReLUHooks[name][1].saveLRPout=True
                hooks[name]=ReLUHooks[name][1]
        elif ('DenseNet' in model.__class__.__name__):
            if (name=='features.fReLU' or \
                (('transition' in name) and (name.count('.')==2) and ('LRP' not in name) \
                 and name[-4:]=='relu')):
                ReLUHooks[name][1].saveLRPout=True
                hooks[name]=ReLUHooks[name][1]
        elif ('ResNet' in model.__class__.__name__):
            if name in chosen:
                ReLUHooks[name][1].saveLRPout=True
                hooks[name]=ReLUHooks[name][1]
        elif ('VGG' in model.__class__.__name__):
            if name in chosen:
                ReLUHooks[name][1].saveLRPout=True
                hooks[name]=ReLUHooks[name][1]
        else:
            raise ValueError('Please specify layer names for all intermediate layers optimized with hidden layer penalization, passing a list as the layer variable')
            
    #print('Hooked lrp Relevance at the output of:',hooks.keys())

    return hooks

def RemoveInplace(net,architecture):
    #print(net)
    #print(net.__class__.__name__)
    #remove inplace operations from DNN architecture
    if ( ( ('DenseNet' in net.__class__.__name__) or ('ResNet' in net.__class__.__name__) ) and
           ('ModifiedResNet' not in net.__class__.__name__) ):
        net2=getBackbone(architecture,dropout=DropoutPresent(net),
                         classes=CountClasses(net))
        net2.load_state_dict(net.state_dict())
    else:
        net2=net
        
    #print(net)
    ChangeInplace(net2)
    
    return net2

def getBackbone(architecture,dropout,classes,pretrained=False):
    if pretrained:
        clss=classes
        classes=1000
    
    if (architecture=='densenet121'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet121(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet161'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet161(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet169'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet169(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet201'):
        import LRPDenseNetZe as LRPDenseNet
        classifierDNN=LRPDenseNet.densenet201(pretrained=pretrained,
                                              num_classes=classes)
    elif (architecture=='densenet264'):
        import LRPDenseNetZe as LRPDenseNet
        if(pretrained):
            raise ValueError('No available pretrained densenet264')
        classifierDNN=LRPDenseNet.densenet264(pretrained=False,
                                              num_classes=classes)
    elif (architecture=='resnet18'):
        import resnet
        classifierDNN=resnet.resnet18(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet34'):
        import resnet
        classifierDNN=resnet.resnet34(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet50'):
        import resnet
        classifierDNN=resnet.resnet50(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet101'):
        import resnet
        classifierDNN=resnet.resnet101(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet152'):
        import resnet
        classifierDNN=resnet.resnet152(pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet18FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet18(num_classes=classes)
    elif (architecture=='resnet34FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        classifierDNN=torch.hub.load('pytorch/vision:v0.10.0', 
                                     'resnet34', pretrained=pretrained,
                                     num_classes=classes)
    elif (architecture=='resnet50FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet50(num_classes=classes)
    elif (architecture=='resnet101FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet101(num_classes=classes)
    elif (architecture=='resnet152FixUpZeroBias'):
        import ZeroBiasResNetFixUp
        if pretrained:
            raise ValueError('No pretrained weights for ResNet with fixup and 0 bias')
        classifierDNN=ZeroBiasResNetFixUp.fixup_resnet152(num_classes=classes)
    elif (architecture=='vgg11'):
        import torchvision
        classifierDNN=torchvision.models.vgg11(pretrained=pretrained,
                                                    num_classes=classes)
    elif (architecture=='vgg11_bn'):
        import torchvision
        classifierDNN=torchvision.models.vgg11_bn(pretrained=pretrained,
                                                       num_classes=classes)    
    elif (architecture=='vgg16'):
        import torchvision
        classifierDNN=torchvision.models.vgg16(pretrained=pretrained,
                                                    num_classes=classes) 
    elif (architecture=='vgg16_bn'):
        import torchvision
        classifierDNN=torchvision.models.vgg16_bn(pretrained=pretrained,
                                                       num_classes=classes) 
    elif (architecture=='vgg13'):
        import torchvision
        classifierDNN=torchvision.models.vgg13(pretrained=pretrained,
                                                    num_classes=classes)
    elif (architecture=='vgg13_bn'):
        import torchvision
        classifierDNN=torchvision.models.vgg13_bn(pretrained=pretrained,
                                                       num_classes=classes) 
    elif (architecture=='vgg19'):
        import torchvision
        classifierDNN=torchvision.models.vgg19(pretrained=pretrained,
                                                    num_classes=classes) 
    elif (architecture=='vgg19_bn'):
        import torchvision
        classifierDNN=torchvision.models.vgg19_bn(pretrained=pretrained,
                                                       num_classes=classes) 
    else:
        raise ValueError('Architecture must be densenet, resnet or VGG')

        
    if pretrained:
        if ('ResNet' in classifierDNN.__class__.__name__):
            classifierDNN.fc=nn.Linear(classifierDNN.fc.in_features,clss)
        elif ('DenseNet' in classifierDNN.__class__.__name__):
            classifierDNN.classifier=nn.Linear(classifierDNN.classifier.in_features,clss)
        elif ('VGG' in classifierDNN.__class__.__name__):
            classifierDNN.classifier[-1]=nn.Linear(classifierDNN.classifier[-1].in_features,clss)
        else:
            raise ValueError('backbone not implemented for ImageNet pretraining')
    if dropout:
        if ('DenseNet' in classifierDNN.__class__.__name__):
            classifierDNN.classifier=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                   classifierDNN.classifier)
        elif ('ResNet' in classifierDNN.__class__.__name__):
            classifierDNN.fc=nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                                   classifierDNN.fc)
        else:
            raise ValueError('Unrecognized backbone')

    if (('ResNet' in classifierDNN.__class__.__name__) and
        hasattr(classifierDNN, 'maxpool')):
        classifierDNN.maxpool.return_indices=True
        classifierDNN.maxpool=nn.Sequential(
            OrderedDict([('maxpool',classifierDNN.maxpool),
                         ('special',ISNetLayers.IgnoreIndexes())]))
        
    
    return classifierDNN

def CountClasses(net):
    if ('DenseNet' in net.__class__.__name__):
        try:
            classes=net.classifier.weight.data.shape[0]
        except:#with dropout
            classes=net.classifier[1].weight.data.shape[0]
    elif ('ResNet' in net.__class__.__name__):
        classes=net.fc.weight.data.shape[0]
    else:
        raise ValueError('Classes counter unrecognized backbone')
    return classes

def DropoutPresent(model):
    if isinstance(model, nn.Dropout):
        return True

    for module in model.children():
        if DropoutPresent(module):
            return True

    return False                         
                         
def RandomLogitIndices(outputs):
    indices=[]
    for i,_ in enumerate(outputs,0):
        if random.randint(1,10)>5:#50% chance of propagating highest logit
            indices.append(torch.argmax(outputs[i],dim=0))
        else:#50% chance of propagating other random logit
            randChoice=random.randint(0,outputs.shape[1]-1)
            while randChoice==torch.argmax(outputs[i],dim=0):
                randChoice=random.randint(0,outputs.shape[1]-1)
            indices.append(randChoice)
    return indices
        
def mask_image(img,mask,mode,img2=None,mask2=None):
        #erase background (forge and refine stages)
        if mode=='privilegedDomain':#image (named mask) from a privileged domain, no need to apply mask
            #print('HERE')
            return mask
        x=torch.mul(img,mask)
        if mode=='zero':
            pass
        elif mode=='noise':
            #use uniform noise as masked images' backgrounds
            #Imask: 1 in background, 0 in foreground
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            noise=torch.rand(x.shape).type_as(x)
            noise=torch.mul(Imask,noise)
            x=x+noise
        elif mode=='noiseGrey':
            #use uniform noise as masked images' backgrounds
            #noise in greyscale
            #Imask: 1 in background, 0 in foreground
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            noise=torch.rand(x[:,0,:,:].shape).type_as(x)
            noise=noise.unsqueeze(1).repeat(1,x.shape[1],1,1)
            noise=torch.mul(Imask,noise)
            x=x+noise
        elif mode=='white':
            Imask=torch.ones(mask.shape).type_as(mask)-mask
            x=x+Imask
        elif mode=='recombine':
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
                
def CleanNet(teacher):
    try:
        teacher=teacher.getBackbone()
    except:
        teacher=teacher
    try:
        ISNetFunctions.remove_all_forward_hooks(teacher,backToo=True)
    except:
        pass
    try:
        delattr(teacher,'LRPBlock')
    except:
        pass
    return teacher
        
def freeze(teacher,student,freezeLastLayer,zeroBias,freezeTeacher=True):
        #freeze teacher:
        if freezeTeacher:
            for param in teacher.parameters():
                param.requires_grad = False
            
        #do not freeze student
        for param in student.parameters():
            param.requires_grad = True
        
        if freezeLastLayer:
            #freeze student classifier layer, copy teacher's classifier
            if ('DenseNet' in student.__class__.__name__):
                try:
                    if student.classifier.weight.data.shape==teacher.classifier.weight.data.shape:
                        student.classifier.weight.data=teacher.classifier.weight.data.clone()
                        try:
                            student.classifier.bias.data=teacher.classifier.bias.data.clone()
                        except:
                            pass
                    else:
                        raise ValueError('freezeLastLayer not implemented for different teacher and student architectures in DenseNet')
                        
                except:#with dropout
                    if student.classifier.weight.data.shape==teacher.classifier.weight.data.shape:
                        student.classifier[1].weight.data=teacher.classifier[1].weight.data.clone()
                        try:
                            student.classifier[1].bias.data=teacher.classifier[1].bias.data.clone()
                        except:
                            pass
                    else:
                        raise ValueError('freezeLastLayer not implemented for different teacher and student architectures in DenseNet')
                            
                            
                for param in teacher.classifier.parameters():
                    param.requires_grad = False
                for param in student.classifier.parameters():
                    param.requires_grad = False
            elif ('ResNet' in student.__class__.__name__):
                if student.fc.weight.data.shape==teacher.fc.weight.data.shape:
                    student.fc.weight.data=teacher.fc.weight.data.clone()
                    try:
                        student.fc.bias.data=teacher.fc.bias.data.clone()
                    except:
                        pass
                else:
                    student.fc=copy.deepcopy(teacher.fc)
                    student.layer4[-1]=nn.Sequential(OrderedDict([
                        ('a',student.layer4[-1]),
                        ('b',nn.Conv2d(in_channels=student.layer4[-1].conv1.in_channels,
                                                  out_channels=teacher.layer4[-1].conv1.in_channels,
                                                  kernel_size=(1,1)))])
                    )
                for param in teacher.fc.parameters():
                    param.requires_grad = False
                for param in student.fc.parameters():
                    param.requires_grad = False
                        
            elif ('vgg' in student.__class__.__name__):
                if student.classifier[-1].weight.data.shape!=\
                teacher.classifier[-1].weight.data.shape:
                    raise ValueError('freezeLastLayer not implemented for different teacher and student architectures in VGG')
                student.classifier[-1].weight.data=teacher.classifier[-1].weight.data.clone()
                try:
                    student.classifier[-1].bias.data=teacher.classifier[-1].bias.data.clone()
                except:
                    pass
                for param in student.classifier[-1].parameters():
                    param.requires_grad = False
                for param in teacher.classifier[-1].parameters():
                    param.requires_grad = False
            elif ('Sequential' in student.__class__.__name__):
                if student[-1].weight.data.shape!=teacher[-1].weight.data.shape:
                    raise ValueError('freezeLastLayer not implemented for different teacher and student architectures in Sequential')
                student[-1].weight.data=teacher[-1].weight.data.clone()
                try:
                    student[-1].bias.data=teacher[-1].bias.data.clone()
                except:
                    pass
                for param in student[-1].parameters():
                    param.requires_grad = False
                for param in teacher[-1].parameters():
                    param.requires_grad = False
            else:
                raise ValueError('Unrecognized backbone')
            
        if zeroBias:
            teacher,student=ZeroTheBias(teacher,student)
        
        return teacher,student

def ZeroTheBias(teacher,student):
    i=0
    for name, param in teacher.named_parameters():
        if 'bias' in name:
            param.requires_grad = False
            #print('bias:',param)
            param.data.fill_(0.0)
            #print('bias:',param)
            i+=1
    for name, param in student.named_parameters():
        if 'bias' in name:
            param.requires_grad = False
            #print('bias:',param)
            param.data.fill_(0.0)
            #print('bias:',param)
            i+=1
    print('Biases det to zero and frozen:', i)
    return teacher,student
            
def CreateLRPBlock(teacher,e,Zb,rule,multiple,selective,highest,detach,randomLogit,storeRelevance):
    if ('DenseNet' in teacher.__class__.__name__):
        teacher.LRPBlock=ISNetLayers._LRPDenseNet(teacher,e=e,Zb=Zb,rule=rule,
                                               multiple=multiple,positive=False,
                                               ignore=None,selective=selective,
                                               highest=highest,detach=detach,
                                               storeRelevance=storeRelevance,
                                               FSP=False,randomLogit=randomLogit)
    elif ('FixupResNet' in teacher.__class__.__name__):                
        teacher.LRPBlock=ISNetLayers._LRPResNetFixUp(teacher,e=e,Zb=Zb,rule=rule,
                                               multiple=multiple,positive=False,
                                               ignore=None,selective=selective,
                                               highest=highest,detach=detach,
                                               amplify=1,
                                               FSP=False,storeRelevance=storeRelevance,
                                               randomLogit=randomLogit)
    elif ('ResNet' in teacher.__class__.__name__):
        #if FSP:
        #    raise ValueError('not implemented')
        if (hasattr(teacher, 'maxpool') and \
        teacher.maxpool.__class__.__name__=='MaxPool2d'):
            teacher.maxpool.return_indices=True
            teacher.maxpool=nn.Sequential(OrderedDict([('maxpool',teacher.maxpool),
                                                ('special',ISNetLayers.IgnoreIndexes())]))
        teacher.LRPBlock=ISNetLayers._LRPResNet(teacher,e=e,Zb=Zb,rule=rule,
                                               multiple=multiple,positive=False,
                                               ignore=None,selective=selective,
                                               highest=highest,
                                               amplify=1,detach=detach,
                                               storeRelevance=storeRelevance,
                                               FSP=False,randomLogit=randomLogit)
    elif ('Sequential' in teacher.__class__.__name__):
        teacher.LRPBlock=ISNetLayers._LRPSequential(teacher,e=e,Zb=Zb,rule=rule,
                                               multiple=multiple,selective=selective,
                                               highest=highest,
                                               amplify=1,detach=detach,
                                               storeRelevance=storeRelevance,
                                               inputShape=SequentialInputShape,
                                               preFlattenShape=SequentialPreFlattenShape,
                                               randomLogit=randomLogit)

    elif('VGG' in teacher.__class__.__name__):
        teacher.LRPBlock=ISNetLayers._LRPVGG(teacher,e=e,Zb=Zb,rule=rule,
                                          multiple=multiple,selective=selective,
                                          highest=highest,
                                          amplify=1,detach=detach,
                                          randomLogit=randomLogit,
                                          storeRelevance=storeRelevance)
    else:
        raise ValueError('Unrecognized backbone')
    return teacher

def ChangeInplace(m: torch.nn.Module):
    #function to remove inplace operations

    children = dict(m.named_children())
    output = {}
    if hasattr(m,'inplace'):
        m.inplace=False
    if children == {}:
        return (m)
    else:
        for name, child in children.items():
            try:
                output[name] = ChangeInplace(child)
            except TypeError:
                output[name] = ChangeInplace(child)
    return output

class DenoiseReLU():
    def __init__(self, module, saveLRPout=False):
            #self.hook = module.register_backward_hook(self.hook_fn)
            self.hook_fwd = module.register_forward_hook(self.hook_f)
            self.hook_back = module.register_full_backward_hook(self.hook_b)
            self.saveLRPout=saveLRPout
            #self.identifier=random.randint(1,99999)
    #def hook_fn(self, module, input, output):
    def hook_b(self, module, grad_input, grad_output):
        #print(grad_input[0].shape,grad_output[0].shape)
        #print('out:',grad_output[0][0,0])
        #print('in:',grad_input[0][0,0])
        #print('Started backward hook ',self.identifier)
        #if self.output is not None:
        #    print('Output is:',self.output[0][0])
        #else:
        #    print('Output is None')
        if globals.LRP:
            if self.saveLRPout:
                self.outLRP=grad_output[0]*self.output
            if torch.is_tensor(globals.e):
                if (not (globals.e.sum().item()==0)):
                    #one per batch element
                    e=globals.e
                    for i in list(range(len(self.output.shape)-1)):
                        e=e.unsqueeze(-1)

                    #use LRP-0 where eps is zero
                    LRP0=1-torch.heaviside(e,torch.zeros(e.shape).type_as(self.output))

                    #LRP0 in denominator
                    #of the attenuated step just avoids nans wherever we will use LRP0
                    atStep=self.output/(self.output+e+LRP0)
                    grad=torch.mul(atStep,grad_output[0])
                    if torch.isnan(grad).any():
                        warnings.warn('NaN in LRP backpropagated quantity (G), substituted by 0')
                        grad=torch.nan_to_num(grad)#nans become 0, 0/0=0

                    if LRP0.sum().item()>0:#there are LRP-0 elements
                        grad=(1-LRP0)*grad+\
                        LRP0*torch.nan_to_num(grad_input[0],nan=0.0,posinf=0.0,neginf=0.0)

                    grad=(grad,)
                    self.output = None#clean memory
                    return grad

            elif isinstance(globals.e, float):#int
                if (globals.e>0):
                    atStep=self.output/(self.output+globals.e)
                    grad=torch.mul(atStep,grad_output[0])
                    if torch.isnan(grad).any():
                        warnings.warn('NaN in LRP backpropagated quantity (G), substituted by 0')
                        grad=torch.nan_to_num(grad)#nans become 0, 0/0=0
                    grad=(grad,)
                    self.output = None#clean memory
                    return grad
                elif (globals.e<0):
                    raise ValueError('Negative epsilon in LRP')
                elif (globals.e==0):
                    pass #do not alter gradient propagation, use LRP-0/gradient*input
                
            else: 
                raise ValueError('Unrecognized globals.e type, should be tensor or float, and it is ',\
                                 globals.e)
        
        self.output = None#clean memory
        #print('I cleaned memory:',self.identifier)
            
    def hook_f(self, module, input, output):
        if globals.LRP:
            self.output = output.clone()
            #print('I saved the output, hook identifier:',self.identifier)
            #print('Output is:',self.output[0][0])
            
        else: 
            self.output = None
            #print('I saved the output, hook identifier:',self.identifier)
            #print('Output is None')
            #print('I erased the output')
            
    def clean(self):
        self.output = None
        if self.saveLRPout:
            self.outLRP = None
            
    def close(self):
        self.hook_fwd.remove()
        self.hook_back.remove()
        
class SaveOutput():
    def __init__(self, module):
        """
        Hook for LRP-Zb.

        Saves layer's activation.

        Args:
            module: PyTorch layer
        """
        self.hook_fwd = module.register_forward_hook(self.hook_f)
    def hook_f(self, module, input, output):
        self.output = output.clone()
    def close(self):
        self.hook_fwd.remove()
        self.hook_back.remove()

def InsertReLUHooks(m: torch.nn.Module,oldName=''):
    
    children = dict(m.named_children())
    output = {}
    if children == {}:
        #m.register_forward_hook(AppendBoth)
        #l=globals.LayerIndex
        #globals.LayerIndex=globals.LayerIndex+1
        #return (m,l)
        if m.__class__.__name__=='ReLU':
            return (m,DenoiseReLU(m))
        else:
            return None
    else:
        for name, child in children.items():
            if oldName=='':
                tmp=InsertReLUHooks(child,oldName=name)
            else:
                tmp=InsertReLUHooks(child,oldName=oldName+'.'+name)
            if tmp is not None:
                if oldName=='':
                    output[name] = tmp
                else:
                    output[oldName+'.'+name] = tmp
    return output

class ZbRuleConvInput (nn.Module):
    def __init__(self,layer,l=0,h=1,op=None):
        super().__init__()
        self.layer=layer
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aK):
        if e==0:
            Zb0=True
        else:
            Zb0=False
            
        try:
            y= ISNetFunctions.ZbRuleConvInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                                 e=globals.e,l=self.l,h=self.h,op=self.op,Zb0=Zb0)
        except:#legacy, if self.op is missing
            y= ISNetFunctions.ZbRuleConvInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                                 e=globals.e,l=self.l,h=self.h,Zb0=Zb0)
        return y
    
class ZbRuleDenseInput (nn.Module):
    def __init__(self,layer,l=0,h=1,op=None):
        super().__init__()
        self.layer=layer
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aK):
        y= ISNetFunctions.ZbRuleDenseInput(layer=self.layer,rK=rK,aJ=aJ,aK=aK,
                             e=globals.e,l=self.l,h=self.h)
        return y
    
class ZbRuleConvBNInput (nn.Module):
    def __init__(self,layer,BN,l=0,h=1,op=None):
        super().__init__()
        self.layer=layer
        self.BN=BN
        self.l=l
        self.h=h
        
    def forward (self,rK,aJ,aKConv,aK):
        try:
            y=ISNetFunctions.ZbRuleConvBNInput(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,
                                       aKConv=aKConv,e=globals.e,
                                       l=self.l,h=self.h)  
        except:#legacy, if self.op is missing
            y=ISNetFunctions.ZbRuleConvBNInput(layer=self.layer,BN=self.BN,rK=rK,aJ=aJ,aK=aK,
                                       aKConv=aKConv,e=globals.e,
                                       l=self.l,h=self.h)  
        return y
    
def ColorHeatmap(heatmap, sepNorm=False):
    import matplotlib
    red = {'red':     [[0.0,  0.0, 0.0],
                       [1.0,  1.0, 1.0]],
             'green': [[0.0,  0.0, 0.0],
                       [1.0,  0.0, 0.0]],
             'blue':  [[0.0,  0.0, 0.0],
                       [1.0,  0.0, 0.0]]}

    blue = {'red':    [[0.0,  0.0, 0.0],
                       [1.0,  0.0, 0.0]],
             'green': [[0.0,  0.0, 0.0],
                       [1.0,  0.0, 0.0]],
             'blue':  [[0.0,  0.0, 0.0],
                       [1.0,  1.0, 1.0]]}
    
    redMap = matplotlib.colors.LinearSegmentedColormap('red', red, N=256)
    blueMap = matplotlib.colors.LinearSegmentedColormap('blue', blue, N=256)

    if sepNorm:
        redHeatmap=redMap(heatmap/np.max(heatmap))
        blueHeatmap=blueMap(-heatmap/np.max(-heatmap))
    else:
        maximum=np.max([np.max(heatmap),np.max(-heatmap)])
        redHeatmap=redMap(heatmap/maximum)
        blueHeatmap=blueMap(-heatmap/maximum)

    heatmap=redHeatmap+blueHeatmap
    
    heatmap=heatmap[:,:,:-1]
    
    return heatmap 

def ColorHeatmap2(heatmap):
    import matplotlib

    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('br',["b", "w", "r"], N=256) 
    
    #center heatmap aroung 0.5, the white value
    maximum=np.max(np.abs(heatmap))+1e-5
    redMap=(heatmap[:,:,0]/maximum)/2+0.5
    blueMap=0.5-(heatmap[:,:,2]/maximum)/2
    
    heatmap=np.where(heatmap[:,:,0]>0,redMap,blueMap)

    heatmap=cmap(heatmap)
    
    
    heatmap=heatmap[:,:,:-1]
    
    return heatmap

def save(image,Name='',mode=''):
    import cv2
    import os
    import matplotlib.pyplot as plt
    os.makedirs('SanityCheckRunning/'+mode,exist_ok=True)
    image=image.squeeze(1).squeeze(0).sum(dim=0).cpu().numpy()
    heatmap=ColorHeatmap(image,sepNorm=True)
    heatmap=ColorHeatmap2(heatmap)
    fig=plt.figure()
    plt.axis('off')
    #plt.imshow(np.ones(heatmap.shape))
    plt.imshow(heatmap)
    ##fig.set_size_inches(9, 9)
    plt.savefig('SanityCheckRunning/'+mode+'/image'+Name+'.png',
                    bbox_inches='tight')

    #image=image.swapaxes(0,1)
    #image=image.swapaxes(1,2)
    #image=image*255
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #os.makedirs('SanityCheckRunning/'+mode,exist_ok=True)
    #cv2.imwrite('SanityCheckRunning/'+mode+'/image'+Name+'.png',image)
    
def saveImg(image,Name='',mode=''):
    import cv2
    import os
    import matplotlib.pyplot as plt
    #image=image*255
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    os.makedirs('SanityCheckRunning/'+mode,exist_ok=True)
    #cv2.imwrite('SanityCheckRunning/'+mode+'/image'+Name+'.png',image)
    image=image.squeeze(0).detach()
    if len(image.shape)==3:
        image=image.permute(1, 2, 0)
    image=image.cpu().numpy()
    fig=plt.figure()
    plt.axis('off')
    #plt.imshow(np.ones(heatmap.shape))
    plt.imshow(image)
    ##fig.set_size_inches(9, 9)
    plt.savefig('SanityCheckRunning/'+mode+'/image'+Name+'.png',
                    bbox_inches='tight')
    
def LoadClip(architecture,text_inputs,device,norm=True):
    import clip
    import torchvision.transforms as transforms
    model, preprocess = clip.load(architecture, device=torch.device("cpu"),
                                  download_root='.')
    model=model.to(device)
    text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).cpu()
    teacher=model.visual.cpu()
    if norm:
        preprocess = transforms.Compose([trans for trans in list(preprocess.transforms) \
         if (isinstance(trans, transforms.Resize) or isinstance(trans, transforms.CenterCrop) \
             or isinstance(trans, transforms.Normalize))])
    else:
        preprocess = transforms.Compose([trans for trans in list(preprocess.transforms) \
         if (isinstance(trans, transforms.Resize) or isinstance(trans, transforms.CenterCrop))])
    return teacher,preprocess,text_features
    
class CLIPOutput(nn.Module):
    def __init__(self,text_features,trainableT=False,trainableW=False,teachereT=100.0,
                 norm=True,bias=False,applyT=True):
        super().__init__()
        self.text_features=text_features
        text_features_norm = text_features/text_features.norm(dim=-1, keepdim=True)
        self.text_features_norm=nn.Parameter(text_features_norm,requires_grad=trainableW)
        if applyT:
            self.t=nn.Parameter(torch.log(torch.tensor(teachereT)),requires_grad=trainableT)
        else:
            self.t=None
        self.norm=norm
        if bias:
            self.b=nn.Parameter(torch.zeros(text_features_norm.shape[0]).type_as(text_features_norm),
                                requires_grad=trainableW)
        else:
            self.b=None
            
    def forward(self, x):
        self.embeddings=x
        if self.norm:
            y = x/x.norm(dim=-1, keepdim=True)
        else:
            y=x
        if self.t is not None:
            logit = (torch.exp(self.t) * y @ self.text_features_norm.T)
        else:
            logit = y @ self.text_features_norm.T
            
        if self.b is not None:
            logit = logit+self.b
            #print(self.b)
        
        #print(logit)
        #soft=logit.softmax(dim=-1)
        #pred=torch.argmax(soft)
        return logit