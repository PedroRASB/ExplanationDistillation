#global variables for the ISNet
global LayerIndex
global X
global XI
global t
global mean_l
global mean_L
global Ml
global detach
global LRP
global e
global pencentageEpsZero

#First definition
LayerIndex=0
X=[]
XI=[]
t=0
mean_l=0
mean_L=0
Ml=0
detach=True

#FastLRP
LRP=False
e=0.01
pencentageEpsZero=0

#Vision transformer
W=[]