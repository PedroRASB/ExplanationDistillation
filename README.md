# ExplanationDistillation

Requirements: Install CLIP (https://github.com/openai/CLIP) and all requirements in the LRP-Flex implementation of layer-wise relevance propagation, on which our LRP procedure is based (https://github.com/PedroRASB/FasterISNet). 

Files:
OfflineStudentZVariableEpsTorch.py: main library for explanation distillation, based on pytorch
OfflineStudentLightningTrainer.py: Pytorch Lightning implementation of explanation distillation. Used in all our distillation experiments, when training the student
OfflineTeacherLightningTrainer.py: Pytorch Lightning implementation used to train the teacher
ISNetFunctionsZe.py, globalsZe.py, ISNetLayersZe.py, globalsZe.py, resnet.py, LRPDenseNet.py, unet.py: supporting files, based on LRP-Flex, 


