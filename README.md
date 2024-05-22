# ExplanationDistillation

Requirements: CLIP (https://github.com/openai/CLIP), PyTorch (1.11.0), PyTorch Lightning (1.6.3), Python (3.9),
torchvision (0.12.0), matplotlib (3.5.1), numpy (1.21.5), h5py (3.7.0), scikit-image (0.19.2), scikit-learn (0.23.2), scipy (1.7.3), pandas (1.4.2).


Files:

OfflineStudentZVariableEpsTorch.py: main library for explanation distillation, based on pytorch

OfflineStudentLightningTrainer.py: Pytorch Lightning implementation of explanation distillation. Used in all our distillation experiments, when training the student

OfflineTeacherLightningTrainer.py: Pytorch Lightning implementation used to train the teacher

ISNetFunctionsZe.py, globalsZe.py, ISNetLayersZe.py, globalsZe.py, resnet.py, LRPDenseNet.py, unet.py: supporting files, based on LRP-Flex, 


