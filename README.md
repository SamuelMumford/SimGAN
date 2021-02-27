# SimGAN

Files needed to run the training for the SimGAN and CycleGAN models (with data excluded)

The main training file is SimGanTrainCL.py, used to train using terminal command line arguments on a AWS GPU. I train with batches of 64 images, saving every 50 iterations and displaying progress every 25.

The Face2MaskTrain/Face2MaskTest folders are where data would be.
PicCheck is where outputs are intermittently stored during training to look at transformed pictures.
Cache stores weight files, and outputs stores saved pictures used to calculate FID scores on a trained model.
