#/bin/sh
python train_dpsvrg_mnist.py --EPOCH 100 --NOISE_MULTIPLIER 1.1 > dpsvrg_Noise1.1.txt
python train_dpsvrg_mnist.py --EPOCH 100 --NOISE_MULTIPLIER 1.3 > dpsvrg_Noise1.3.txt
python train_dpsvrg_mnist.py --EPOCH 100 --NOISE_MULTIPLIER 1.5 > dpsvrg_Noise1.5.txt


