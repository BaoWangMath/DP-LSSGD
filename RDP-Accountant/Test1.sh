#/bin/sh
python train_dplssgd_mnist.py --EPOCH 500 --SIGMA 0.0 > dplssgd_sigma0.0.txt
python train_dplssgd_mnist.py --EPOCH 500 --SIGMA 0.1 > dplssgd_sigma0.1.txt
python train_dplssgd_mnist.py --EPOCH 500 --SIGMA 0.2 > dplssgd_sigma0.2.txt
python train_dplssgd_mnist.py --EPOCH 500 --SIGMA 0.5 > dplssgd_sigma0.5.txt
