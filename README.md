# lagrangean-s3vm
a python 2.7 implementation of lagrangean-s3vm

*the core routine is in lagrangian_s3vm.py; see method documentation for details

*utils.py contains:
- a cross-validation utility method (basically wrapping that of sklearn); lagrangean-s3vm uses it for initializing a supervised model on the labeled training set
- a method which loads the COIL20 dataset

*datasets/ contains a convenient dump (a few MBs) of the COIL20 dataset, which can be used for testing

*test.ipynb is a jupyter notebook, showing how to use each method on a COIL20 classification instance

#####################################

please cite us if you use this software!

@article{Bagattini18_a,
title={A Simple and Effective Lagrangean-based Combinatorial Algorithm for {S3VM}s},
author={F. Bagattini and P.Cappanera and F. Schoen},
journal={Proceedings of the Third International Conference on Machine Learning, Optimization and Big Data},
year={forthcoming 2018},
publisher={Springer}
}

@article{Bagattini18_b,
title={Lagrangean-based Combinatorial Optimization for Large Scale {S3VM}s},
author={F. Bagattini and P.Cappanera and F. Schoen},
journal={IEEE Transactions of Neural Networks and Learning Systems},
year={forthcoming 2018},
publisher={IEEE}
}
