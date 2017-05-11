# lagrangean-s3vm
a python 2.7 implementation of lagrangean-s3vm

*the core routine is in lagrangian_s3vm.py; see method documentation for details

*utils.py contains:
- a cross-validation utility method (basically wrapping that of sklearn); lagrangean-s3vm uses it for initializing a supervised model on the labeled training set
- a method which loads the COIL20 dataset

*datasets/ contains a convenient dump (a few MBs) of the COIL20 dataset, which can be used for testing

*test.ipynb is a jupyter notebook, showing how to use each method on a COIL20 classification instance
