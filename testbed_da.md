Target: Nov 30
Papers:
+ Maximum Classifier Discrepancy for Unsupervised Domain Adaptation K. Saito
  Domain adaptation method that utilizes distribution of individual classes
  in source and target domains. Simple terms, focus on samples that are far
  away then utilize the generator in producing only those samples. You want to
  steer away from samples that are too close to the boundary.
    1. Train discriminators (classifiers) F1, F2 to maximize discrepancy given
       target features.
    2. train the generator to fool discriminator by min. the discrepancy.
+ Domain Generalization: A Survey K. Zhou
  Methods for DA and DG. Useful for finding new DA methods



Tasks:
+ Research domain adaptation techniques
    + Stochastic classifiers for unsupervised domain adaptation (STAR)
    + Correlation alignment for deep domain adaptation (CORAL)
+ Build gnuradio for generating dataset & evaluate generated samples

Ideas:


Goals:
+ Journal on domain adaptation in RFML
+ Real world testbed for the VTC dataset. Domain adaptation for SNR and other
  parameters (doppler, etc)


Questions:
+ Should be use additional datasets besides radioml 2018.01a and real-world?


Issues:


Done:
+ Modularize codebase fast
+ Test codebase - make sure it works
+ Maximum classifier discrepancy
