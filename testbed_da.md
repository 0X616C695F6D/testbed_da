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



Tasks:
+ Research domain adaptation techniques
    + MCD
+ Modularize codebase fast
+ Test codebase - make sure it works
+ Build gnuradio for generating dataset
+ Test our VTC method on real world dataset

Ideas:


Goals:
+ Journal on domain adaptation in RFML
+ Real world testbed for the VTC dataset. Domain adaptation for SNR and other
  parameters (doppler, etc)


Questions:
+ Should be use additional datasets besides radioml 2018.01a and real-world?


Issues:


Done:


