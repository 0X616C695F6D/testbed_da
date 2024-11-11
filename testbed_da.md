Target: Nov 30
Papers:
+ Domain Generalization: A Survey K. Zhou
  Methods for DA and DG. Useful for finding new DA methods
+ Maximum Classifier Discrepancy for Unsupervised Domain Adaptation K. Saito
  Domain adaptation method that utilizes distribution of individual classes
  in source and target domains. Simple terms, focus on samples that are far
  away then utilize the generator in producing only those samples. You want to
  steer away from samples that are too close to the boundary.
    1. Train discriminators (classifiers) F1, F2 to maximize discrepancy given
       target features.
    2. train the generator to fool discriminator by min. the discrepancy.
+ Stochastic classifiers for unsupervised domain adaptation (STAR) 
  Similar to MCD - we still max/min discrepancy between source and target domain
  but we use multi classifiers instead of just two. These classifiers are
  derived from gaussian distribution with RV parameters mu and sigma. Simply
  take the first layer weight vector, do multiple distributions and assume that
  is another classifier.


Tasks:
+ Research domain adaptation techniques
    + Correlation alignment for deep domain adaptation (CORAL)
    + CyCADA
+ Build gnuradio for generating dataset & evaluate generated samples


Ideas:
+ New idea: new class in target domain; what happens then? can you calculate the
  discrepancy between the labelled data in source&target and unlabelled in
  target? has this been done before?
  + Looks like its been done before: unknown class, or remove signal, or
    adversarially calculate discrepancy like I initially predicted...


Goals:
+ Journal on domain adaptation in RFML
+ Real world testbed for the VTC dataset. Domain adaptation for SNR and other
  parameters (doppler, etc)


Questions:
+ Should we use additional datasets besides radioml 2018.01a and real-world?
    + radioml 2016? Others?
+ How many local/global DA methods?
    + G is dann, cycada. L is mcd and stochastic.


Issues:


Done:
+ Stochastic classifier (STAR)
+ Maximum classifier discrepancy (MCD)
+ Test codebase - make sure it works
+ Modularize codebase fast
