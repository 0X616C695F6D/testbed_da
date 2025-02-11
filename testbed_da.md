Target: Eo. December
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
+ Deep CORAL: Correlation alignment for unsupervised domain adaptation


Tasks:
+ Better technique? https://github.com/zhaoxin94/awesome-domain-adaptation
    + Maybe zero-shot, one-shot
+ Add more modulations schemes


Ideas:
+ New idea: new class in target domain; what happens then? can you calculate the
  discrepancy between the labelled data in source&target and unlabelled in
  target? has this been done before?
  + Looks like its been done before: unknown class, or remove signal, or
    adversarially calculate discrepancy like I initially predicted...
+ Add Resnet & CNN


Goals:
+ Journal on domain adaptation in RFML
+ Real world testbed for the VTC dataset. Domain adaptation for SNR and other
  parameters (doppler, etc)


Questions:
+ Can I somehow use the new TorchSig paper? It has inbuilt ML inference on GR
+ Getting accurate SNR on a real testbed is not possible, should we mix or try
  our best to categorize samples
+ Should we use additional datasets besides radioml 2018.01a and real-world?
    + radioml 2016? Others?
    + Torchsig to produce datasets?
+ How many local/global DA methods?
    + G is dann, cycada. L is mcd and stochastic.


Issues:


Done:
+ Add 24, 26 SNR
+ Validate the new adaptation techniques
+ Build 3 PSK OTA Tx-Rx & generate data
+ Verify current implementations are correct
    + Correlation alignment for deep domain adaptation (CORAL)
    + Stochastic classifier (STAR)
    + Maximum classifier discrepancy (MCD)
    + Domain adversarial neural network (DANN) - VTC24
    + Baseline - VTC24
+ Test codebase - make sure it works
+ Modularize codebase fast
