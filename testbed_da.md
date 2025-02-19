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
+ Evaluate when trained and tested on target (for OTA) - this should be golden
  performance. Then compare with the graph we have for OTA SNR-to-SNR
+ Cross-validation: Batch dataset and make sure to validate all of them
+ https://scikit-learn.org/1.5/auto_examples/decomposition/plot_ica_blind_source_separation.html
+ Vary number of source SNR and plot 'X' with target for comprehesive results
    + So decrease SNR, run, decrease, run, etc.
+ Ablation study; vary target dataset size(1%, 5%, 10%)


Ideas:
+ GAN-based model, generate target and map back to sim
+ Transformer, convolution+transformer model
+ Signal -> image, try CNN on the image
    + Has anyone performed image-compute on spectrum analyzer output?
+ Try these libraries out
    + https://github.com/KevinMusgrave/pytorch-adapt
    + https://github.com/thuml/Transfer-Learning-Library
    + https://github.com/facebookresearch/DomainBed
+ Introduce an unknown constellation on target


Goals:
+ Journal on domain adaptation in RFML
+ Adapt simulated to OTA raw IQ signals using many UDA techniques & models.
+ Validation of VTC paper, on real world collected data.


Questions:
+ Do we want to adapt to a higher SNR?
+ Can I somehow use the new TorchSig paper? It has inbuilt ML inference on GR
+ Getting accurate SNR on a real testbed is not possible, should we mix or try
  our best to categorize samples
+ Should we use additional datasets besides radioml 2018.01a and real-world?
    + radioml 2016? Others?
    + Torchsig to produce datasets?
+ How many local/global DA methods?
    + G is dann, cycada. L is mcd and stochastic.


Poor:
+ TSNE to visualize clusters.
    + why: should show clusters if SNR levels can be differentiated
    + maybe also do based on constellation not snr?


Done:
+ Add deep ResNet model for all domain adaptation techniques
+ Pairwise evaluation OTA vs Simulated
+ Collect longer OTA frame size
+ Collect longer simulated frame size
+ Fix normalization method to be accurate
    + why: plotting signal, sometimes X is 0.002 to -0.006, other times 6 to -6;
      we are doing global frame normalization (based on global max)
+ All DA technique comparison in comparison\_plot.html
+ Add joint adaptation networks
+ Add 24, 26 SNR
+ Collect SNR varying samples
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
