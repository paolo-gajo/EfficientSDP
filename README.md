## Experiment 0:
Reproduce MTRFG.
'augment_train': 0,
'augment_val': 0,
'augment_test': 0,
'augment_k_train': 0,
'augment_k_val': 0,
'augment_k_test': 0,
'keep_og_train': 1, (irrelevant since no augmentation)
'keep_og_val': 1, (irrelevant since no augmentation)
'keep_og_test': 1, (irrelevant since no augmentation)

val_results_best_f1={'P': 0.6618, 'R': 0.6318, 'F1': 0.6465}
test_results_f1={'P': 0.6093, 'R': 0.5888, 'F1': 0.5989}


## Experiment 1.1:
Attempt to improve the performance with data augmentation.
Augment train and keep validation/test fixed.
Keep the original samples even if they do not have more than one topological sorting.

config:
'augment_train': 1,
'augment_val': 0,
'augment_test': 0,

'augment_k_train': [1, 5, 10, 20, 100],
'augment_k_val': 0,
'augment_k_test': 0,

'keep_og_train': 1,
'keep_og_val': 1, (irrelevant since no augmentation)
'keep_og_test': 1, (irrelevant since no augmentation)

k = 1
val_results_best_f1={'P': 0.6621, 'R': 0.64, 'F1': 0.6508}
test_results_f1={'P': 0.6122, 'R': 0.5888, 'F1': 0.6003}

k = 5
val_results_best_f1={'P': 0.6611, 'R': 0.6532, 'F1': 0.6572}
test_results_f1={'P': 0.6048, 'R': 0.598, 'F1': 0.6014}

k = 10
val_results_best_f1={'P': 0.6457, 'R': 0.6217, 'F1': 0.6335}
test_results_f1={'P': 0.6159, 'R': 0.5947, 'F1': 0.6051}

k = 20
val_results_best_f1={'P': 0.6414, 'R': 0.6293, 'F1': 0.6353}
test_results_f1={'P': 0.6068, 'R': 0.602, 'F1': 0.6044}

k = 100
val_results_best_f1={'P': 0.6513, 'R': 0.6324, 'F1': 0.6417}
test_results_f1={'P': 0.6001, 'R': 0.5795, 'F1': 0.5897}


## Experiment 1.2:
Attempt to improve the performance with data augmentation.
Augment train and keep validation/test fixed.
Do NOT keep the original samples for train if they do not have more than one topological sorting. This way we try to show that we need less gold data and can just augment to match the original performance.

This sounds like a really cool problem where you try to match the 'complexity' of the graphs in your training data with the validation/testing data. Especially useful when you have little data.

config:
'augment_train': 1,
'augment_val': 0,
'augment_test': 0,

'augment_k_train': [1, 5, 10, 20, 100],
'augment_k_val': 0,
'augment_k_test': 0,

'keep_og_train': 0,
'keep_og_val': 1, (irrelevant since no augmentation)
'keep_og_test': 1, (irrelevant since no augmentation)

k = 1
val_results_best_f1={'P': 0.6242, 'R': 0.5958, 'F1': 0.6097}
test_results_f1={'P': 0.5959, 'R': 0.5723, 'F1': 0.5838}

k = 5
val_results_best_f1={'P': 0.6343, 'R': 0.6091, 'F1': 0.6214}
test_results_f1={'P': 0.5898, 'R': 0.5743, 'F1': 0.5819}

k = 10
val_results_best_f1={'P': 0.6552, 'R': 0.623, 'F1': 0.6387}
test_results_f1={'P': 0.6058, 'R': 0.5822, 'F1': 0.5937}

k = 20
val_results_best_f1={'P': 0.6302, 'R': 0.6211, 'F1': 0.6256}
test_results_f1={'P': 0.5817, 'R': 0.5756, 'F1': 0.5786}

k = 100
val_results_best_f1={'P': 0.6391, 'R': 0.6286, 'F1': 0.6338}
test_results_f1={'P': 0.5845, 'R': 0.571, 'F1': 0.5776}


## Experiment 1.3:
Repeat 1.1 with val augmented to k = 100.

config:
'augment_train': 1,
'augment_val': 1,
'augment_test': 0,

'augment_k_train': [1, 5, 10, 20, 100],
'augment_k_val': 100,
'augment_k_test': 0,

'keep_og_train': 1,
'keep_og_val': 1,
'keep_og_test': 1, (irrelevant since no augmentation)

k = 1
val_results_best_f1={'P': 0.6318, 'R': 0.6096, 'F1': 0.6205}
test_results_f1={'P': 0.614, 'R': 0.5954, 'F1': 0.6046}

k = 5
val_results_best_f1={'P': 0.6375, 'R': 0.6297, 'F1': 0.6336}
test_results_f1={'P': 0.6109, 'R': 0.6, 'F1': 0.6054}

k = 10
val_results_best_f1={'P': 0.6381, 'R': 0.6118, 'F1': 0.6247}
test_results_f1={'P': 0.601, 'R': 0.5795, 'F1': 0.5901}

k = 20
val_results_best_f1={'P': 0.6377, 'R': 0.6233, 'F1': 0.6304}
test_results_f1={'P': 0.6121, 'R': 0.6, 'F1': 0.606}

k = 100
val_results_best_f1={'P': 0.6244, 'R': 0.6086, 'F1': 0.6164}
test_results_f1={'P': 0.6035, 'R': 0.5947, 'F1': 0.5991}


## Experiment 1.4:
Repeat 1.2 with val augmented to k = 100.

config:
'augment_train': 1,
'augment_val': 1,
'augment_test': 0,

'augment_k_train': [1, 5, 10, 20, 100],
'augment_k_val': 100,
'augment_k_test': 0,

'keep_og_train': 0,
'keep_og_val': 1,
'keep_og_test': 1, (irrelevant since no augmentation)

k = 1
val_results_best_f1={'P': 0.6062, 'R': 0.5794, 'F1': 0.5925}
test_results_f1={'P': 0.5959, 'R': 0.5723, 'F1': 0.5838}

k = 5
val_results_best_f1={'P': 0.6202, 'R': 0.6015, 'F1': 0.6107}
test_results_f1={'P': 0.5986, 'R': 0.5769, 'F1': 0.5876}

k = 10
val_results_best_f1={'P': 0.6422, 'R': 0.6107, 'F1': 0.626}
test_results_f1={'P': 0.6058, 'R': 0.5822, 'F1': 0.5937}

k = 20
val_results_best_f1={'P': 0.6192, 'R': 0.6019, 'F1': 0.6104}
test_results_f1={'P': 0.5941, 'R': 0.571, 'F1': 0.5823}

k = 100
val_results_best_f1={'P': 0.5892, 'R': 0.5761, 'F1': 0.5825}
test_results_f1={'P': 0.5752, 'R': 0.5703, 'F1': 0.5728}


## Experiment 2:
Attempt to improve the performance by:
1. integrating a GNN in the model

with a gnn in the model we shouldn't need to use data augmentation in the train and be able to handle permutations in the validation/testing

## Experiment 2.1-4:
Repeat 1.1, 1.2, 1.3, 1.4 with gnn=1

## Experiment 2.5:

keep train fixed and augment validation

config:
'augment_train': 0,
'augment_val': 1,
'augment_test': 0,

'augment_k_train': 1,
'augment_k_val': [1, 5, 10, 20, 40, 60, 80, 100],
'augment_k_test': 0,

'keep_og_train': 1,
'keep_og_val': 1,
'keep_og_test': 1, (irrelevant since no augmentation)

2. using relative positional encodings between the step nodes
LEARNING EFFICIENT POSITIONAL ENCODINGS WITH GRAPH NEURAL NETWORKS: https://arxiv.org/pdf/2502.01122
Comparing Graph Transformers via Positional Encodings: https://arxiv.org/pdf/2402.14202

MPNN model:
treat the step reps as a fully connected graph where the adj matrix is the attention output of the encoder model, averaged/maxed across the L dim for the token reps belonging to the same step

EDGE inference:
rather than considering the fully connected attention, calculate new weighted edges between the steps and use them to weigh the different blocks of the attention

Experiment 3:
use different latent graph inference model:
1. [Differentiable Graph Module (DGM) for Graph Convolutional Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9763421): during training DGM takes A_gold and X to get a better A. during inference it will take only X as the input.
2. [LATENT GRAPH INFERENCE USING PRODUCT MANIFOLDS](https://openreview.net/pdf?id=JLR_B7n_Wqr)
3. [Latent Graph Inference with Limited Supervision](https://proceedings.neurips.cc/paper_files/paper/2023/file/67101f97dc23fcc10346091181fff6cb-Paper-Conference.pdf)

## Questions:
1. Do we care about augmenting the test set to show robustness to permutation?

TODO: Bert no pos embeddings and gnn
TODO (maybe): use word-level GNNs instead of LSTMs in the tagger and biaffine parser