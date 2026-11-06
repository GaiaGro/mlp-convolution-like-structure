# Final Experimental Protocol

## Research question

To what extent can a fully connected residual MLP trained on images develop
properties associated with convolutional models, despite having no
convolutional weight sharing or spatially structured layers?

The project studies four complementary aspects:

1. spatial locality in first-layer weights;
2. reuse of learned local patterns across image positions;
3. robustness and prediction stability under image translations;
4. similarity between internal MLP and CNN representations.

These properties are analysed separately rather than combined into a single
convolution-likeness score.

## Dataset and split

CIFAR-10 is used throughout the project.

The official 50,000-image training set is split with stratified sampling and
split seed 2026 into:

- 45,000 training images;
- 5,000 validation images.

The official 10,000-image test set is used for final evaluation and the CKA
analysis.

Images are normalized using the CIFAR-10 mean and standard deviation recorded
in `configs/experiment.yaml`.

## Models

### Residual MLP

The MLP receives the full 32 x 32 RGB image and flattens it directly. It uses:

- input dimension 3 x 32 x 32;
- hidden width 512;
- three residual MLP blocks;
- expansion ratio 2;
- LayerNorm;
- GELU activations;
- dropout 0.1;
- a linear 10-class classifier.

The MLP contains no convolution, attention, pooling, patchification or
architectural weight sharing.

### Small CNN reference

The reference CNN uses:

- a 3 x 3 convolutional stem;
- channel widths [64, 128, 256, 384];
- residual blocks per stage [1, 1, 2, 1];
- BatchNorm and GELU;
- adaptive global average pooling;
- a linear 10-class classifier.

The CNN was chosen to remain within 10 percent of the MLP parameter count.

## Training conditions

Five MLP conditions are evaluated:

1. `baseline`: cross-entropy on the original image;
2. `exposure`: mean cross-entropy on original and translated views;
3. `consistency`: exposure plus prediction consistency and
   variance-covariance regularization;
4. `augmentation`: cross-entropy after standard image augmentation;
5. `augmentation_consistency`: standard augmentation combined with the full
   consistency objective.

The CNN is trained under two conditions:

1. `baseline`;
2. `augmentation`.

With five final seeds, this produces 25 final MLP runs and 10 final CNN runs,
for 35 final trained models in total.

## Image transformations

Standard augmentation consists of:

- random 32 x 32 crop with reflect padding of 4 pixels;
- random horizontal flip.

Translation training uses one axial shift sampled from:

- horizontal shifts of -2, -1, +1 or +2 pixels;
- vertical shifts of -2, -1, +1 or +2 pixels.

The boundary rule is sampled from reflect, replicate and circular padding.

For `augmentation_consistency`, the translated view is generated from the same
augmented image.

Final translation evaluation uses distances 1, 2 and 4 pixels under reflect,
replicate and circular padding. Distance 4 is not used during validation-based
model selection.

## Consistency objective

For exposure:

`L_exposure = 0.5 CE(x, y) + 0.5 CE(Tx, y)`.

For consistency:

`L_consistency = L_exposure + lambda JS(p(x), p(Tx)) + rho R(z, z_T)`.

`R` contains the variance-floor and covariance penalties applied to the
penultimate representations of the original and translated views. The
variance target is 1.0 and the relative covariance coefficient is 0.04.

The original and translated branches use paired stochastic dropout masks so
that their difference reflects the image transformation rather than
independent dropout noise.

The selected consistency coefficients used for final training are:

- lambda = 7.5;
- rho = 0.2.

## Development and optimization

Development seeds 11 and 22 are used for optimizer, epoch-budget and
consistency-weight selection. They are separate from the five final seeds.

The optimizer search compares the candidate learning rates and weight decays
specified in `configs/experiment.yaml`. The selected settings are:

- MLP: AdamW, learning rate 0.0003, weight decay 0.0001;
- CNN: AdamW, learning rate 0.001, weight decay 0.0001.

The baseline MLP epoch-budget comparison evaluates 100, 200 and 300 epochs.
The selected final budget is 100 epochs and is then used for every final MLP
and CNN condition.

All final runs use:

- batch size 256;
- cosine learning-rate decay;
- 10 warmup epochs;
- label smoothing 0.1;
- mixed precision on CUDA when available.

The consistency development grid evaluates:

- lambda in {0.5, 1.0, 2.0, 5.0, 7.5, 10.0};
- rho in {0, 0.05, 0.1, 0.2};
- both development seeds.

Candidates are evaluated using clean validation accuracy, shifted validation
accuracy, effective rank and active-dimension retention. The selected values
are lambda = 7.5 and rho = 0.2.

## Final seeds and checkpoints

Final training uses seeds:

- 43;
- 44;
- 45;
- 46;
- 47.

Checkpoint selection uses validation performance under the same rule for all
conditions. The test set is not used to select hyperparameters or checkpoints.

## Final performance evaluation

Each of the 35 final checkpoints is evaluated on the full CIFAR-10 test set.
The evaluation records:

- clean accuracy and clean loss;
- shifted accuracy for distances 1, 2 and 4;
- results separately for reflect, replicate and circular padding;
- mean shifted accuracy;
- worst-padding shifted accuracy;
- distance-based accuracy AUC;
- prediction consistency between clean and translated inputs.

Per-example predictions are saved for statistical comparisons and diagnostic
visualizations.

## Performance statistics

The final paired comparisons are:

1. exposure minus baseline;
2. consistency minus exposure;
3. augmentation minus baseline;
4. augmentation-consistency minus consistency;
5. CNN baseline minus MLP baseline.

The training seed is the experimental unit. Results include paired differences
across the five final seeds and 95 percent confidence intervals.

Clean prediction differences are additionally analysed using exact McNemar
tests with Holm correction across the predefined comparison family.

## First-layer structural analysis

Structural analysis is performed on the MLP parameter
`input_projection.weight`, reshaped for each neuron into RGB image coordinates
with shape 3 x 32 x 32.

Squared weight magnitude is used as spatial energy.

### Spatial locality

For each neuron, the fraction of total first-layer weight energy contained in
the highest-energy spatial window is measured for window sizes:

- 3 x 3;
- 5 x 5;
- 7 x 7;
- 9 x 9.

The primary locality metric is the maximum-window energy fraction for a 7 x 7
window.

### Translated-template reuse

For each first-layer neuron, the maximum-energy 7 x 7 RGB weight patch is
extracted. A bank of 16 local templates is learned using sign-invariant
spherical k-means.

The primary template-reuse analysis uses:

- patch size 7;
- 16 templates;
- 5 folds;
- 5 repetitions;
- 10 k-means initializations;
- maximum 100 iterations;
- tolerance 1e-6;
- analysis seed 2026.

Template banks are fitted on training folds of neurons and evaluated on held-out
neurons. For each held-out neuron, every spatial translation of every learned
template is considered. The primary score is the fraction of the neuron's total
weight energy reconstructed by its best translated template match.

## Structural controls

The structural metrics are evaluated against three controls.

### Random initialization

A matched randomly initialized MLP is generated for each final seed.

### Spatial shuffle

For each trained MLP condition and seed, the spatial positions of first-layer
weights are shuffled independently within each neuron while preserving RGB
triplets. Twenty shuffle repetitions are used with analysis seed 2026.

Shuffle repetitions are averaged within seed and condition before paired
seed-level inference.

### Pixel-permutation training

Twenty-five additional MLPs are trained using a fixed spatial pixel permutation
(seed 0), covering all five MLP conditions and all five final training seeds.
RGB triplets are preserved and the permutation is applied after image-space
transformations.

Structural metrics are reported both in the permuted model coordinates and
after remapping the learned weights back to the original image coordinates.

## Structural statistics

Primary structural inference uses the five training seeds as the experimental
units.

For both primary structural metrics, trained MLPs are compared with:

- matched random initialization;
- the within-seed mean spatial-shuffle control;
- pixel-permutation weights in model coordinates;
- pixel-permutation weights remapped to original-image coordinates.

Paired seed differences and 95 percent confidence intervals are reported.
Neurons and shuffle repetitions are not treated as independent experimental
units.

## Structural sensitivity analysis

The primary 7 x 7 / 16-template definition is supplemented by four one-factor
sensitivity variants:

1. 5 x 5 patches with 16 templates;
2. 9 x 9 patches with 16 templates;
3. 7 x 7 patches with 8 templates;
4. 7 x 7 patches with 32 templates.

The same trained checkpoints and controls are reused. No additional model
training is performed for these sensitivity analyses.

## Descriptive spatial-template visualization

A separate descriptive analysis is used to visualize learned template families
across final seeds.

For each MLP training condition:

1. the maximum-energy 7 x 7 patch is extracted from each of the 512 first-layer
   neurons in each of the five final seeds;
2. the resulting 2,560 patches are pooled within condition;
3. one 16-template sign-invariant spherical-k-means bank is fitted using the
   same primary template parameters;
4. every neuron's complete 32 x 32 first-layer weight map is matched against all
   translated positions of all 16 templates;
5. the best template identity, spatial position and squared match score are
   recorded.

The resulting assignments are used for two descriptive visualizations:

- a gMLP-style figure showing filters associated with each condition's dominant
  local template across different spatial positions;
- a 16-family atlas for the `augmentation_consistency` condition, split into
  families 1-8 and 9-16, showing the learned template and the strongest matching
  neuron from each final seed.

These visualizations complement, but do not replace, the cross-validated
quantitative template-reuse analysis.

## Representational similarity with linear CKA

Linear centered kernel alignment (CKA) is used to compare internal MLP and CNN
representations on the same 10,000 clean CIFAR-10 test images.

Five MLP representation points are analysed:

- Projection;
- Block 1;
- Block 2;
- Block 3;
- Output norm.

Seven CNN representation points are analysed:

- Stem;
- Stage 1;
- Stage 2;
- Stage 3a;
- Stage 3b;
- Stage 4;
- Global pool.

CNN convolutional feature maps are global-average-pooled over their spatial
coordinates before CKA so that each image is represented by one feature vector.
The analysis therefore measures image-level representational similarity rather
than formal spatial equivariance.

For each final seed, every MLP condition is compared with both the baseline CNN
and the augmentation-trained CNN from the same seed. This produces:

`5 seeds x 5 MLP conditions x 2 CNN conditions x 5 MLP layers x 7 CNN layers = 1,750 CKA values`.

For every layer pair, the five seed-level CKA values are summarized by their
mean and 95 percent confidence interval. Best-matching CNN layers are also
recorded for each MLP layer and condition.

## Confusion-matrix visualization

Clean CIFAR-10 confusion matrices are computed from the saved final test
predictions for:

- MLP baseline;
- MLP augmentation-consistency;
- CNN baseline;
- CNN augmentation.

Each seed-specific matrix is row-normalized by true class and the five matrices
are then averaged, so every displayed cell is the mean percentage of examples
from one true class assigned to one predicted class across final seeds.

## Figures generated from the final analyses

The repository contains figures covering:

- clean and translation robustness across MLP and CNN conditions;
- pixel-permutation geometry;
- spatial concentration of first-layer weights;
- primary locality and template-reuse results;
- structural controls;
- structural sensitivity;
- spatial template reuse and template-family atlases;
- MLP-CNN representational similarity using CKA;
- clean CIFAR-10 confusion matrices.

## Scope of the conclusions

The project does not treat the MLP as becoming a CNN and does not combine all
measurements into a single convolution-likeness score. Instead, locality,
local-pattern reuse, translation robustness and representational similarity are
kept as distinct measurements of different convolution-like properties.

Translation experiments measure robustness and prediction stability under
image shifts; they are not presented as a direct formal measurement of
translation equivariance.
