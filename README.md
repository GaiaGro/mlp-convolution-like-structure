# Do MLPs Learn Convolution-Like Structure on Images?

This project investigates whether a fully connected residual MLP can develop convolution-like properties on CIFAR-10 despite having no convolutional layers or architectural weight sharing.

The analysis focuses on four aspects:

1. spatial locality in first-layer weights;
2. reuse of learned local patterns across image positions;
3. robustness to image translations;
4. representational similarity with a parameter-matched CNN.

Five MLP training conditions are evaluated across five final seeds: `baseline`, `exposure`, `consistency`, `augmentation`, and `augmentation_consistency`. A matched CNN trained with and without augmentation provides the architectural reference.

The project includes structural controls, sensitivity analyses, template-family visualizations, linear CKA, translation evaluation and confusion-matrix diagnostics.

The complete methodology is documented in `docs/protocol.md`.
