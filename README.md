# NN-Invariant-Elements

Research on identifying  some  of  the  invariant  elements  of  Neural  Networks  to geometric transformations using image data. For the purposes of this research, I have used 3 widely used datasets, but only the work on the CIFAR-10 dataset has been uploaded as the approach on the other datasets is very similar. 

Image Transformations that our models' robustness have been tested against:
- translation
- rotation
- horizontal & vertical flipping
- shear

All images have had their size increased to 56 x 56 size. We achieved this by extending the background size of the images by filling the empty space with zeros. This approach allows  for  different  positional  observations  without  the  objects,  or  any part of them, being moved outside of the boundaries of the image.

7 different models have been used, specifically;
- Multilayer Perceptron (MLP) - used as the baseline model
- (Standard) Convolutional Neural Network (CNN)
- CNN with max pooling layers
- CNN with average pooling layers
- CNN with dropout layers (0.3)
- CNN with a global pooling layer
- CNN with a global average pooling layer

The architecture of the models has been loosely based on the one found on Tensorflow's and Keras' tutorial page. Moreover, data augmentation was employed to investigate the improvement in performance of the MLP and standard CNN models. No significant regularization was used in any of the models, as our objective was just to investigate the degree of invariance of each model.
