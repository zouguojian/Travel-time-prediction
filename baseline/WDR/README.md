# WDR

* WDR model has three main blocks: 1) the wide model is similar to the wide model 
in Wide & Deep network. We use a second order cross-product transformation followed
 by an affine transformation to get a 256 dimensional output; 2) the deep model embeds
  the sparse features into a 20 dimensional space, then processes the concatenated features
   by a 3-hidden-layer MLP with a ReLU [14] activation to get a 256 dimensional output. 
   The size of all the three hidden layers in the MLP is 256; 3) the recurrent model is a 
   variant of standard RNN. The feature of each road segment is first projected into a 256 
   dimensional space by a fully connected layer with ReLU as the activation function. The 
   transformed feature is then fed into to a standard LSTM with cell size 256. 
   
