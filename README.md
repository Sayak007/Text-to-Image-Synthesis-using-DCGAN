# PRINCIPLES OF DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS (DC-GAN) AND ITS APPLICATION IN TEXT-IMAGE SYNTHESIS

This is a Tensorflow implementation of synthesizing images. The images are synthesized using the GAN-CLS Algorithm. This implementation is built on top of the excellent DCGAN in Tensorflow.
The architecture used is given below:

![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Images/img2.jpg)
Image Source: [Generative Adversarial Text-to-Image Synthesis Paper](https://arxiv.org/abs/1605.05396)

The main objective function used is:

![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Images/img1.jpg)<br /> 
G = Generator, D = Discriminator, Pd (x) = distribution of real data, P(z) = distribution of generator, 
x = sample from Pd (x), z = sample from P(z), D(x) = Discriminator network, G(z) = Generator network

## Datasets
We executed our algorithm on the Oxford-102 flower dataset [Ref. 6]. For the Oxford-102 dataset, it has 102 classes, which contains 82 training classes and 20 test classes. Each of the images in the dataset has 10 corresponding text descriptions. We have used the DC-GAN architecture for training the model on this dataset and we have done twice, once for 100 epochs with one set of captions and another for 200 epochs with a different set of captions. 
<br/>Source: [Oxford-102 flower dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## Software Specs
* Python 3 or above
* Anaconda/ Jupyter/ Google Colab
* TensorFlow 
* Tensor Layer
* scikit-learn : for skip thought vectors
* NLTK : for skip thought vectors
* Numpy 

## Hardware used
* Processor: AMD Ryzen 5 2400G
* Clock Speed: 3.6 GHz
* Graphics: RX Vega 11 Graphics
* RAM: 8GB DDR4
* RAM Speed: 2933 MHz
* Hard Drive: 1 TB SATA
* OS: Windows 10 pro
* Architecture: 64bit

## Parameters
* z_dim: Noise Dimension. Default is 100.
* t_dim: Text feature dimension. Default is 256.
* batch_size: Batch Size. Default is 64.
* image_size: Image dimension. Default is 64.
* gf_dim: Number of conv in the first layer generator. Default is 64.
* df_dim: Number of conv in the first layer discriminator. Default is 64.
* gfc_dim: Dimension of gen untis for for fully connected layer. Default is 1024.
* caption_vector_length: Length of the caption vector. Default is 1024.
* data_dir: Data Directory. Default is Data/.
* learning_rate: Learning Rate. Default is 0.0002.
* beta1: Momentum for adam update. Default is 0.5.
* epochs: Max number of epochs. Default is 100/200.
* resume_model: Resume training from a pretrained model path.
* data_set: Data Set to train on. Default is flowers.

## Algorithm
* Import Dataset
* Save the following  as pickles:
  * Caption Pickle – Merge captions with image in a vector format
  * Image Test, Train Pickle – Train-Test Split of the Pickle
  * vocab Pickle – Synthesizing of the words in the captions
* Load data from pickle
* Define main_train :
  * CNN encoding
  * Defining Loss functions
  * Loop until number of epochs:
    * Noise to Generator:
      * Batch Normalization
      * Leaky ReLU
      * Convolution
      * Transpose Convolution
    * Discriminator:
      * Inputs the real image pickle data
      * Inputs the generated samples from generator G(z)
      * Leaky ReLU
      * Batch Normalization
      * Convolutions(4x)
    * Decision Making Real or Fake:
      * If fake the discriminator output serves as a feedback input for the generator in the next epoch
    * Print the generated image for nth epoch, along with the Generator losses, discriminator losses and RNN-losses.
  * End loop

## Experimental Results:
### 100 epochs with one set of captions.
In this experiment, the same dataset has been trained for 100 epochs using a set of captions (texts) as noted below and accordingly the following images are generated from them. Here each epoch took roughly 40 minutes execution time and total 100 epochs took 70 hours that is almost 3 days.
#### Captions:
* the flower shown has yellow anther red pistil and bright red petals.
* this flower has petals that are yellow, white and purple and has dark lines
* the petals on this flower are white with a yellow center
* this flower has a lot of small round pink petals.
* this flower is orange in color, and has petals that are ruffled and rounded.
* the flower has yellow petals and the center of it is brown.
* this flower has petals that are blue and white.
* these white flowers have petals that start off white in color and end in a white towards the tips.
#### Result:
![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Result/train_100.png)

### 200 epochs with another set of captions.
In this experiment the same dataset has been trained for 200 epochs using a set of captions(texts) as noted below and accordingly the following images are generated from them. Here each epoch took roughly 1hour of execution time and total 200 epochs took almost 10 days.
#### Captions:
the flower shown has purple and white petals
* this flower has petals that are yellow and purple and has dark lines
* the petals on this flower are red with a yellow center
* this flower has a lot of small round orange petals
* this flower is dark orange in color, and has petals that are ruffled and rounded
* the flower has yellow petals and the center of it is brown
* this flower has petals that are yellow and white
* these flowers have pink colored petals

#### Result:
![image](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Result/train_200.png)

## Report
* [Project Report](https://github.com/Sayak007/Text-to-Image-Synthesis-using-DCGAN/blob/main/Project%20Report.pdf)

## References

* Generative Adversarial Nets by Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio; Part of Advances in Neural Information Processing Systems 27 (NIPS 2014).
* Generate the corresponding Image from Text Description using Modified GAN-CLS Algorithm by Fuzhou Gong and Zigeng Xia ; University of Chinese Academy of Sciences; arXiv:1806.11302v1 [cs.LG] 29 Jun 2018.
* Generative Adversarial Text to Image Synthesis by Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, Honglak Lee ; Neural and Evolutionary Computing (cs.NE); Computer Vision and Pattern Recognition ; ICML 2016.
* Skip-Thought Vectors by Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler; Computation and Language (cs.CL); Machine Learning (cs.LG) ; arXiv:1506.06726.
* [Tensor flow Implementation of Deep Convolutional Generative Adversarial Networks](https://www.tensorflow.org/tutorials/generative/dcgan)
* Automated flower classification over a large number of classes, Nilsback, M-E. And Zisserman, A.; Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008).
* [DCGAN Torch Code](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
