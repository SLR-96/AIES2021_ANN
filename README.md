# AIES2021_ANN
Artificial neural networks project from the 2021 course on Artificial Intelligence and Expert Systems (AIES)


## Mask Wearing Detection Using Different AI Based Object Detection Algorithms

### Abstract
In this paper four different object detection artificial neural networks are used for a mask wearing detection task. Namely YOLO V4, SSD MobileNet V2, CenterNet MobileNet V2, and EfficientDet D0. In another task, the bounding boxes indicating the faces on dataset images are cropped and classified by training a MobileNet V2 and a VGG16 network. By comparing the four object detection networks used in this paper, it is shown that the YOLO V4 network is the best, and CenterNet MobileNet V2 is the second best choice for the task at hand. For the classification experiment, it is shown that both MobileNet V2 and VGG16 can show good results, but MobileNet V2 is better at handling data imbalance.

### I.	INTRODUCTION
Since the COVID-19 pandemic outbreak, it has been mandatory in many public places to wear a face mask. But monitoring mask wearing -especially in big crowds- is an energy and time-consuming task, sometimes requiring many people standing guard, while still not being as efficient as possible.

This is where artificial neural networks come in handy. Object detection algorithms utilizing convolutional neural networks have been in use for many years, and have been improved to great accuracies, sometimes surpassing even human capacity at specific tasks. Many of these object detection networks which have been trained on large diverse datasets and shown great accuracies, have been made available to public. One of these datasets is the Microsoft COCO dataset, containing photos of 91 objects types with a total of 2.5 million labeled instances in 328k images. [1] A comparison of the best object detection models was made in [2], from which the results of YOLO V4 and the models available in TensorFlow V2 were chosen to be shown on Table I. As it can be seen on [3] and [4], YOLO is one of the best state-of-the-art algorithms for object detection. While some may argue that currently there are algorithms with better speed and accuracy, there’s no question in the superiority of YOLO in accessibility and validity, since it has been proven on a wide range of different object detection tasks to have optimum results. And while YOLO V5 is currently available, it is still under development [4], and hasn’t been proven to have better results compared to YOLO V4. Therefore, in this paper YOLO V4 has been chosen to be compared with TensorFlow V2 models.

The dataset used in this paper, consists of 853 png format images of people wearing masks correctly, incorrectly, or not wearing at all, which make up 3 different classes for the object detection task. The annotations indicating the coordinates and the associated class of the boxes enclosing instances of the said three classes, have been created using the LabelImg tool, in the PASCAL VOC format. In total there are 3,232 instances of faces with mask, 717 without mask, and 123 with masks worn incorrectly, in the whole dataset. Width and height of the images are in the range of 156 to 600 pixels, mostly below 400.

![image](https://user-images.githubusercontent.com/65850584/221652821-977566d2-6903-492d-86c3-710662744937.png)
TABLE I. COMPARISON OF OBJECT DETECTION METHODS TRAINED ON MS COCO DATASET [2]

In the following sections, first the chosen methods and their parameters are explained in detail in section 2. Then the results of training these networks are shown on section 3. And finally a conclusion is made in section 4.

### II.	METHODS AND PARAMETERS
As mentioned above, the size of the images used in this dataset is relatively small (below 400 pixels on either width or height). Therefore, when analyzing the results of different methods from Table I, in order to choose methods to be used, those working well on smaller image sizes are compared against each other. Of course, other than image sizes, the precision and speed of each network is also considered, but not taken for granted, since the computational load of the backbone used can greatly affect the speed.

From Table I it is evident that YOLO V4 is potentially the best choice, considering both precision and speed, even with small image sizes. Comparing the rest, CenterNet seemingly has the advantage in precision, and among those with smaller image sizes, EfficientDet D0 and SSD show promising results in speed. Faster R-CNN on the other hand, doesn’t fare well on either speed or precision, and since it’s shown these results with a relatively light backbone architecture, it’s crossed out and is not used in the experiments of this paper.

For all methods used, the dataset is split into %20 test and %80 train. The following outlines the parameters and specifics of the four chosen object detection methods, namely YOLO V4, SSD, CenterNet, EfficientDet, and two image classification methods used on the cropped boxes, which utilize the VGG16 and MobileNet V2 models. The parameters chosen are either based on speculation, or trial and error.

#### A.	YOLO V4
Since YOLO V4 has been optimized on DarkNet, here as well, it has been used as the backbone. The annotations used in YOLO need to be in its own special format. Therefore, a conversion of annotations from PASCAL VOC to the YOLO format has been used beforehand. The input image size is chosen to be 416*416 since it works better with factors of 32. Batch size of 64 is chosen based on the maximum that the GPU can handle. The optimizer and loss function are custom made for YOLO V4. The activation function used for the first 70 layers of the network is Mish, and for the next 35 layers Leaky ReLU has been used. Learning rate is constant with a value of 0.001, chosen as to learn slowly in a stable fashion, and a 0.95 momentum is used in order to avoid local minima. These parameters have been used for 3000 steps, after which the momentum is decreased to 0.5, after the 3200th step it’s decreased to 0.2 and finally after the 3400th step it’s decreased to 0 in order to get the final stable results. The network is trained on 3500 steps, using the GPU by Google Colab.

#### B.	SSD
The backbone used for this network is MobileNet V2. MobileNet V2 is one of the most efficient CNNs, providing great speed and accuracy for small datasets, and makes trial and error easier and less time consuming. The input image size is chosen to be 400*400. Batch size of 16 is chosen based on the maximum that the GPU can handle. Adam optimizer is used with its learning rate scheduled with cosine decay and warmup phases, starting from 1e-6 and going up to 0.01 in 2000 steps and then going down to 0 in the 20,000th step, as depicted in Fig. 1. The warmup phase in the beginning prevents early overfitting by keeping the learning rate low until all the dataset has been used for training at least once. The cosine decay phase lets the network learn effectively in the beginning then gradually decreases the learning rate to prevent overfitting or loss divergence. Swish is used as the activation function, which is like ReLU but smoothed in x=0 point, which avoids a singularity point and prevents saturation on negative inputs. Weighted smooth L1 is used as the loss function for localization and weighted sigmoid focal is used for classification, since focal loss functions are able to handle data imbalance. Finally, the network is trained in 20k steps, using the GPU provided by Google Colab.

![image](https://user-images.githubusercontent.com/65850584/222807593-9b5ba74a-4b14-43cf-98cb-51775112fc9c.png)
Fig. 1. Cosine decay learning rate with a warmup phase of 2k steps for training in 20k steps

#### C.	CenterNet
MobileNet V2 is once again used as the backbone here, for the same reasons explained for SSD. The input image size is chosen to be 416*416. Batch size of 16 is chosen based on the maximum that the GPU can handle. Adam optimizer is used with a learning rate starting from 1e-8 and going up to 0.01 in the 3000 step warmup phase and going down to 0 in the 30,000th step, using cosine decay. Once again Swish is used as the activation function. L1 is used as the loss function for localization and penalty reduced logistic focal loss is used for classification. Finally, the network is trained in 30k steps, using an Nvidia GeForce GTX 960m GPU.

#### D.	EfficientDet
Since EfficientDet has been optimized to work on EfficientNet B0 for small image sizes, here as well it has been used as the backbone. The input image size is chosen to be 512*512, the size on which the network has been optimized. Batch size of 16 is chosen based on the maximum that the GPU can handle. Momentum optimizer is used with a momentum of 0.9 and learning rate starting from 1e-8 and going up to 0.01 in the 2000 step warmup phase and going down to 0 in the 20,000th step, using cosine decay. Once again Swish is used as the activation function. Weighted smooth L1 is used as the loss function for localization and weighted sigmoid focal loss is used for classification. Finally, the network is trained in 20k steps, using the GPU provided by Google Colab.

#### E.	Classification of cropped boxes
Since MobileNet V2 has been used in this research as the backbone of two of the object detection networks, it only seems appropriate to use it for the classification task as well. MobileNet V2 has also the advantage of being light, which makes the training process much faster, and prevents overfitting since our dataset is relatively small and not very diverse.

VGG16 is also used for comparison against MobileNet V2, since it’s one of the most generalizable and versatile pre-trained CNNs.

In the preprocessing stage, boxes enclosing the faces in the dataset are cropped and resized with scale to fit a 224*224 frame and the remaining pixels have the value 0, making the color black. Then they are divided into the three main classes, namely “worn mask incorrectly”, “with mask” and “without mask”. Some of the images from these three classes have been randomly chosen and plotted on Fig. 2, from which it is evident that the dataset has errors, and even the correct ones are low in quality. These three classes make up 123, 3232, and 717 images respectively, which is a greatly imbalanced dataset, and creates a problem in training the network. In order to face this problem, a few measures have been taken; firstly, the class containing masks worn incorrectly has been augmented to 5 times its size by rotating in range of 15 degrees, shifting width and height in range of %5, shifting brightness in range of %30 to %100, applying shear in range of %5, shifting color channels in range of %10, and finally randomly flipping the images horizontally. Then the loss used during the training is weighted by classes, weights being proportional to the number of images in each class. And finally, bias is used on the last layer of the network, which is initialized by the following formula:

![image](https://user-images.githubusercontent.com/65850584/221670672-24b1992b-1216-4fc6-8daf-81c15cd69408.png)

Where ni is the amount of data in the i’th class, fi is the frequency of a class in the whole dataset, and bi is the bias used for the i’th class.

![image](https://user-images.githubusercontent.com/65850584/221670900-6792e870-f5ba-446f-8aed-a21a4a4fd46c.png)
Fig. 2. Samples of three classes of cropped faces from the dataset

The input image size is the default 224*224, the size on which the network has been optimized. All the layers of the pre-trained network are frozen and at the bottom of the network 3 dense layers with respectively 16, 16 and 8 neurons are put in place and after each a dropout layer of %20 is put in order to prevent overfitting, since the dataset is small and imbalanced. Batch size of 256 is chosen because an imbalanced dataset requires a large batch size, though unfortunately the GPU limits this parameter. Adam optimizer is used with a constant learning rate of 0.001. PReLU is used as the activation function. Categorical cross entropy is used as the loss function. Finally, the network is trained in 100 epochs, using the GPU provided by Google Colab.

### III.	RESULTS
#### A.	Object detection
After a few rounds of trial and error to get the optimal results and finally, fully training all four networks, in this section we go through the obtained results, analyze them and compare them to see which methods are better choices for the task at hand.

Since the loss function used on the networks differ from each other, they cannot be compared directly, but a comparison of the loss changing behavior during the training, can give us an idea of the network’s learning ability. From Fig. 3 to Fig. 6, loss changes during the training are plotted.

![image](https://user-images.githubusercontent.com/65850584/221671676-5daf562e-8d51-4fbc-bed0-0c7f04c0f3b5.png)
Fig. 3. Changes of loss during the training of YOLO V4

![image](https://user-images.githubusercontent.com/65850584/222807998-176ba2a0-6257-48df-992d-a038bd52123b.png)
Fig. 4. Changes of loss during the training of SSD

![image](https://user-images.githubusercontent.com/65850584/222808161-98b9558a-f181-43a4-8ba1-ca0d9d71ec46.png)
Fig. 5. Changes of loss during the training of CenterNet

![image](https://user-images.githubusercontent.com/65850584/222808765-34d19df6-d80f-4995-bf8a-297da4baca8a.png)
Fig. 6. Changes of loss during the training of EfficientDet

The loss changes of YOLO V4 and EfficientDet as depicted on Fig. 3 and Fig. 6, are very similar and might lead one to think they show similar results, while in fact as it shall be discussed, it’s quite the opposite. While YOLO V4 is very efficient in finding the global minimum quickly, EfficientDet gets stuck in a local minimum and cannot continue to learn.

Loss of the SSD network as depicted on Fig. 4 is ascending at first up to around 5k steps and then starts to descend. This pattern is sometimes necessary for loss change in order to pass local minima. CenterNet on the other hand, has a rather more oscillatory loss changing behavior, as seen on Fig. 5, but keeps its descending behavior throughout the training process.

The trained networks are tested on three of the challenging images to compare the results. The chosen image numbers are 139, 156 and 822, the results of which are shown on Fig. 7 to Fig. 10. The threshold of acceptable predicted accuracies is %35.

These figures merely show an overall demonstration of the predictions on the images. By looking at the detailed results of these predictions and comparing them with the ground truth, more tangible results are obtained. By doing the necessary calculations, these results are acquired and shown on Table II. It is worth noting that these results are not to be taken for granted, since they only represent the results obtained from testing four random images. Their main purpose is to give a general understanding of the trained networks’ behaviors.

From Table II it is evident that the YOLO V4 method is the best choice among the tested networks. Among the methods from TensorFlow V2 however, CenterNet seems to be the best and EfficientDet the worst choice. Another interesting point to note from this table, is the precision of EfficientDet always being 1, which means that there are no false positives in any of the tested images and for any of the three classes. In other words, EfficientDet doesn’t predict a result unless it’s pretty sure.

![image](https://user-images.githubusercontent.com/65850584/221672202-d60193c5-d0ce-4bb2-be9d-16dd8d983dbd.png)
TABLE II. CLASSIFICATION METRICS FROM THE AVERAGE OF FOUR RANDOM IMAGES, CALCULATED FOR THE FOUR OBJECT DETECTION ALGORITHMS USED.

![image](https://user-images.githubusercontent.com/65850584/221672295-b37289b5-7a9c-444f-879f-9bb28eefc778.png)
Fig. 7. YOLO V4 results. Image numbers from the top: 822, 139, 156, and 672

![image](https://user-images.githubusercontent.com/65850584/221672390-55f099d5-c323-4899-ae53-ca35565f9054.png)
Fig. 8. SSD results. Image numbers from the top: 822, 139, 156, and 672

![image](https://user-images.githubusercontent.com/65850584/221672481-1e655108-1e4a-4984-ae8e-0baa567aae41.png)
Fig. 9. CenterNet results. Image numbers from the top: 822, 139, 156, and 672

![image](https://user-images.githubusercontent.com/65850584/221672510-a3d38eef-602d-445d-9210-55edaae11de3.png)
Fig. 10. EfficientDet results. Image numbers from the top: 822, 139, 156, and 672

#### B.	Classification of cropped boxes
Since the architecture of the fully connected layers at the bottom of the model for both VGG16 and MobileNet V2 is created equally in this experiment, and since the batch size, optimizer, number of epochs, and the loss function are also the same for both models, their loss and accuracy changing behavior can be compared directly. Graphs of the loss changes are shown on Fig. 11 and Fig. 12, and changes of accuracy are shown on Fig. 13 and Fig. 14.

![image](https://user-images.githubusercontent.com/65850584/222809171-b7fcd20b-8853-4214-b4ca-663b490a6a8b.png)
Fig. 11. Changes of train and validation loss during the training of MobileNet V2

![image](https://user-images.githubusercontent.com/65850584/222809252-e8328c8b-0381-4a7b-abef-d0e7c0cf78d0.png)
Fig. 12. Changes of train and validation loss during the training of VGG16

![image](https://user-images.githubusercontent.com/65850584/222809424-d3ba84cf-713e-43ba-bec2-661d24140ec8.png)
Fig. 13. Changes of train and validation accuracy during the training of MobileNet V2

![image](https://user-images.githubusercontent.com/65850584/222809500-c2e1651e-804f-44a5-97ab-177832f74a4d.png)
Fig. 14. Changes of train and validation accuracy during the training of VGG16

The reason for the difference between train and validation losses is mainly the more balanced data for training, since one class of the training data has been augmented, but the validation data is remained untouched. But it’s also a product of the low amount and quality of data which leads to overfitting, despite taking measures such as using a simple fully connected structure, using dropout layers, and setting a high batch size and a low learning rate.

As it can be seen on Fig. 13 and Fig. 14, even in the first few epochs, training and validation accuracies are high, and throughout the training process they frequently oscillate. The reason for these behaviors is the imbalanced dataset. On an imbalanced dataset, even predicting all images from one class can result in great accuracies, and despite correcting its behavior, it might suddenly aggravate the accuracy.

The classification metrics for both networks used in this experiment, are shown on Table III. And the confusion matrices for both training and validation are shown on Fig. 15 for MobileNet V2, and on Fig. 16 for VGG16.

![image](https://user-images.githubusercontent.com/65850584/221673393-4b812421-01cf-439a-8602-7f9235891456.png)
TABLE III. CLASSIFICATION METRICS FOR MOBILENET V2 AND VGG16.

From the F1 score obtained from the two models, it can be concluded that both networks show similarly good results. However, it is worth noting that the test F1 score obtained for the “mask worn incorrectly” class, reached its maximum at 0.59 only with the MobileNet V2 model, which means that this model architecture can better handle data imbalance.

![image](https://user-images.githubusercontent.com/65850584/222809722-2acbf922-b9d2-486d-a7a2-5747f5b2462b.png)
Fig. 15. Train and test data confusion matrix for MobileNet V2

![image](https://user-images.githubusercontent.com/65850584/222809785-813ebc42-5d29-4053-89ab-d076d4ac6dd6.png)
Fig. 16. Train and test data confusion matrix for VGG16

### IV.	CONCLUSION
In this paper four object detection methods were trained on a dataset of images containing three classes of objects, namely faces with masks, without mask, or wearing masks incorrectly. We have demonstrated that the YOLO V4 shows great superiority over the used networks both in learning speed and accuracy. Among other methods used, which are all obtained from TensorFlow V2 models, CenterNet with the backbone of MobileNet V2 has shown the best results.

In another experiment, the bounding boxes from the dataset have been used to crop the faces in the dataset and train a classification algorithm to classify the said three mask wearing classes. We have shown that both MobileNet V2 and VGG16 pre-trained CNN models show good results on this task, but MobileNet V2 can better handle data imbalance.

### REFERENCES
[1]	T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” May 2014, [Online]. Available: http://arxiv.org/abs/1405.0312.

[2]	A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, “YOLOv4: Optimal Speed and Accuracy of Object Detection,” Apr. 2020, [Online]. Available: http://arxiv.org/abs/2004.10934.

[3]	C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “Scaled-YOLOv4: Scaling Cross Stage Partial Network,” Nov. 2020, [Online]. Available: http://arxiv.org/abs/2011.08036.

[4]	G. Tata, S.-J. Royer, O. Poirion, and J. Lowe, “DeepPlastic: A Novel Approach to Detecting Epipelagic Bound Plastic Using Deep Visual Models,” May 2021, [Online]. Available: http://arxiv.org/abs/2105.01882.
