# intel-oneAPI

#### Team Name - Single Londe
#### Problem Statement -  Object Detection For Autonomous vehicles 
#### Team Leader Email - arnvsnigi@gmail.com

## A Brief of the Prototye:
### Prototype Name:
CitySegNet 
### Description:
CitySegNet is a deep learning prototype designed for semantic image segmentation on the Cityscapes dataset. The Cityscapes dataset is a popular benchmark for urban scene understanding, consisting of high-resolution images captured from various cities. The goal of semantic image segmentation is to assign a class label to each pixel in an image, enabling fine-grained understanding of urban scenes.<br><br>
![WhatsApp Image 2023-06-07 at 4 55 43 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/dd56a60a-1d06-47f7-8654-49e61ce6719e) 
### Dataset:
The Cityscapes Dataset is a large dataset that focuses on semantic understanding of urban street scenes. The dataset contains 5000 images with fine annotations across 50 cities, different seasons, varying scene layout and background.The dataset is annotated with 30 categories, of which 19 categories are included for training and evaluation (othersare ignored). The training, validation, and test set contains 2975, 500, and 1525 fine images, respectively. An additional 20000 images with coarse (polygonal) annotations are also provided, but are only used for training. 
### Architecture:
We use DUC (Dense upsampling convulution ) which is a CNN based model for semantic segmentation which uses an image classification network (ResNet) as a backend and achieves improved accuracy in terms of mIOU score using two novel techniques. The first technique is called Dense Upsampling Convolution (DUC) which generates pixel-level prediction by capturing and decoding more detailed information that is generally missing in bilinear upsampling. Secondly, a framework called Hybrid Dilated Convolution (HDC) is proposed in the encoding phase which enlarges the receptive fields of the network to aggregate global information. It also alleviates the checkerboard receptive field problem ("gridding") caused by the standard dilated convolution operation.
<br><br>![WhatsApp Image 2023-06-07 at 5 05 01 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/3f70f6be-5f1b-4709-986d-efd909627cbc)
<br><br>
We used the DeepLab-V2 [3] ResNet-101 framework to train our baseline model. Specifically, the network has a downsampling rate of 8, and dilated convolution with rate of 2and 4 are applied to res4b and res5b blocks, respectively.An ASPP module with dilation rate of 6, 12, 18, and 24 is added on top of the network to extract multiscale context information. The prediction maps and training labels are downsampled by a factor of 8 compared to the size of original images, and bilinear upsampling is used to get the final prediction. Since the image size in the Cityscapes dataset is 1024 × 2048, which is too big to fit in the GPU memory,we partition each image into twelve 800×800 patches with partial overlapping, thus augmenting the training set to have 35700 images.This data augmentation strategy is to make sure all regions in an image can be visited. This is an improvement over random cropping, in which nearby regions may be visited repeatedly<br><br>![WhatsApp Image 2023-06-07 at 4 59 32 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/1c2ff2a8-ce04-439d-98c1-c1754d38f99e)<br><br>

We examine the effect of DUC on the baseline network. In DUC the only thing we change is the shape of the top convolutional layer. For example, if the dimension of the top convolutional layer is 68×68×19 in the baseline model (19 is the number of classes), then the dimension of the same layer for a network with DUC will be 68 × 68 × (r2 × 19) where r is the total downsampling rate of the network (r = 8 in this case). The prediction map is then reshaped to size 544×544×19. DUC will introduce extra parameters compared to the baseline model, but only at the top convolutional layer. We train the ResNet-DUC network the same way as the baseline model for 20 epochs, and achieve a mean IOU of 74.3% on the validation set, a 2% increase compared to the baseline model.

### Result:
mean Intersection Over Union (mIOU) is the metric used for validation. For each class the intersection over union (IOU) of pixel labels between the output and the target segmentation maps is computed and then averaged over all classes to give us the mean intersection over union (mIOU).<br><br>
![WhatsApp Image 2023-06-07 at 5 33 09 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/9ecbdb22-f8a2-4d5d-9240-d1f78d571583)

We used MXNet framework to compute mIOU of the models on the validation set described above. Use the notebook duc-validation to verify the mIOU of the model. The scripts cityscapes_loader.py, cityscapes_labels.py and utils.py are used in the notebook for data loading and processing.
## Tech Stack: 
### Intel® Neural Compressor
An open-source Python library supporting popular model compression techniques on all mainstream deep learning frameworks (TensorFlow, PyTorch, ONNX Runtime, and MXNet).
### Intel® Extension for PyTorch\*
Intel® Extension for PyTorch\* extends PyTorch\* with up-to-date features optimizations for an extra performance boost on Intel hardware. 
   
## Step-by-Step Code Execution Instructions:
## Inference
We utilized MXNet as our framework for carrying out the process of inference. To grasp the usage of the aforementioned models for inference, please refer to the notebook titled "duc-inference” [duc-inference](dependencies/inference.ipynb). Below, you will find a concise explanation of the inference procedure.
## Model
The model ResNet101_DUC_HDC uses ResNet101 as a backend network with both Dense Upsampling Convolution (DUC) and Hybrid Dilated Convolution (HDC) techniques.

|Model        |Download  | ONNX version |[mIOU](#metric) (%)|
|-------------|:--------------|:--------------|:--------------|
|ResNet101_DUC_HDC|    [248.6 MB](model/ResNet101-DUC-7.onnx)  |1.2.2  |81.92 |
|ResNet101_DUC_HDC-12-int8|   [62.5 MB](model/ResNet101-DUC-12-int8.onnx)  |1.9.0  |81.62 |

> Compared with the ResNet101_DUC_HDC-12, ResNet101_DUC_HDC-12-int8's mIOU decline ratio is 0.37% and performance improvement is 1.69x.
>
> Note the performance depends on the test hardware. 
> 
> Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.
### Input
Given that the model has been trained on the cityscapes dataset, which consists of urban street scene images, it is recommended to provide an input image of a street scene for optimal results. The size of the image does not impose any restrictions. In the inference notebook, an example is demonstrated using a png image.<br><br>
![Screenshot 2023-06-08 at 1 21 52 AM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/33a10dc1-4f18-4988-a8a3-ef455438078b)

### Preprocessing
The DUC (Dense Upsampling Convolution) layer divides the image into d^2 subparts, where d represents the downsampling rate. To ensure accurate reshaping of the image after passing through the DUC layer, a small border is added to the input image. This extrapolation helps maintain the integrity of the reshaped image. Following this step, the image undergoes normalization through mean subtraction. Check [duc-preprocess.py](dependencies/preprocess.py) for code.

### Output
The output of the network is a tensor of shape (1 X `label_num` X `H` * `W`) where `H` and `W` are the height and width of the output segmented map.
* **Run Prediction**:
```bash
python duc-inference.py city_example.jpg
```
### Postprocessing
The output tensor is reshaped and resized to give the softmax map of shape (`H` X `W` X `label_num`). The raw label map is computed by doing an argmax on the softmax map. The script [cityscapes_labels.py](dependencies/cityscapes_labels.py) contains the segmentation category labels and their corresponding color map. Using this the colorized segmented images are generated. Check [duc-postprocess.py](dependencies/postprocess.py) for code.<br><br>
![Screenshot 2023-06-08 at 1 20 15 AM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/638642ab-df04-44eb-8825-537a0ba5dbb7)
![Screenshot 2023-06-08 at 1 20 24 AM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/e4bd8fb9-a2bf-4225-9abb-9bf407e923ad)

### Confidence obtained
![Screenshot 2023-06-08 at 1 20 30 AM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/4be86aac-1552-47de-8cbf-0b24a60cf5ba)<br><br>
Check duc-inference.py for more.


## What I Learned:
  By developing the CitySegNet prototype for image segmentation on the Cityscapes dataset, We have gained several valuable learnings and experiences, including:

1. Understanding of Semantic Image Segmentation: We have gained a deeper understanding of the field of semantic image segmentation, its challenges, and its applications. This includes knowledge of popular network architectures, training methodologies, and evaluation metrics specific to image segmentation tasks.

2. Dataset Handling: Working with the Cityscapes dataset has have familiarized us with the process of handling large-scale datasets for computer vision tasks. We have learned how to preprocess and augment the dataset, extract relevant features, and prepare the data for training and evaluation.

3. Deep Learning Frameworks: Developing CitySegNet would have required proficiency in deep learning frameworks such as TensorFlow or PyTorch. We have gained experience in utilizing these frameworks to build, train, and evaluate deep learning models for image segmentation tasks.

4. Model Architecture and Hyperparameter Tuning: Experimenting with the DUC architecture and modifying it for improved performance has taught us about the importance of model architecture and hyperparameter selection. We have learned to fine-tune various aspects of the model, such as the number of layers, filter sizes, learning rates, and activation functions, to achieve better results.

5. Performance Evaluation: Evaluating the performance of CitySegNet using metrics like mIOU, pixel accuracy, and class-wise accuracy has given us insights into the strengths and weaknesses of the model. we have have learned how to interpret evaluation results and identify areas for improvement.

6. Practical Applications: Developing CitySegNet has exposed us to real-world applications of image segmentation, particularly in the context of urban scene understanding. We have gained an understanding of how accurate segmentation can contribute to fields such as autonomous driving, urban planning, and augmented reality.

7. Challenges and Future Directions: While developing the prototype, we  have encountered challenges such as overfitting, limited training data, or the need for more advanced techniques to handle complex urban scenes. This experience has sparked ideas for future research and improvements in image segmentation algorithms.

Overall, developing the CitySegNet prototype would have provided you with a hands-on experience in tackling the intricacies of image segmentation, deep learning model development, and the practical implications of this technology in urban environments.

## References:
* All models are from the paper [Understanding Convolution for Semantic Segmentation](https://arxiv.org/abs/1702.08502).
* [TuSimple-DUC repo](https://github.com/TuSimple/TuSimple-DUC), [MXNet](http://mxnet.incubator.apache.org)
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
* DUC github object-detection-segmentation by jcwchen
