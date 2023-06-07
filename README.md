# intel-oneAPI

#### Team Name - Single Londe
#### Problem Statement -  Object Detection For Autonomous vehicles 
#### Team Leader Email - arnvsnigi@gmail.com

## A Brief of the Prototype:
**Prototype Name:** CitySegNet <br><br>
**Description:**
CitySegNet is a deep learning prototype designed for semantic image segmentation on the Cityscapes dataset. The Cityscapes dataset is a popular benchmark for urban scene understanding, consisting of high-resolution images captured from various cities. The goal of semantic image segmentation is to assign a class label to each pixel in an image, enabling fine-grained understanding of urban scenes.<br><br>
![WhatsApp Image 2023-06-07 at 4 55 43 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/dd56a60a-1d06-47f7-8654-49e61ce6719e) <br><br>
**Dataset:** The Cityscapes Dataset is a large dataset that focuses on semantic understanding of urban street scenes. The dataset contains 5000 images with fine annotations across 50 cities, different seasons, varying scene layout and background.The dataset is annotated with 30 categories, of which 19 categories are included for training and evaluation (othersare ignored). The training, validation, and test set contains 2975, 500, and 1525 fine images, respectively. An additional 20000 images with coarse (polygonal) annotations are also provided, but are only used for training. <br><br>
**Architecture:** We use DUC (Dense upsampling convulution ) which is a CNN based model for semantic segmentation which uses an image classification network (ResNet) as a backend and achieves improved accuracy in terms of mIOU score using two novel techniques. The first technique is called Dense Upsampling Convolution (DUC) which generates pixel-level prediction by capturing and decoding more detailed information that is generally missing in bilinear upsampling. Secondly, a framework called Hybrid Dilated Convolution (HDC) is proposed in the encoding phase which enlarges the receptive fields of the network to aggregate global information. It also alleviates the checkerboard receptive field problem ("gridding") caused by the standard dilated convolution operation.
<br><br>![WhatsApp Image 2023-06-07 at 5 05 01 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/3f70f6be-5f1b-4709-986d-efd909627cbc)
<br><br>
We used the DeepLab-V2 [3] ResNet-101 framework to train our baseline model. Specifically, the network has a downsampling rate of 8, and dilated convolution with rate of 2and 4 are applied to res4b and res5b blocks, respectively.An ASPP module with dilation rate of 6, 12, 18, and 24 is added on top of the network to extract multiscale context information. The prediction maps and training labels are downsampled by a factor of 8 compared to the size of original images, and bilinear upsampling is used to get the final prediction. Since the image size in the Cityscapes dataset is 1024 × 2048, which is too big to fit in the GPU memory,we partition each image into twelve 800×800 patches with partial overlapping, thus augmenting the training set to have 35700 images.This data augmentation strategy is to make sure all regions in an image can be visited. This is an improvement over random cropping, in which nearby regions may be visited repeatedly<br><br>![WhatsApp Image 2023-06-07 at 4 59 32 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/1c2ff2a8-ce04-439d-98c1-c1754d38f99e)<br><br>

We examine the effect of DUC on the baseline network. In DUC the only thing we change is the shape of the top convolutional layer. For example, if the dimension of the top convolutional layer is 68×68×19 in the baseline model (19 is the number of classes), then the dimension of the same layer for a network with DUC will be 68 × 68 × (r2 × 19) where r is the total downsampling rate of the network (r = 8 in this case). The prediction map is then reshaped to size 544×544×19. DUC will introduce extra parameters compared to the baseline model, but only at the top convolutional layer. We train the ResNet-DUC network the same way as the baseline model for 20 epochs, and achieve a mean IOU of 74.3% on the validation set, a 2% increase compared to the baseline model.<br><br>

**Result:** mean Intersection Over Union (mIOU) is the metric used for validation. For each class the intersection over union (IOU) of pixel labels between the output and the target segmentation maps is computed and then averaged over all classes to give us the mean intersection over union (mIOU).<br><br>
![WhatsApp Image 2023-06-07 at 5 33 09 PM](https://github.com/arnvsnigi/intel-oneAPI/assets/22290716/9ecbdb22-f8a2-4d5d-9240-d1f78d571583)

We used MXNet framework to compute mIOU of the models on the validation set described above. Use the notebook duc-validation to verify the mIOU of the model. The scripts cityscapes_loader.py, cityscapes_labels.py and utils.py are used in the notebook for data loading and processing.
## Tech Stack: 
   List Down all technologies used to Build the prototype **Clearly mentioning Intel® AI Analytics Toolkits, it's libraries and the SYCL/DCP++ Libraries used**
   
## Step-by-Step Code Execution Instructions:
## Inference
We utilized MXNet as our framework for carrying out the process of inference. To grasp the usage of the aforementioned models for inference, please refer to the notebook titled "duc-inference” [duc-inference](dependencies/duc-inference.ipynb). Below, you will find a concise explanation of the inference procedure.
### Input
Given that the model has been trained on the cityscapes dataset, which consists of urban street scene images, it is recommended to provide an input image of a street scene for optimal results. The size of the image does not impose any restrictions. In the inference notebook, an example is demonstrated using a png image.

### Preprocessing
The DUC (Dense Upsampling Convolution) layer divides the image into d^2 subparts, where d represents the downsampling rate. To ensure accurate reshaping of the image after passing through the DUC layer, a small border is added to the input image. This extrapolation helps maintain the integrity of the reshaped image. Following this step, the image undergoes normalization through mean subtraction. Check [duc-preprocess.py](dependencies/duc-preprocess.py) for code.

### Output
The output of the network is a tensor of shape (1 X `label_num` X `H` * `W`) where `H` and `W` are the height and width of the output segmented map.

### Postprocessing
The output tensor is reshaped and resized to give the softmax map of shape (`H` X `W` X `label_num`). The raw label map is computed by doing an argmax on the softmax map. The script [cityscapes_labels.py](dependencies/cityscapes_labels.py) contains the segmentation category labels and their corresponding color map. Using this the colorized segmented images are generated. Check [duc-postprocess.py](dependencies/duc-postprocess.py) for code.
  
## What I Learned:
   Write about the biggest learning you had while developing the prototype
