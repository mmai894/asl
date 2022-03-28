# MMAI 894 Deep Learning Final Project - Team Bathurst

## Table of Contents
- [Scope](#scope)
- [Business Problem](#business-problem)
- [Proposed Solution](#proposed-solution)
- [Dataset](#dataset)
- [Modeling](#modeling)
  - [Models Explored](#models-explored)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
- [Results](#results)
- [Limitations](#limitations)
- [Future Directions](#future-directions)
- [References](#references)

## Scope
The purpose of this project is to use deep learning to build image classification models that can recognize American Sign Language (ASL) gestures. Using these models, we seek to provide a means for people to learn ASL to enhance communication with the hearing- and/or speech-impaired. We hope that our solution will eventually become full-fledged virtual interpreter that can be used in daily life.

## Business Problem
ASL is used by an estimate 500,000 people in North America and is the most common sign language used in the region[[1]](https://www.startasl.com/history-of-sign-language/). A shortage of qualified ASL interpreters, the rapid development and adoption of technology, and virtual environments in society present new barriers to communication for ASL users. Therefore, there is a need to improve the communication of ASL users in society. This need presents a business opportunity to provide an assistive technology solution to teach ASL to individuals in hopes of facilitating communication.

## Proposed Solution
Our group proposes a deep learning solution that provides a means for people to learn ASL basics by learning how to sign the letters of the English alphabet in ASL. Deep learning techniques has been shown to be more performant than previous state-of-the-art machine learning methods for computer vision tasks[[2]](https://www.sciencedirect.com/science/article/abs/pii/S095741742030614X). Because image classification involves computer vision, Convolutional Neural Networks (CNN) form the basis of our solution. On an ASL learning platform, our solution would be able to recognize the ASL signs gestured by users and provide feedback on whether they have signed correctly or not. We believe that our solution will improve quality of life and general accessibility in a society that is becoming more virtual.

## Dataset
The ASL image dataset used to train our deep learning models is found [here](https://www.kaggle.com/kapillondhe/american-sign-language).

The dataset contained colour images of ASL gestures signing the English alphabet. Each image in the dataset was 400x400 pixels and in JPG format. There was a class of images for each letter, space character, and background-only control images for 28 classes. The dataset was relatively balanced across all classes; it was originally separated into training and test sets, with each class having 4,546-5,996 images in the training set and 4 images in the test set.

![sample images from ASL images dataset](README%20Figures/rawdata.jpg)\
**Figure 1.** Sample images from the ASL alphabet image dataset. Left to right: letter A, the space character, background-only image

The images in the dataset were clear and did not have any artifacts that needed to be removed. The images were not blurry, distorted, or poorly cropped to the extent the sign was not in view. The background of the images was minimal and had good contrast with the signing hand. The signing hand is centred and not at an angle, making it clear in all images.

## Modeling

### Models Explored
- Custom Convolution Neural Network (CNN)
- Residual Neural Network (ResNet)
- VGG16

### Data Preprocessing
Preprocessing steps that were taken include downscaling images to a smaller size to accelerate model training for custom CNN and ResNet solutions and align with the training image size of the pre-trained VGG16 model. Images were then normalized by dividing the original pixel values by 255 and scaling them to a 0-1 range to decrease computation time.

Image augmentation was introduced to help address the bias in the dataset and mitigate the risk of model overfitting. Example image augmentation techniques include brightness change, rotation, flip, scale, and skew. Our group used augmented image datasets to train the deep learning models.

![image augmentation comparison](README%20Figures/augmentation.jpg)\
**Figure 2.** Unaugmented image signing the letter I (left) and augmented (flip and rotation applied) normalized image signing the letter I (right)

### Model Training
Because the dataset was not imbalanced, we chose accuracy as the model evaluation metric. In this multiclass classification problem, the main criterion for success is maximizing the accuracy across all classes.

After repeated attempts to train the model, the maximum batch size we could reach was 64 due to memory limitations. The ADAM optimizer with no pre-configuration and a loss function of categorical cross-entropy were chosen for the baseline model.

![baseline CNN model architecture](README%20Figures/baselineCNN.jpg)\
**Figure 3.** Baseline CNN model architecture

To increase accuracy with our custom CNN model, a Sobel filter preprocessing function was added. The Sobel filter was used as a preprocessing function to highlight the edges.

![images after applying Sobel filter](README%20Figures/sobel.jpg)\
**Figure 4.** Dataset image of the letter V (left) and letter H (right) after applying the Sobel filter

![ResNet model architecture](README%20Figures/resnetarchitecture.jpg)\
**Figure 5.** ResNet model architecture

We attempted to transfer learning by utilizing the pre-trained VGG16 model and added two dense layers with 1024 neurons each.

## Results
**Table 1.** Results of the best performing model of each architecture

| Metric | VGG16 | CNN | ResNet |
| ----------- | ----------- | ----------- | ----------- |
| Validation Set Accuracy | 0.9867 | 0.9822 | 0.9872 |
| Validation Set Loss | 0.1256 | 0.5647 | 0.1726 |
| Training Set Accuracy | 0.998 | 0.9947 | 0.9941 |
| Training Set Loss | 0.0622 | 0.141 | 0.0442 |
| Test Set Accuracy | 1.0 | 1.0 | 1.0 |

## Model Deployment Architecture

Automating development, deployment, and maintenance of Machine Learning (ML) models with best practices is crucial for Machine Learning Operations (MLOps). Our method will be based on the Cross-Industry Standard Process for Data Mining (CRISP-DM), which provides a framework for conceptualizing the continuous data mining lifecycle. A key component of this process is establishing measurable business value, as well as planning for how the solution will be maintained. With that goal in mind, our team has put together a deployment architecture inspired by CRISP-DM that meets the business acceptance criteria in production. 

![Model Deployment Architecture](README%20Figures/modeldeploymentarchitecture.png)\
**Figure 6.** Model Deployment Architecture

In keeping with the MLOps philosophy of continuous improvement, we will ensure the following criteria are in place as model health monitoring guidelines: 

* Observe any changes that could affect the prediction of the model in unexpected ways. 
* Fine-tune and retrain the model periodically to prevent staleness and evaluate the data quantitatively and qualitatively. 
* Educate the new team members if a model ownership change occurs. 

## Limitations
A limitation of the dataset is the similar skin tone of the signing hand and homogeneous background that were present in all images. As such, the risk of model overfitting is likely high. To remedy this issue, training with a set of images that are diverse in time of day, signing hand skin tones, lighting conditions, and locations is suggested. A diverse training set should yield a more generalizable solution for use in real-world settings.

A limitation of our results is the small size of the test dataset. We believe there are insufficient sample test sets to properly evaluate and compare the different models as they all achieved perfect accuracy. It may have been better to remove a portion of the training dataset and allocate it to the test set. This could have also helped with training times and computational requirements. A smaller training dataset could have allowed further experimentation, cutting down the model training times that exceeded 15 hours for all models.

## Future Directions
Our models accurately classify the letter based on a single picture of the corresponding ASL gesture. While it is yet unable to transcribe sentences and paragraphs in real-time, it does bring us a step closer to a model that could have such a capability. Our solution can build upon letter recognition and incorporate full words into the detection algorithm for future developments. The solution could eventually be developed into an educational program for learning the basics of ASL and be the first stepping stone in becoming fluent in ASL. Businesses could use the program to teach client-facing customers basic sentences to provide exceptional customer service for those who rely on ASL as their primary form of communication. Moreover, the application can be integrated into smartphones to include ASL as a method of interacting with voice-activated assistants such as Apple’s Siri. Lastly, the application could be further developed to provide live closed captioning for individuals using sign language in virtual meetings. This would allow the individual to be much more involved in the meeting rather than simply typing their message.

## References
1. Jay, M. (2021, February 15). American Sign Language. Start ASL. https://www.startasl.com/history-of-sign-language/
2. Rastgoo, R., Kiani, K., & Escalera, S. (2020). Sign Language Recognition: A Deep Survey. Expert Systems with Applications, 164, 113794. doi:10.1016/j.eswa.2020.113794
