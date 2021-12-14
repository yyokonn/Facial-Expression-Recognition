# CS229 Final Project: Facial Expression Recognition on Masked Faces
**Contributor**:
- Yoko Nagafuchi (yokongf@stanford.edu)
- Zifei Xu (zifei98@stanford.edu)
- Sally (Hanqing) Yao (yaohanqi@stanford.edu)

**Summary**:
Through our project, we tackle the task of facial expression recognition (FER) for images of human faces that are partially covered with a mask. In real life, facial expressions play a key role in communicating with others because they reveal and convey people's emotions and reactions. However, wearing a mask, which has become a norm as we face a global pandemic, has prohibited us from seeing the whole facial expression, such that it has become difficult for us to communicate smoothly. Therefore, in our application-based project, we approach this problem with a FER program that utilizes computer vision techniques to predict expressions, given facial images partially covered with a mask. 

The input to our algorithm is a set of color images of human faces, which we pre-process by resizing and applying mask images onto the faces, as well as with other data augmentation techniques. Then, we use 6 types of models to output a predicted emotion: Logistic Regression, Naive Bayes, Random Forest, Support Vector Machine (SVM), K-Means, and Convolutional Neural Network (CNN)-with-ResNet models to classify the images into 7 classes, labeling each face with one of the 7 emotions.

**Dataset**:
- Karolinska Directed Emotional Faces (KDEF):(https://www.kdef.se)
- MaskTheFace module by Aqeel Anwar for applying masks onto facial images: (https://github.com/aqeelanwar/MaskTheFace)

**Programs**:
- Emotion_Recognition.py: Training on one of the models-Naive Bayes, Logistic Regression, Random Forest, K-Means, SVM
- CNN.py: Training on customized CNN model
- ResNet.py: Training on modified pretrained ResNet18 model


