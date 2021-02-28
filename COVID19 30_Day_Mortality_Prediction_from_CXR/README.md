# HW3 COVID19 30-day Mortality Prediction from CXR
This is the third assignment for CS4602. <br>
By using the chest X-ray of 1393 patients and their medical records, the mortality of these patients after 30 days are predicted. <br>

## Dataset Information
### Training Data
- Number of patient: 1393 in total 
  - 1229 in class 0 and 164 in class 1
- Data per patient:
  - An 320x320 chest X-ray image
  - Medical records with 47 attributes stored in a csv

### Testing Data
- 457 patients with X-ray images and medical records


## Model
The entire model pipeline contains two steps. <br>
First, the image files are trained with DenseNet 201, which is provided in the keras API (https://keras.io/api/applications/), in a transfer learning approach. The predictions are then stored in a csv file - 107062274.csv. <br>
Then, the csv file with the medical records and the image prediction csv file are joined together to a csv file. <br>
Finally, a SVM model, which is provided from sklearn, would be trained based on the above-mentioned csv file, and the new prediction result would be store to a new csv file - Bonus_107062274.csv.  <br>

### Data Preprocess
Different methods are used to do the preprocess for the two data types. <br>
#### The medical record CSV file
- Missing information: Filled the most frequent values for the missing categorical features and median for missing numerical. <br>
- One hot transformation: Transfer text-like attributes that are not trainable for the model into one hot columns. <br>
```data_dum = pd.get_dummies(df, prefix=['s', 'd'], columns=['sex', 'ed_diagnosis'])```

#### Image files
- Preprocess for model: Different models needed different preprocess input command for loading the input images. In this study, DenseNet 201 is chosen.
```x = keras.applications.densenet.preprocess_input(inputs)```
- Data augmetation: Use (-0.01, 0.01) x 360 degree to randomly rotate data for augmentation. <br>
```x = keras.layers.experimental.preprocessing.RandomRotation((-0.01, 0.01))(x)```

  
### Selected Model Design
The final results are predicted based on the following model design. <br>
#### 107062274_HW3_Model
- Input: 320x320 chest X-ray images
- Output: Mortality prediction with [0, 1] values
- Method: Transfer learning
  - Base model: DenseNet 201 with the last 20 layers set as trainable.
  - Additional layers: 3 Dense layers with activation='relu' followed by different drop out values
``` 
inputs = keras.Input(shape=IMG_SHAPE)
x = keras.applications.densenet.preprocess_input(inputs)
x = keras.layers.experimental.preprocessing.RandomRotation((-0.01, 0.01))(x)
x = base_model(x, training=False)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x) 
model = keras.Model(inputs, outputs)
```
- Below are the training and validation performances using different training and validation dataset combinations.
<p align="center">
  <img width="1000" height="300" src=./figures/loss(1).png?raw=true>
</p>  
<!--- ![loss(1)](./figures/loss(1).png?raw=true) --->
<p align="center">
  <img width="1000" height="300" src=./figures/loss(2).png?raw=true>
</p>  
<!--- ![loss(2)](./figures/loss(2).png?raw=true) --->

<p align="center">
  <img width="1000" height="300" src=./figures/loss(3).png?raw=true>
</p>  
<!--- ![loss(3)](./figures/loss(3).png?raw=true) --->

#### Bonus_107062274_HW3_Model
- Classification model: SVM was choosed after comparing the performances of other classification models, including RandomForest, Naive Bayes, Decisoin Tree, and AdaBoost. <br>
```
- StandardScaler: In the model pipeline, StandardScaler is applied to normalized the input to range in [-1, 1]. <br>
```
```
- SVM kernal: After trying linear, poly, sigmoid, and rbf, rbf kernels are used because it output the most stable f1 scores. <br>
- C and gamma tunning: Grid search algorithm is used here as a kind of  greedy search to optimize the value of C and gamma. <br>

```

## Approach Comparison
### Dimension Reduction
I tried using different dimension reduction methods to reduce the dimension of the image data and train them on an SVM model. <br> 
The f1 socres are below 0.25 on average. <br>
Therefore, I decided to turn to another method - transfer learning.
### Transfer Learning
There are many base models selectable in Keras Applications (as showned as below). <br>
I tried various of combination. The Densenet models turned out to have the most stable and well-performed results. Therefore, Densenet201 is used in this study. <br>

## Experiment and Results
I tried tuning the parameters in this model in various ways. Below are some experiments that are done along with the results.
### Three additional Dense layers (activation = 'relu')
```
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
```
#### Different Learning Rate (last 15 DenseNet201 layers trainable)
- Basic setting: class weighted 1:9 and data augmentation with RandomRotation((-0.01, 0.01))
- Learning Rate: 0.0001 
<p align="center">
  <img width="1000" height="300" src=./figures/loss.png?raw=true>
</p> 
<!--- ![loss](./figures/loss.png?raw=true) --->

- Learning Rate: 0.00001 
<p align="center">
  <img width="1000" height="300" src=./figures/loss1.png?raw=true>
</p>  
<!--- ![loss1](./figures/loss1.png?raw=true) --->

- Learning Rate: 0.000001 
<p align="center">
  <img width="1000" height="300" src=./figures/loss2.png?raw=true>
</p>
<!--- ![loss2](./figures/loss2.png?raw=true) --->

#### Different Learning Rate Comparison (last 30 DenseNet201 layers trainable)
- Basic setting: class weighted 1:9 and data augmentation with RandomRotation((-0.01, 0.01))
- Learning Rate: 0.0001 
<p align="center">
  <img width="1000" height="300" src=./figures/loss5.png?raw=true>
</p>
<!--- ![loss5](./figures/loss5.png?raw=true) --->

- Learning Rate: 0.00001 
<p align="center">
  <img width="1000" height="300" src=./figures/loss4.png?raw=true>
</p>
<!--- ![loss4](./figures/loss4.png?raw=true) --->

- Learning Rate: 0.000001 | fine_tune_at:-30 | class_weight:class_0:0.1/ class_1:0.9
- Three Dense layers with activation='relu'
- RandomRotation((-0.01, 0.01))
<p align="center">
  <img width="1000" height="300" src=./figures/loss11.png?raw=true>
</p>
<!--- ![loss11](./figures/loss11.png?raw=true) --->

### Two additional Dense layers (activation = 'relu')
```
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
```
#### Different Learning Rate Comparison (last 30 DenseNet201 layers trainable)
- Basic setting: class weighted 1:9 and data augmentation with RandomRotation((-0.01, 0.01))
- Learning Rate: 0.0001
<p align="center">
  <img width="1000" height="300" src=./figures/loss10.png?raw=true>
</p>
<!--- ![loss10](./figures/loss10.png?raw=true) --->

- Learning Rate: 0.00001
<p align="center">
  <img width="1000" height="300" src=./figures/loss9.png?raw=true>
</p>
<!--- ![loss9](./figures/loss9.png?raw=true) --->

- Learning Rate: 0.000001 
<p align="center">
  <img width="1000" height="300" src=./figures/loss7.png?raw=true>
</p>
<!--- ![loss7](./figures/loss7.png?raw=true) --->

<p align="center">
  <img width="1000" height="300" src=./figures/loss8.png?raw=true>
</p>
<!--- ![loss8](./figures/loss8.png?raw=true) --->

#### Different Dropout Comparison (last 30 DenseNet201 layers trainable)
- Basic setting: Learning Rate: 0.000001/ Class weighted 1:9/ Data augmentation with RandomRotation((-0.01, 0.01))/ Two dense layers
- Dropout:0.5/ 0.3
<p align="center">
  <img width="1000" height="300" src=./figures/loss6.png?raw=true>
</p>
<!--- ![loss6](./figures/loss6.png?raw=true) --->

<p align="center">
  <img width="1000" height="300" src=./figures/loss15.png?raw=true>
</p>
<!--- ![loss15](./figures/loss15.png?raw=true) --->
- Dropout:0.5/ 0.5
<p align="center">
  <img width="1000" height="300" src=./figures/loss14.png?raw=true>
</p>
<!--- ![loss14](./figures/loss14.png?raw=true) --->
<p align="center">
  <img width="1000" height="300" src=./figures/loss16.png?raw=true>
</p>
<!--- ![loss16](./figures/loss16.png?raw=true) --->
- Dropout:0.5/ 0.6
<p align="center">
  <img width="1000" height="300" src=./figures/loss13.png?raw=true>
</p>
<!--- ![loss13](./figures/loss13.png?raw=true) --->

- Dropout:0.5/ 0.8
<p align="center">
  <img width="1000" height="300" src=./figures/loss12.png?raw=true>
</p>
<!--- ![loss12](./figures/loss12.png?raw=true) --->

## Reference Link:
- https://www.tensorflow.org/tutorials/images/transfer_learning
- 
-
- 
-
