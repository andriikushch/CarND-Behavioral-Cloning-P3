# Use Deep Learning to Clone Driving Behavior

The goal of this project is to create a neural network that can learn how to drive a car by watching how you drive.

![https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/output.gif?raw=true](https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/output.gif?raw=true)

Here is a list of my steps to achieve this:

- I will use the simulator to create data to train the network.
- Preprocess data by cropping and normalizing, that it will contain relevant data and proper shape.
- Design and teach the network.
- Test, visualize and write a report.


## Required files

The submission includes:

- **model.py** 
- **drive.py**
- **model.h5** stored by parts in folder `model`:	
    - please use `cat model/model.h* > model.h5` to get the original `model.h5` file (because github for free does not support files larger then 100 mb)
- writeup report (**README.md**)
- **center.jpg, left.jpg, right.jpg**
	- samples from the training set
- **video.mp4**
    - video result
- **output.gif**
    

## The Code

This implementation was able to successfully train network which can clone the behavior of driving.

Data preprocessing and model are in the `model.py`. For memory efficiency, there is a generator used.

## Model Architecture and Training Strategy

The selected architecture was inspired by [https://devblogs.nvidia.com/deep-learning-self-driving-cars/](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

![https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

Because of computation limitation in a given environment, I had to simplify it a little bit. To prevent overfitting and improve learning I've added `BatchNormalization` and `Dropout`:

![https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/model.png?raw=true](https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/model.png?raw=true)

Code in `model.py` :

```python
# add Convolution2D layers
model.add(Convolution2D(filters=24, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Convolution2D(filters=36, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Convolution2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu'))

# add fully connected layers
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))
```

In order to prepare training and validation sets I used :

```python
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

And shuffle samples in generator code (see `model.py -> generator()` for details). 


### Examples of training images

| Left  | Center  | Right  |
|---|---|---|
| ![https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/left.jpg?raw=true](https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/left.jpg?raw=true)  | ![https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/center.jpg?raw=true](https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/center.jpg?raw=true)  | ![https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/right.jpg?raw=true](https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/right.jpg?raw=true) |

For every image, I also generated an augmented image by flipping the original one (see the generator in `model.py`).


# Result

Here is original resulting video file [https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/video.mp4](https://github.com/andriikushch/CarND-Behavioral-Cloning-P3/blob/master/video.mp4)

As we see from the video the car can drive a lap without any significant issue.