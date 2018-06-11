# **Behavioral Cloning** 

This is the [third project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) in the [Udacity Self-Driving car nanodegree](https://udacity.com/course/self-driving-car-engineer-nanodegree--nd013).


Included:
- [model.py](model.py) The model architecture and utilities to train the model
- [model.h5](model.h5) The trained model
- [drive.py](drive.py) The (slightly modified) file to serve the model to the [simulator](https://github.com/udacity/self-driving-car-sim)
- [resources/track*](resources) The  datasets used for training
- [video.mp4](video.mp4) The video result of track 1
- [writeup_report.md](writeup_report.md) The report per the rubric 


Usage:

```sh
usage: model.py [-h] [--sample_data_dir SAMPLE_DATA_DIR [SAMPLE_DATA_DIR ...]]
                [--model MODEL] [--model_out MODEL_OUT] [--epochs EPOCHS]
                [--steering_angle_correction STEERING_ANGLE_CORRECTION]
                [--input_image INPUT_IMAGE [INPUT_IMAGE ...]]
                [--conv_layers CONV_LAYERS]
                {train,fine_tune,test,plot,visualize}
```

**Examples:**

Train model:

```#> python model.py train -d resources/track* -o model.h5 --epochs 5 --steering_angle_correction 0.25```

Test model:

```#> python model.py test -d resources/track* -m model.h5```

Fine-tune model (fixing all the convolutional layers):

```#> python model.py plot -m model.h5 -d resources/fine_tune_data```


Plot model architecture:

```#> python model.py plot -m model.h5```


Visualize first `n` convolutional layers:

```#> python model.py visualize -m model.h5 -i examples/center_2016_12_01_13_37_02_212.jpg -conv_layers 1```



####Some videos:


 ##### Track 1 on sample data
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/2tk5wp1-zDc/0.jpg)](http://www.youtube.com/watch?v=2tk5wp1-zDc)

##### Track 1 on final model
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/SPC1h_Lp72Q/0.jpg)](http://www.youtube.com/watch?v=)

##### Track 2 on final model
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/yl-srZHqV78/0.jpg)](http://www.youtube.com/watch?v=yl-srZHqV78)