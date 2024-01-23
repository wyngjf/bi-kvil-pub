# Recording

In practice, you may record demonstrations using your preferred camera, and adapt 
the perception pipeline to fit the specific data format and camera parameters. 
**Note that, K-VIL uses structured demonstrations of the task, and favors data with enough shape or pose variations. 
It may degenerate if quality of the demonstration is limited. Try to be a good teacher.**

We describe examples using AzureKinect and Stereolab ZED. 

## AzureKinect

Setup AzureKinect using the provided script in `robot-vision` package. See tutorials in `robot-vision` package.

Record with:

```shell
cd path/to/robot-vision
python robot_vision/dataset/azure_kinect/data.py -r
```
GUI pops up, press `Space` to start recording and `Esc` to finish.

The data will be saved to `robot_vision/data/OBJECT` folder.

Copy all demonstrations of a task to a folder with meaningful name.


## Stereolab ZED

Setup Stereolab ZED SDK following official instructions.

Record with:
```shell
cd path/to/robot-vision
python robot_vision/dataset/stereo/zed/recorder.py
```
Recording starts when the GUI pops up. Press `Esc` to stop.

Recordings are saved to `~/dataset/raw`. 

Copy all demonstrations of a task to a folder with meaningful name.