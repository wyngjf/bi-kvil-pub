# Demo Preprocessing and K-VIL

## Prepare data

Download the example dataset from [TODO] or see below how to record your own data.

```shell
source scripts/download.sh
```

Download checkpoints of DON and place it under `$DEFAULT_CHECKPOINT_PATH/dcn`. 
```
$DEFAULT_CHECKPOINT_PATH/dcn
- object1
- object2
- ...
- config.yaml
```

The raw data looks like the file structure below. And create a folder `canonical/dcn` and file `canonical_cfg.yaml` with the object that you would like to detect using
Dense Object Net and how many candidate points you would like to have:

```yaml
num_candidates:
    plate: 300
    spoon: 300

dcn_obj_cfg:
    plate: plate
    spoon: spoon
```

```
- /abs_path/of/demonstrations/
    - canonical
        - dcn
            - canonical_cfg.yaml    [ object name to DCN checkpoint name map ]
    - recordings
        - trail1
            - the raw video         [ (ZED: xxxx.svo) or (AzureKinect: xxxx.mkv) ]
            - images                [ (stereo: left_xxxxxx.png, right_xxxxxx.png) or (mono: xxxxxx.png) ]
            - depth                 [ (ZED: empty dir) or (AzureKinect: xxxxxx.png) ]
            - param.yaml            [ the camera intrinsic parameters ]
            - scene_graph.yaml      [ configuration of the objects involved in the task ]
            - seg.yaml              [ motion segmentation points ]
        - ...
```

## Create scene graph configuration

Scene graph generation is not the focus of this work. You need to provide what object should be detected. 
For example, below shows the `scene_graph.yaml` file in the `PS1` place spoon scenario, 
we use GroundedSAM to obtain the instance segmentation masks. 

For each object, specify a text prompt for GroundingDINO to generate a bounding box of this object, which is then used 
by SAM as prompt to generate binary mask. We are interested in plate, spoon, table and person in this example. You can 
specify mask erosion parameters as well. 

```shell
obj_cfg:
    plate: plate
    spoon: spoon
    table: table
    person: person
erode:
    plate: 2
    spoon: 2
    table: 2
    person: 2
```
Create such configuration for each of your demonstration video. 


## Prepare dataset for Bi-KVIL

run 

```shell
# For stereo
kvil_demo -p $DEFAULT_DATASET_PATH/PS1 -n kvil -t 30 -v -tv -k
# For Monocular
kvil_demo -p $DEFAULT_DATASET_PATH/PS1 -n kvil -t 30 -v -tv -k --mono
# for help
kvil_demo -h
```



<details>
<summary>Folder structure of a specific task after the perception pipeline</summary>

```
- /abs_path/of/demonstrations/
    - canonical
        - dcn
            - obj1
                - can_inlier.yaml
                - can_outlier.yaml
                - coordinates_3d_fixed.yaml
                - coordinates_3d.yaml
                - depth.png
                - descriptor.yaml
                - intrinsics.yaml
                - mask.png
                - overlay.jpg
                - rgb.png
                - uv_colors.yaml
                - uv.yaml
                - ...
            - obj2
            - ...
            - canonical_cfg.yaml
            - descripts.pth
        - ...
    - recordings
        - trail1
            - the raw video
            - depth
            - images
            - param.yaml
            - scene_graph.yaml
            - seg.yaml
        - ...
    - namespace
        - canonical
        - config
        - data
            - trial1
                - rgb       [ down-sampled rgb images (left view if stereo): xxxxxx.png ]
                - depth     [ corresponds to rgb, also visible depth images for human ]
                - mask      [ all, obj1, obj2, ... ]
                - dcn
                - human     [ some model may only have holistic body model ]
                    - graphormer
                    - rtmpose
                - obj
                - flow
                - results
            - ...
        - video
        - viz [ similar structure as 'data' ]
```
</details>
<br>


# Run Bi-KVIL

```shell
# first time or force re-compute
kvil -p $DEFAULT_DATASET_PATH/PS1 -n 6 -f
# reload to view pre-computed results
kvil -p $DEFAULT_DATASET_PATH/PS1 -n 6 -r
# for help
kvil -h
```

The Bi-KVIL GUI shows the extracted keypoints, frame and geometric constraints. 

# Use Your Own Data

To record your own video, you can use either a stereo camera (e.g. Stereolab ZED) 
or a monocular RGB-D camera (e.g. AzureKinect). see [_tutorial_record_demo.md](_tutorial_record_demo.md)

1. Record the demonstration videos, and put them in a folder following the file structures described above
2. Extract images
    - if you use ZED or AzureKinect, run
    ```shell
    # For ZED
    kvil_demo -p absolute/path/to/the/demonstration -n kvil
    # For AzureKinect
    kvil_demo -p absolute/path/to/the/demonstration -n kvil --mono
    # select the trials to work on and 'Enter'
    # select 'extract' and 'Enter'
    ```
    - otherwise, you need to extract images with custom code and provide the camera parameters.
3. Since the recordings may include motions before and after the actual task. You need to segment the clip of video that 
   that is relevant to the task, i.e. to annotate begin and end of the task. 
    Run the script below and follow the command line instruction
    ```shell
    kvil_annotate -p absolute/path/to/the/demonstration -n kvil
    # e.g. 
    kvil_annotate -p $DEFAULT_DATASET_PATH/PS1 -n kvil
    # for help
    kvil_annotate -h
    ```
    The limitation of the current approach:

    1. There is not a automatic segmentation model, you have to manually select segmentation point for task under a namespace.
    2. We do not explicitly handle occlusion. We use the object position before it is occluded. Therefore, you can use
       `-o` option to specify the object being occluded, and mark the first frame where the occlusion occurs.

4. Preprocessing. Run
    ```shell
    # For stereo
    kvil_demo -p $DEFAULT_DATASET_PATH/PS1 -n kvil -t 30 -v -tv -k
    # For Monocular
    kvil_demo -p $DEFAULT_DATASET_PATH/PS1 -n kvil -t 30 -v -tv -k --mono
    ```
