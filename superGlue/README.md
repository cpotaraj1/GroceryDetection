To run the demo_image.py
* download the test image zip file and rename the extracted folder to "images"
* Make sure you have a target image 
* make sure the weights are present in models/weigths folder

## Contents
There are two main top-level scripts in this repo:

1. `demo_image.py` : for a given target image, it finds the possible match in the input folder
2. `demo_superglue.py`: runs a live demo on a webcam, IP camera, image directory or movie file

### Run the demo on a live webcam

Run the demo on the target image against the image folder
```sh
python ./demo_image.py --input images/ --target_img target.jpg
```

### Run the demo on a live webcam

Run the demo on the default USB webcam (ID #0), running on a CUDA GPU if one is found:

```sh
python ./demo_superglue.py
```
Keyboard control:

* `n`: select the current frame as the anchor
* `e`/`r`: increase/decrease the keypoint confidence threshold
* `d`/`f`: increase/decrease the match filtering threshold
* `k`: toggle the visualization of keypoints
* `q`: quit

## TODO tasks
* Need to see how we can get the batchwise matching enable
* Currently the time it takes to match one image against 16k images is roughly 10 minutes which is no where close to the real time application we target