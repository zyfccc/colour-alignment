# Colour-Alignment-for-Relative-Colour-Constancy-via-Non-standard-References

Demonstration code and dataset for the paper titled "Colour alignment for relative colour constancy via non-standard references". For further exploration, please cite our paper:

Arxiv link: https://arxiv.org/abs/2112.15106


## Work in a nutshell

![framework](https://user-images.githubusercontent.com/5927085/195493933-0419b71f-7e6c-4414-ae1a-ad14bf994b04.jpg)

* The concept of relative colour constancy (RCC): the ability to perceive colours of objects, invariant to the colour of light source **and camera**.

* A three-step process for performing colour alignment and achieving RCC: camera response calibration, response linearisation, and colour matching.

* A balance-of-linear-distances (BoLD) feature for camera response calibration that works with non-standard references, e.g., colour charts without true colour values.

* A new image dataset that contains images taken by different cameras under varied illuminations. (To be released).


## Code

### Requirements

* Tensorflow 1.14.0 (for the Optimisation approach)
* opencv-python 4.4.0


### Demonstration

We demonstrate how to perform the camera response calibration, response linearisation, and colour matching steps of the proposed colour alignment reported in the paper using an example image dataset.

For performing the Selection approach of the camera response calibration on the dataset, run:

```
python camera_response_calibration_selection.py --img_path './datasets/modified_Middlebury_calibration/' --tag_path './datasets/modified_Middlebury_calibration/tags.json' --crf_save_path './results/crfs/' --vis 1
```

* `img_path`: path to the calibration image dataset.
* `tag_path`: path to the label json file. 
* `crf_save_path`: where the generated CRFs will be stored.
* `vis`: whether to visualise the generated CRF. `1` indicates to visualise while `0` otherwise.

The generated camera response functions (CRFs) will be saved in the destination folder indicated by `crf_save_path`.


Next, run the following script to perform response linearisation of the images in the dataset:

```
python response_linearisation.py --img_path './datasets/modified_Middlebury_test/' --crf_save_path './results/crfs/' --output_path './results/images/'
```

* `img_path` is the path to the test image dataset.
* `output_path` indicates where the linearised images will be stored.

Response linearised output images will be stored in the destination folder indicated by `output_path`.


Eventually, the response linearised images will be colour matched to each other by executing the following script:

```
python colour_matching_and_benchmark.py --img_path './results/images/' --tag_path './datasets/modified_Middlebury_test/tags.json' --items '19,14' --intensity True --chromaticity True
```




## Files structure

```
- dataset/
   - dorf/
      | dorfCurves.txt     # Database of Response Functions
      | invemor.txt        # Inverse Empirical Model of Responses
   - modified_Middlebury_calibration/    # for camera calibration
      - CanonEOS1DsMarkII/
         ...
      - CanonPowerShotG9/
         ...
      ...
      | tags.json    # label information
   - modified_Middlebury_test/    # for benchmark
      - CanonEOS1DsMarkII/
         ...
      - CanonPowerShotG9/
         ...
      ...
      | tags.json    # label information
- libs/     # supporting libraries
   ...
- results/
   - crfs      # storing calibrated crfs
   - images    # storing response linearised images
| camera_response_calibration_selection.py      # camera response calibration script using the Selection approach
| camera_response_calibration_optimisation.py   # camera response calibration script using the Optimisation approach for comparison
| response_linearisation.py                     # script for image response linearisation
| colour_matching_and_benchmark.py              # image colour matching for relative colour constancy evaluation
```


## Dataset

The `dataset` folder contains the image dataset for camera calibration and demonstration for RCC performance evaluation.

A list of the cameras selected in the modified Middlebury dataset:
* Canon EOS-1Ds Mark II
* Canon PowerShot G9
* Canon PowerShot G10
* Canon PowerShot S50
* Canon PowerShot S60
* Casio EX-Z55
* Nikon D70
* Nikon D200
* Olympus E10
* Olympus E500
* Panasonic DMC-LX3
* Pentax K10D
* Sony DSLR-A100
* Sony DSLR-A300

`tags.json` file provides label information that indicates regions of colour patches in the images.

Indexes of colour patches in this demonstration:

![indexes1](https://user-images.githubusercontent.com/5927085/195492641-a51cd19f-1769-4850-bc5d-973016e65491.png)

Indexes of colour patches reported in the paper:

![indexes2](https://user-images.githubusercontent.com/5927085/195492698-b40b3235-0d15-4f49-942f-7fee88569d4f.png)