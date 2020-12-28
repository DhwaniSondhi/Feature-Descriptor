## Feature-Descriptor
This is the assignment of COMP6341 COMPUTER VISION.  This assignment aims to detect discriminating features in an image and find the best matching features in other images. Features need to be invariant to translation, rotation, and illumination. This assignment is further used to create the software for building panoramas.

### Description
**Feature detection**
- Aimed to identify points of interest in the image using the Harris corner detection.
- For each point in the image, considered a window of pixels around that point.
- Computed the Harris matrix H for that point, defined as:
<img src="https://github.com/DhwaniSondhi/Feature-Descriptor/blob/master/images/1.PNG" alt="alt" width="800" height="80"/><br/>
- Computed the corner strength function (the "Harris operator"):<br/>
<img src="https://github.com/DhwaniSondhi/Feature-Descriptor/blob/master/images/2.PNG" alt="alt" width="400" height="80"/><br/>
- Compared the strength for each point with a user-defined threshold and include points with greater value.
- Selected points should have strength value to be the local maximum in at least a 3x3 neighborhood.


**Feature description**
- Created a descriptor for the feature centered at each interest point taking into consideration the rotational invariance. 
- Considered contrast invariant features.

**Feature matching**
- Aimed for a given feature in one image, find the best matching feature in one or more other images. 
- Compared to two features by outputting a distance between them.
- Two distance measures used for comparison:
  1. SSD distance: threshold on the match score.
  2. Ratio Test: (score of the best feature match)/(score of the second-best feature match).
- Implemented [adaptive non-maximum suppression](https://www.microsoft.com/en-us/research/publication/multi-scale-oriented-patches/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F70120%2Ftr-2004-133.pdf).

### How to run?
- Set up an environment and install cv2 and NumPy.
- Keep the images in the same folder.
- Give the image names as input.
- Output produced is the points matched between the two images.

### Outputs
<img src="https://github.com/DhwaniSondhi/Feature-Descriptor/blob/master/results/building1/Output.png" alt="alt" width="700" height="400"/> <br/>
<img src="https://github.com/DhwaniSondhi/Feature-Descriptor/blob/master/results/building2/Output.png" alt="alt" width="700" height="400"/> <br/>
<img src="https://github.com/DhwaniSondhi/Feature-Descriptor/blob/master/results/mountains/Output.png" alt="alt" width="700" height="400"/> <br/>

[Click here for more outputs](https://github.com/DhwaniSondhi/Feature-Descriptor/tree/master/results) <br/>
[Click here for the assignment description](https://github.com/DhwaniSondhi/Feature-Descriptor/blob/master/Assignment.pdf)
