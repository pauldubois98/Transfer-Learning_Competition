# Transfer-Learning_Competition

Code for the Transfer-Learning competition.

## Re-Labelling

We explore metrics in the re-labelling folder.
We then relabel the FFHQ dataset (in the data subfolder).
We finally perform a metrics analysis.

Here are statistics on the first 1000 images of the FFHQ dataset:

![nose area analysis](data/nose_area.png)
![nose width analysis](data/nose_width.png)
![lips area analysis](data/lips_area.png)
![eyes ration analysis](data/eyes_ratio.png)

## Transformations

### Continous Transformations

We calculate continuous deformations of the images to make the desired transformations:

- first, for each keypoint, we define the desired translation
- then, we interpolate between the keypoints
- finally, we smooth out the translation map

_(Inverting this map gives the opposite transformation)_

![continuous deformation explanation](continuousDeformations/deformation_expl.png)

For the nose, we have 4 possible transformations:

![nose transformations](continuousDeformations/nose_transfo.png)

---

We use a similar technique to make large and small lips:

![lips transformations](continuousDeformations/lips_transfo.png)

_we found that when the mouth is open,
this usually do not work that well._

---

And again to make round or narrow eyes:
![eyes transformations](continuousDeformations/eyes_transfo.png)

### Skin tone transforms

To change the skin tone, we create a mask of the skin (using RGB conditions):

![skin mask](continuousDeformations/skin_mask.png)

_note that we used RGB conditions,
but HVS conditions might work better..._

using this mask, we get what skin tone is the person;
we then apply the correspondong transform in the HVS colorspace:

![skin tones changes](continuousDeformations/skin_tones.png)

_note that we only transform the V and S values,
this gives better results, and can be interpreted as
"the H value is a chracteristic of the person,
while the V&S values correspond to their skin tone"_

### Bag under eyes

To create bags under eyes, we just darken the region under the eyes:

![bag under eyes transformations](continuousDeformations/bag_under_eyes.png)
