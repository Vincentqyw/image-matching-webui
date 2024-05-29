Welcome to WxBS version 1.1 --  Wide (multiple) Baseline Dataset.

It contains 34 very challenging image pairs with manually annotated ground truth correspondences.
The images are organized into several categories:

- WGALBS: with Geometric, Appearance and iLlumination changes
- WGBS: with Geometric (viewpoint) changes
- WLABS: with iLlumination and Appearance changes. The viewpoint change is present, but not significant
- WGSBS: with Geometric and Sensor (thermal camera vs visible) changes
- WGABS: with Geometric and Appearance changes.

Compared to the original dataset from 2015, v.1.1 contains more correspondences, which are also cleaned, and 3 additional image pairs: WGALBS/kyiv_dolltheater, WGALBS/kyiv_dolltheater2, WGBS/kn-church.
We also provide cross-validation errors for each of the GT correspondences. 
They are estimated in the following way:

- the fundamental matrix F is estimated with OpenCV 8pt algorithm (no RANSAC), using all points, except one.
F, _  = cv2.findFundamentalMat(corrs_cur[:,:2], corrs_cur[:,2:], cv2.FM_8POINT)
Then the symmetrical epipolar distance is calculatd on that held-out point. We have used kornia implementation of the symmetrical epipolar distance:


From Hartley and Zisserman, symmetric epipolar distance (11.10)
sed = (x'^T F x) ** 2 /  (((Fx)_1**2) + (Fx)_2**2)) +  1/ (((F^Tx')_1**2) + (F^Tx')_2**2))

https://kornia.readthedocs.io/en/latest/geometry.epipolar.html#kornia.geometry.epipolar.symmetrical_epipolar_distance


The labeling is done using [pixelstitch](https://pypi.org/project/pixelstitch/)

There are main intended ways of using the dataset.
a) First, is evaluation of the image matchers, which are estimating fundamental matrix. One calculates reprojection error on the GT correspondences and report mean error, or the percentage of the GT correspondences, which are in agreement with the estimated F. For more details see the paper[1]

b) For the methods like [CoTR](https://arxiv.org/abs/2103.14167), which look for the correspondences in the image 2, given the query point in image 1, one can directly calculate error between returned point and GT correspondence.
 

***
If you are using this dataset, please cite us:

[1] WxBS: Wide Baseline Stereo Generalizations. D. Mishkin and M. Perdoch and J.Matas and K. Lenc. In Proc BMVC, 2015

@InProceedings{Mishkin2015WXBS, 
   author = {{Mishkin}, D. and {Matas}, J. and {Perdoch}, M. and {Lenc}, K. },
   booktitle = {Proceedings of the British Machine Vision Conference}, 
   publisher = {BMVA}, 
   title = "{WxBS: Wide Baseline Stereo Generalizations}",
   year = 2015,
   month = sep
}
