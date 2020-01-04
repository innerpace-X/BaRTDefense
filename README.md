# BaRTDefense
An unofficial implementation of BaRT Defense  
From paper 《Barrage of Random Transforms for Adversarially Robust Defense》  
This project ismainly based on the codes from the paper's supplement.

link: http://openaccess.thecvf.com/content_CVPR_2019/html/Raff_Barrage_of_Random_Transforms_for_Adversarially_Robust_Defense_CVPR_2019_paper.html

This model is fine-tuned from ResnetV2-50 with 1216 images.
We evaluated Linf norm attack in 16 epsilon with 200 iterations to this model, only 50 images succeed.

We have removed some of transformations from this paper, because they cost too much time.

This project was built in:  
Tensorflow 1.14.0  
scipy 1.1.0  
scikit-image 0.16.2  
Pillow 6.0.0  
