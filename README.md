## CSegNet Crack Detection
This repository contains CSegNet model code and crack detection dataset for detection purposes.

This crack segmentation dataset contains around 11.298 images which are merged from 10 available crack segmentation dataset.
The name prefix of each image is assigned to the corresponding dataset that the image belong to. 
There're also images which contain no crack, which could be filtered out by the pattern "noncrack*"
All the images in the dataset are resized to the size of (448, 448).

the two folders images and masks contain all the images.

The specific dataset used in this study is as follows:

Crack500:
>@inproceedings{zhang2016road,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={Image Processing (ICIP), 2016 IEEE International Conference on},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={arXiv preprint arXiv:1901.06340},
  year={2019}
}

CrackForest: 
>@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}

@inproceedings{cui2015pavement,
  title={Pavement Distress Detection Using Random Decision Forests},
  author={Cui, Limeng and Qi, Zhiquan and Chen, Zhensong and Meng, Fan and Shi, Yong},
  booktitle={International Conference on Data Science},
  pages={95--102},
  year={2015},
  organization={Springer}
}

GAPs384: 
>@inproceedings{eisenbach2017how,
  title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
  author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus
          and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike
          and Gross, Horst-Michael},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  pages={2039--2047},
  year={2017}
}

CrackTree:
>@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}

AEL: 
>@article{amhaz2016automatic,
  title={Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection.},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent}
}

Sylvie Chambon:
https://www.irit.fr/~Sylvie.Chambon/Crack_Detection_Database.html

Volker & Rissbilder:
>@article{pak2021crack,
  title={Crack Detection Using Fully Convolutional Network in Wall-Climbing Robot},
  author={Park, J.J., Fong, S.J., Pan, Y., Sung, Y.},
  journal={Springer},
  volume={715},
  pages={267--272},
  year={2021},
  doi={10.1007/978-981-15-9343-7_36}
}

Eugen Muller:
>@article{ham2021training,
  title={Training a semantic segmentation model for cracks in the concrete lining of tunnel},
  author={Sangwoo Ham, Soohyeon Bae, Hwiyoung Kim, Impyeong Lee, Gyu-Phil Lee, Donggyou Kim},
  journal={ Journal of Korean Tunnelling and Underground Space Association 2021},
  volume={23},
  number={6},
  pages={549--558},
  year={2021},
  doi={10.9711/KTAJ.2021.23.6.549}
}

DeepCrack:
>@article{liu2019deepcrack,
  title={DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  doi={10.1016/j.neucom.2019.01.036}
}

