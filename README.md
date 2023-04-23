# DDA4210Project

Greetings!

This is the code implementation of our group project of DDA4210. More specifically, we mainlly refered to the cartoon-gan struture with a little bit modifications in training logic. The whole project was aimed at playing around style transfer, getting farmiliar with deep learning coding, and having a lot of fun. We train a model that could transfer landscape photos of mountan Tai and Guilin to trainditional Chinese painting style. 

### Result Checking
You can find all the report-shown images in ./result

### Reproducing Results
You need to first download required Chinese painting dataset from: https://paperswithcode.com/dataset/chinese-traditional-painting-dataset
Also, if you are interested in anime faces transfer, you need to download: https://www.kaggle.com/datasets/prasoonkottarathil/gananime-lite, and https://www.kaggle.com/datasets/ashwingupta3012/male-and-female-faces-dataset
Then, create a folder names `data` and unzip download data to this folder.

#### Requirements
python >= 3.8
pytorch >= 1.10
install pillow, cv2, torchvision with the lastest version

#### Preprocessing
The preprocessing was done by preprocessing.ipynb, where we remove low quality images and refine image size.

#### Edge promothing
For anime style type, cartoon-gan applied edge promoting to assist the model learning knowledge of strokes. Though this part was not included in the Chinese painting style transder procedure, it could be done by run edge_promoting.ipynb.

#### Pre-train
Pre-training the generator to reconstruct input content image. Run the pipeline written in pretrain.ipynb and pretrain_anime.ipynb for Chinese traditional printing and Anime faces respectively.

#### Training
Run the two train.ipynb
