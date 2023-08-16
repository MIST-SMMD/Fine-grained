# MIST-SMMD:A Spatio-Temporal Information Extraction Method Based on Multimodal Social Media Data
### [Preprint](https://www.preprints.org/manuscript/202305.1205/v2) | [Paper](https://www.mdpi.com/journal/ijgi)
<br/>

>A spatio-temporal information extraction method based on multimodal social media data: A case study on urban inundation  
>[Yilong Wu](https://github.com/uyoin),  [Yingjie Chen](https://github.com/FalleNSakura2002),[Rongyu Zhang](https://github.com/hz157), [Zhenfei Cui](http://geo.fjnu.edu.cn/main.htm), [Xinyi Liu](http://geo.fjnu.edu.cn/main.htm), [Jiayi Zhang](http://geo.fjnu.edu.cn/main.htm), [Meizhen Wang](http://dky.njnu.edu.cn/info/1213/3986.htm), [Yong Wu](http://geo.fjnu.edu.cn/3e/21/c4964a81441/page.htm)<sup>*</sup>  
> Nodata  

Discussions about the paper are welcomed in the [discussion panel](https://github.com/discussions).

![Data](img/LSGL.png)

## Introduction
To extract fine-grained spatial information from high-quality microblog image-text data above, a series of image processing techniques are required to compare it with street view images that already contain spatial information and thereby screen out the best match for spatial information migration. In this process, the matching degree between the social media images and street view images determines the reliability of the fine-grained spatial information. To maximize the reliability of this process as much as possible, we designed a cascade model LSGL (LoFTR-Seg Geo-Localization) based on match-extraction-evaluation. 
In this cascade model, we have used [LoFTR](https://github.com/zju3dv/LoFTR) and [DETR](https://github.com/facebookresearch/detr) as part of the model, respectively, thanks to the original authors for their efforts
- ### Effect of Each Level of the Model on the Matching Results:
![Figure 1](img/figure1.png)

## Colab demo
Want to run MIST-SMMD with custom image pairs without configuring your own GPU environment? Try the Colab demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BO1gBlShIJn0E0LILbBlghXcaQ85N5XQ?usp=sharing)

## Installation
**Conda:**
```bash
conda env create -f environment.yaml
conda activate mist
```
**Pip:**
``` bash
pip install -r requirements.txt
```
