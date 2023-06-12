# Watermark Learning Resources
This Github repository summarizes a list of **Watermark Learning** resources. For more details and the categorization criteria, please refer to our [survey](https://github.com/20110240069/Watermark-Learning-Resources). 

We will try our best to continuously maintain this Github Repository in a weekly manner.


#### Why Watermark Learning?
Backdoor learning is an emerging research area, which discusses the security issues of the training process towards machine learning algorithms. It is critical for safely adopting third-party training resources or models in reality.  

<!-- Note: 'Backdoor' is also commonly called the 'Neural Trojan' or 'Trojan'. -->


## News
<!-- * 2023/01/25: I am deeply sorry that I have recently suspended the reading of related papers and the updates of this Repo, due to some personal issues such as sickness and writing Ph.D. dissertation. I will restart the update as soon as possible.
* 2022/12/05: I slightly change the repo format by placing conference papers before journal papers. Specifically, in the same year, please place the conference paper before the journal paper, as journals are usually submitted a long time ago and therefore have some lag.
* 2022/12/05: I add three ECCV'22 papers. All papers for this conference should have been included now.
* 2022/10/06: I add a new Applied Intelligence paper [Agent Manipulator: Stealthy Strategy Attacks on Deep
Reinforcement Learning](https://link.springer.com/article/10.1007/s10489-022-03882-w) in [Reinforcement Learning](#reinforcement-learning).
* 2022/10/06: I add four new arXiv papers, including **(1)** [ImpNet: Imperceptible and blackbox-undetectable backdoors in compiled neural networks](https://arxiv.org/pdf/2210.00108.pdf) in [Other Attacks](#other-attacks), **(2)** [Shielding Federated Learning: Mitigating Byzantine Attacks with Less Constraints](https://arxiv.org/pdf/2210.01437) in [Federated Learning](#federated-learning), **(3)** [Backdoor Attacks in the Supply Chain of Masked Image Modeling](https://arxiv.org/pdf/2210.01632.pdf) in [Semi-Supervised and Self-Supervised Learning](#semi-supervised-and-self-supervised-learning), and **(4)** [Invariant Aggregator for Defending Federated Backdoor Attacks](https://arxiv.org/pdf/2210.01834.pdf) in [Federated Learning](#federated-learning). -->


## Reference
If our repo or survey is useful for your research, please cite our paper as follows:
<!-- ```
@article{li2022backdoor,
  title={Backdoor learning: A survey},
  author={Li, Yiming and Jiang, Yong and Li, Zhifeng and Xia, Shu-Tao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
``` -->


## Contributing
<p align="center">
  <img src="http://cdn1.sportngin.com/attachments/news_article/7269/5172/needyou_small.jpg" alt="We Need You!">
</p>

Please help to contribute this list by contacting [me](http://fudanmas.com/#/pages/home/real) or add [pull request](https://github.com/20110240069/Watermark-Learning-Resources/pulls)

Markdown format:
```markdown
- Paper Name. 
  [[pdf]](link) 
  [[code]](link).
  - Author 1, Author 2, Author 3. *Conference/Journal*, Year.
```
**Note**: In the same year, please place the conference paper before the journal paper, as journals are usually submitted a long time ago and therefore have some lag. (*i.e.*, Conferences-->Journals-->Preprints)


## Table of Contents
- [Survey](#survey)
- [Toolbox](#toolbox)
- [Dissertation and Thesis](#dissertation-and-thesis)
- [White-box Watermark](#white-box-watermark) 
- [Black-box Watermark](#black-box-watermark) 
- [No-box Watermark](#no-box-watermark)
- [Attack on Watermark](#attack-on-watermark)
- [Evaluation](#evaluation)
- [Other Model Protection Methods](#other-model-protection-methods)
  - [Fragile watermark](#fragile-watermark)
  - [Hardware-based](#hardware-based)  
  - [Model Fingerprinting](#model-encryption)
  - [Model Hashing](#model-hashing)
  - [Active Control](#active-control)
  - [Reversible Watermarking](#reversible-watermarking)
  - [Model Encryption](#model-encryption)
- [Competition](#competition)


## Survey
- A systematic review on model watermarking for neural networks. [[link]](https://www.frontiersin.org/articles/10.3389/fdata.2021.729663/full)
  - Franziska Boenisch. Frontiers in big Data, 2021.

- DNN intellectual property protection: Taxonomy, attacks and evaluations. [[pdf]](https://www.researchgate.net/profile/Mingfu-Xue/publication/351844272_DNN_Intellectual_Property_Protection_Taxonomy_Attacks_and_Evaluations_Invited_Paper/links/61236bb91e95fe241aed7882/DNN-Intellectual-Property-Protection-Taxonomy-Attacks-and-Evaluations-Invited-Paper.pdf)
  - Mingfu Xue, Jian Wang, Weiqiang Liu. Great Lakes Symposium on VLSI, 2021.

- A survey of deep neural network watermarking techniques. [[pdf]](https://arxiv.org/pdf/2103.09274.pdf)
  -  Yue Lia , Hongxia Wangb and Mauro Barnic. Neurocomputing, 2021.

- Protecting artificial intelligence IPs: a survey of watermarking and fingerprinting for machine learning. [[pdf]](https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cit2.12029?download=true)
  - Francesco Regazzoni, Paolo Palmieri, Fethulah Smailbegovic, Rosario Cammarota, Ilia Polian. CAAI Transactions on Intelligence Technology, 2021.
- Intellectual property protection for deep learning models: Taxonomy, methods, attacks, and evaluations. [[pdf]](https://arxiv.org/pdf/2011.13564.pdf)
  - Mingfu Xue, Yushu Zhang, Jian Wang, and Weiqiang Liu. IEEE Transactions on Artificial Intelligence, 2021.

- Copyright protection of deep neural network models using digital watermarking: a comparative study. [[link]](https://link.springer.com/article/10.1007/s11042-022-12566-z)
  - Alaa Fkirin, Gamal Attiya, Ayman El-Sayed, Marwa A. Shouman. Multimedia Tools and Applications, 2022.
  
- Intellectual property protection of DNN models. [[link]](https://link.springer.com/article/10.1007/s11280-022-01113-3)
  - Sen Peng, Yufei Chen, Jie Xu, Zizhuo Chen, Cong Wang and Xiaohua Jia. World Wide Web, 2022.

 

## Toolbox



## Dissertation and Thesis




## White-box Watermark
### 2017
- Embedding watermarks into deep neural networks.
  [[pdf]](https://cafetarjome.com/wp-content/uploads/1004/translation/602af92d8bd0142a.pdf)
  - Uchida, Yusuke and Nagai, Yuki and Sakazawa, Shigeyuki and Satoh, Shin'ichi. *ICMR*, 2017.

### 2019

- Deepsigns: An end-to-end watermarking framework for protecting the ownership of deep neural networks.
  [[pdf]](https://aceslab.org/sites/default/files/DeepSigns.pdf)
  - BD Rouhani, H Chen, F Koushanfar. *ASPLOS*, 2019.

- Deepmarks: A secure fingerprinting framework for digital rights management of deep learning models.
  [[pdf]](https://aceslab.org/sites/default/files/DeepMarks_ICMR.pdf)
  - Chen, Huili and Rouhani, Bita Darvish and Fu, Cheng and Zhao, Jishen and Koushanfar, Farinaz. *ICMR*, 2019.

- Rethinking deep neural network ownership verification:Embedding passports to defeat ambiguity attacks.
  [[pdf]](https://proceedings.neurips.cc/paper/8719-rethinking-deep-neural-network-ownership-verification-embedding-passports-to-defeat-ambiguity-attacks.pdf)
  [[code]](https://github.com/kamwoh/DeepIPR)
  - Fan, Lixin, Kam Woh Ng, and Chee Seng Chan. *NIPS*, 2019.

- Visual decoding of hidden watermark in trained deep neural network.
  - Sakazawa, Shigeyuki and Myodo, Emi and Tasaka, Kazuyuki and Yanagihara, Hiromasa. *MIPR*, 2019.

### 2020
- Watermarking in deep neural networks via error back-propagation.
  [[pdf]](https://hzwu.github.io/dnnWatermarking2020.pdf)
  - Wang, Jiangfeng and Wu, Hanzhou and Zhang, Xinpeng and Yao, Yuwei. *Electronic Imaging*, 2020.

- Watermarking neural network with compensation mechanism.
  - Feng Le,Zhang Xinpeng. *KSEM*, 2020.

- Passport-aware normalization for deep model protection.
  [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf)
  - Zhang, Jie and Chen, Dongdong and Liao, Jing and Zhang, Weiming and Hua, Gang and Yu, Nenghai. *NIPS*, 2020.

- Adam and the Ants: On the Influence of the Optimization Algorithm on the Detectability of DNN Watermarks.
  [[pdf]](https://www.mdpi.com/1099-4300/22/12/1379)
  - Cortiñas-Lorenzo, Betty, and Fernando Pérez-González. *Entropy*, 2020.

### 2021
- Delving in the loss landscape to embed robust watermarks into neural networks.
  - Tartaglione, Enzo and Grangetto, Marco and Cavagnino, Davide and Botta, Marco. *ICPR*, 2021.

- RIGA:Covert and robust white-box watermarking of deep neural networks.
  [[pdf]](https://www.researchgate.net/profile/Tianhao-Wang-15/publication/345829945_RIGA_Covert_and_Robust_White-Box_Watermarking_of_Deep_Neural_Networks/links/5faf684645851518fda2e38c/RIGA-Covert-and-Robust-White-Box-Watermarking-of-Deep-Neural-Networks.pdf)
  - Tianhao Wang, Florian Kerschbaum. *WWW*, 2021.

- White-box watermarking scheme for fully-connected layers in fine-tuning model.
  [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3437880.3460402)
  - Kuribayashi, Minoru and Tanaka, Takuro and Suzuki, Shunta and Yasui, Tatsuya and Funabiki, Nobuo. *IH&MMSec*, 2021.

- Watermarking Deep Neural Networks with Greedy Residuals.
  [[pdf]](https://openreview.net/pdf?id=8FUlWs8cLq)
  [[code]](https://github.com/eil/greedy-residuals)
  - Liu, Hanwen, Zhenyu Weng, and Yuesheng Zhu. *ICML*, 2021.

- Spread-transform dither modulation watermarking of deep neural network.
  [[pdf]](https://arxiv.org/pdf/2012.14171.pdf)
  [[code]](https://github.com/bunny859000040/ST-DM_DNN_watermarking)
  - Yue Li, Benedetta Tondi, Mauro Barni. *Journal of Information Security and Applications*, 2021.

- A Feature-Map-Based Large-Payload DNN Watermarking Algorithm.
  [[code]](https://github.com/bunny859000040/feature_based_DNN_watermarking)
  - Yue Li, Lydia Abady, Hongxia Wang, Mauro Barni. *IWDW*, 2021.

- Don't Forget to Sign the Gradients!
  [[pdf]](https://arxiv.org/abs/2103.03701.pdf)
  - Omid Aramoon, Pin-Yu Chen, Gang Qu. *MLSys*, 2021.

- You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership.
  [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2021/file/0dfd8a39e2a5dd536c185e19a804a73b-Paper.pdf)
  [[code]](https://github.com/VITA-Group/NO-stealing-LTH)
  - Xuxi Chen, Tianlong Chen, Zhenyu Zhang, Zhangyang Wang. *NIPS*, 2021.

### 2022
- Fostering The Robustness Of White-Box Deep Neural Network Watermarks By Neuron Alignment.
  [[pdf]](https://arxiv.org/abs/2112.14108.pdf)
  - Fang-Qi Li, Shi-Lin Wang, Yun Zhu:. *ICASSP*, 2022.

- Fused Pruning based Robust Deep Neural Network Watermark Embedding.
  [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9956100)
  - Tengfei Li, Shuo Wang, Huiyun Jing, Zhichao Lian, Shunmei Meng, Qianmu Li. *ICPR*, 2022.

- Cosine Model Watermarking Against Ensemble Distillation.
  [[pdf]](https://doi.org/10.48550/arXiv.2203.02777.pdf)
  - Laurent Charette, Lingyang Chu, Yizhou Chen, Jian Pei, Lanjun Wang, Yong Zhang. *AAAI*, 2022.

- Encryption Resistant Deep Neural Network Watermarking.
  [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746461)
  - Guobiao Li, Sheng Li, Zhenxing Qian, Xinpeng Zhang. *ICASSP *, 2022.

- Identification for Deep Neural Network: Simply Adjusting Few Weights!
  - Yingjie Lao, Peng Yang, Weijie Zhao, Ping Li. *ICDE*, 2022.

- Subnetwork-Lossless Robust Watermarking for Hostile Theft Attacks in Deep Transfer Learning Models.
  - Jia, Ju and Wu, Yueming and Li, Anran and Ma, Siqi and Liu, Yang. *TDSC*, 2022.

- FedIPR: Ownership verification for federated deep neural network models.
  [[pdf]](https://arxiv.org/abs/2109.13236.pdf)
  [[code]](https://github.com/purp1eHaze/FedIPR)
  - Lixin Fan, Bowen Li, Hanlin Gu, Jie Li, Qiang Yang. *PAMI*, 2022.

- Defending against model stealing via verifying embedded external features.
  [[pdf]](https://arxiv.org/abs/2112.03476.pdf)
  [[code]](https://github.com/zlh-thu/StealingVerification)
  - Yiming Li, Linghui Zhu, Xiaojun Jia, Yong Jiang, Shu-Tao Xia, Xiaochun Cao. *AAAI*, 2022.

- Collusion Resistant Watermarking for Deep Learning Models Protection.
  - Sayoko Kakikura, Hyunho Kang, Keiichi Iwamura. *ICACT*, 2022.

- AIME: watermarking AI models by leveraging errors.
  - Dhwani Mehta, Nurun N. Mondol, Farimah Farahmandi, Mark M. Tehranipoor. *DATE*, 2022.

- Leveraging Multi-task Learning for Umambiguous and Flexible Deep Neural Network Watermarking.
  [[pdf]](http://ceur-ws.org/Vol-3087/paper_5.pdf)
  - Fangqi Li, Lei Yang, Shilin Wang, Alan Wee-Chung Liew:. *SafeAI@AAAI*, 2022.

### 2023
- A Robustness-Assured White-Box Watermark in Neural Networks.
  - Lv, Peizhuo and Li, Pan and Zhang, Shengzhi and Chen, Kai and Liang, Ruigang and Ma, Hualong and Zhao, Yue and Li, Yingjiu. *TDSC*, 2023.

- Intellectual property protection for deep semantic segmentation models.
  - Hongjia Ruan, Huihui Song, Bo Liu, Yong Cheng, Qingshan Liu. *FCS*, 2023.

- Deep Learning Model Protection using Negative Correlation-based Watermarking with Best Embedding Regions.
  - Kakikura, Sayoko and Kang, Hyunho and Iwamura, Keiichi. *ICACT*, 2023.

## Black-box Watermark
### 2018
- Turning your weakness into a strength:Watermarking deep neural networks by backdooring.
  [[pdf]](http://arxiv.org/abs/1802.04633.pdf)
  - Yossi Adi, Carsten Baum, Moustapha Cissé, Benny Pinkas, Joseph Keshet. *USENIX Security*, 2018.

- Protecting intellectual property of deep neural networks with watermarking.
  [[pdf]](https://gzs715.github.io/pubs/WATERMARK_ASIACCS18.pdf)
  - Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph. Stoecklin, Heqing Huang, Ian M. Molloy. *AsiaCCS*, 2018.

- Watermarking deep neural networks for embedded systems.
  [[pdf]](https://www.cs.ucla.edu/~miodrag/papers/Guo_ICCAD_2018.pdf)
  - Jia Guo, Miodrag Potkonjak. *ICCAD*, 2018.

### 2019
- Robust watermarking of neural network with exponential weighting.
  [[pdf]](http://arxiv.org/abs/1901.06151.pdf)
  - Ryota Namba, Jun Sakuma. *AsiaCCS*, 2019.

- How to prove your model belongs to you:A blind-watermark based framework to protect intellectual property of DNN.
  [[pdf]](https://zhenglisec.github.io/zheng_files/papers/acsac19.pdf)
  [[code]](https://github.com/zhenglisec/Blind-Watermark-for-DNN)
  - Zheng Li, Chengyu Hu, Yang Zhang, Shanqing Guo. *ACSAC *, 2019.

### 2020
- Adversarial frontier stitching for remote neural network watermarking.Neural Computing and Applications.
  [[pdf]](http://arxiv.org/abs/1711.01894.pdf)
  - Erwan Le Merrer, Patrick Pérez, Gilles Trédan. *Neural Computing and Applications*, 2020.

- Protecting IP of deep neural networks with watermarking:A new label helps.
  [[pdf]](https://www.researchgate.net/profile/Qi-Zhong-8/publication/341242767_Protecting_IP_of_Deep_Neural_Networks_with_Watermarking_A_New_Label_Helps/links/6141af1fea4aa800110495cf/Protecting-IP-of-Deep-Neural-Networks-with-Watermarking-A-New-Label-Helps.pdf)
  [[code]](https://github.com/ingako/PEARL)
  - Qi Zhong, Leo Yu Zhang, Jun Zhang, Longxiang Gao, Yong Xiang. *PAKDD*, 2020.

- Protecting the intellectual property of deep neural networks with watermarking: The frequency domain approach.
  [[pdf]](https://www.researchgate.net/profile/Qi-Zhong-8/publication/350859425_Protecting_the_Intellectual_Property_of_Deep_Neural_Networks_with_Watermarking_The_Frequency_Domain_Approach/links/6141b0b7dabce51cf4522387/Protecting-the-Intellectual-Property-of-Deep-Neural-Networks-with-Watermarking-The-Frequency-Domain-Approach.pdf)
  - Li, Meng and Zhong, Qi and Zhang, Leo Yu and Du, Yajuan and Zhang, Jun and Xiang, Yong. *TrustCom*, 2020.

- Secure neural network watermarking protocol against forging attack.
  [[pdf]](https://scholar.archive.org/work/3mijfyq5gjfqdns5sxgkkcl27q/access/wayback/https://jivp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13640-020-00527-1)
  - Renjie Zhu, Xinpeng Zhang, Mengte Shi, Zhenjun Tang. * EURASIP Journal on Image and Video Processing*, 2020.

- Watermarking deep neural networks in image processing.
  [[pdf]](https://csyhquan.github.io/manuscript/21-tnnls-Watermarking%20Deep%20Neural%20Networks%20in%20Image%20Processing.pdf)
  - Yuhui Quan, Huan Teng, Yixin Chen, Hui Ji. *TNNLS*, 2020.

- SpecMark: A Spectral Watermarking Framework for IP Protection of Speech Recognition Systems.
  [[pdf]](https://aceslab.org/sites/default/files/2020_SpecMark.pdf)
  [[code]]()
  - . *INTERSPEECH*, 2020.

### 2021
- Dawn: Dynamic adversarial watermarking of neural networks.
  [[pdf]](http://arxiv.org/abs/1906.00830.pdf)
  [[code]](https://github.com/ssg-research/dawn-dynamic-adversarial-watermarking-of-neural-networks)
  - Sebastian Szyller, Buse Gul Atli, Samuel Marchal, N. Asokan. *ACM MM*, 2021.

- Piracy-resistant DNN watermarking by block-wise image transformation with secret key.
  [[pdf]](https://arxiv.org/pdf/2104.04241)
  - Maung Maung, April Pyone, and Hitoshi Kiya. *IH&MMSec*, 2021.

- Robust watermarking for deep neural networks via bi-level optimization.
  [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Robust_Watermarking_for_Deep_Neural_Networks_via_Bi-Level_Optimization_ICCV_2021_paper.pdf)
  - Peng Yang, Yingjie Lao, Ping Li. *ICCV*, 2021.

- Robust black-box watermarking for deep neural network using inverse document frequency.
  [[pdf]](https://arxiv.org/pdf/2103.05590)
  - Mohammad Mehdi Yadollahi, Farzaneh Shoeleh, Sajjad Dadkhah, Ali A. Ghorbani. *DASC/PiCom/CBDCom/CyberSciTech*, 2021.

- Protecting intellectual property of generative adversarial networks from ambiguity attacks.
  [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ong_Protecting_Intellectual_Property_of_Generative_Adversarial_Networks_From_Ambiguity_Attacks_CVPR_2021_paper.pdf)
  [[code]](https://github.com/dingsheng-ong/ipr-gan)
  - Ding Sheng Ong, Chee Seng Chan, Kam Woh Ng, Lixin Fan, Qiang Yang. *CVPR*, 2021.

- Entangled Watermarks as a Defense against Model Extraction.
  [[pdf]](https://www.usenix.org/system/files/sec21-jia.pdf)
  [[code]](https://github.com/cleverhans-lab/entangled-watermark)
  - Hengrui Jia, Christopher A. Choquette-Choo, Varun Chandrasekaran, Nicolas Papernot. *USENIX Security*, 2021.

- Persistent watermark for image classification neural networks by penetrating the autoencoder.
  - Fang-Qi Li, Shi-Lin Wang. *ICIP*, 2021.

- Watermarking graph neural networks by random graphs.
  [[pdf]](https://arxiv.org/pdf/2011.00512)
  - Xiangyu Zhao, Hanzhou Wu, Xinpeng Zhang. *ISDFS*, 2021.

- Yes We can: Watermarking Machine Learning Models beyond Classification.
  [[pdf]](https://hal.science/hal-03220793/document)
  - Sofiane Lounici, Mohamed Njeh, Orhan Ermis, Melek Önen, Slim Trabelsi. *CSF*, 2021.

- WAFFLE: Watermarking in federated learning.
  [[pdf]](https://arxiv.org/pdf/2008.07298)
  [[code]](https://github.com/ssg-research/WAFFLE)
  - Buse G. A. Tekgul, Yuxi Xia, Samuel Marchal, N. Asokan. *SRDS*, 2021.

### 2022
- Speech Pattern Based Black-Box Model Watermarking for Automatic Speech Recognition.
  [[pdf]](https://arxiv.org/pdf/2110.09814)
  - Haozhe Chen, Weiming Zhang, Kunlin Liu, Kejiang Chen, Han Fang, Nenghai Yu. *ICASSP*, 2022.

- Sparse Trigger Pattern Guided Deep Learning Model Watermarking.
  [[pdf]](https://homepage.iis.sinica.edu.tw/papers/lcs/24868-F.pdf)
  - Chun-Shien Lu. *IH&MMSec*, 2022.

- Protect, show, attend and tell: Empowering image captioning models with ownership protection.
  [[pdf]](https://arxiv.org/pdf/2008.11009)
  [[code]](https://github.com/jianhanlim/ipr-imagecaptioning)
  - Jian Han Lim, Chee Seng Chan, Kam Woh Ng, Lixin Fan, Qiang Yang. *Pattern Recognition*, 2022.

- Watermarking pre-trained encoders in contrastive learning.
  [[pdf]](https://arxiv.org/pdf/2201.08217)
  - Yutong Wu, Han Qiu, Tianwei Zhang, Jiwei Li, Meikang Qiu. *ICDIS*, 2022.

- SSLGuard: A Watermarking Scheme for Self-supervised Learning Pre-trained Encoders.
  [[pdf]](https://arxiv.org/pdf/2201.11692)
  [[code]](https://github.com/tianshuocong/SSLGuard)
  - Tianshuo Cong, Xinlei He, Yang Zhang. *CCS*, 2022.

- Certified Neural Network Watermarks with Randomized Smoothing.
  [[pdf]](https://proceedings.mlr.press/v162/bansal22a/bansal22a.pdf)
  - Arpit Bansal, Ping-Yeh Chiang, Michael J. Curry, Rajiv Jain, Curtis Wigington, Varun Manjunatha, John P. Dickerson, Tom Goldstein. *ICML*, 2022.

- TADW: Traceable and Anti-detection Dynamic Watermarking of Deep Neural Networks.
  - Dong, Jinwei and Wang, He and He, Zhipeng and Niu, Jun and Zhu, Xiaoyan and Wu, Gaofei. *SCN*, 2022.

- BlindSpot: Watermarking Through Fairness.
  [[pdf]](https://www.eurecom.fr/publication/6890/download/sec-publi-6890.pdf)
  - Sofiane Lounici, Melek Önen, Orhan Ermis, Slim Trabelsi. *IH&MMSec*, 2022.

- Method for copyright protection of deep neural networks using digital watermarking.
  [[pdf]](https://www.researchgate.net/profile/Yuliya-Vybornova-2/publication/359036410_Method_for_copyright_protection_of_deep_neural_networks_using_digital_watermarking/links/6286444239fa2170315cb73b/Method-for-copyright-protection-of-deep-neural-networks-using-digital-watermarking.pdf)
  - Yuliya D. Vybornova. *ICMV*, 2022.

- Rose: A robust and secure dnn watermarking.
  [[pdf]](https://arxiv.org/pdf/2206.11024)
  - Kassem Kallas, Teddy Furon. *WIFS*, 2022.

- Watermarking of deep recurrent neural network using adversarial examples to protect intellectual property.
  [[pdf]](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2021.2008613)
  - Pulkit Rathi, Saumya Bhadauria, Sugandha Rathi. *Applied Artificial Intelligence*, 2022.

- RoSe: A RObust and SEcure Black-Box DNN Watermarking.
  [[pdf]](https://hal.inria.fr/hal-03806393/file/WIFS_2022-2.pdf)
  - Kallas, Kassem and Furon, Teddy. *WIFS*, 2022.

### 2023
- A Novel Model Watermarking for Protecting Generative Adversarial Network.
  - Tong Qiao, Yuyan Ma, Ning Zheng, Hanzhou Wu, Yanli Chen, Ming Xu, Xiangyang Luo. *Computers & Security*, 2023.

- Deep neural network watermarking based on a reversible image hiding network.
  - Wang, Linna and Song, Yunfei and Xia, Daoxun. *PAA*, 2023.

- Unambiguous and High-Fidelity Backdoor Watermarking for Deep Neural Networks.
  [[code]](http://github.com/ghua-ac/dnn_watermark)
  - Hua, Guang and Teoh, Andrew Beng Jin and Xiang, Yong and Jiang, Hao. *TNNLS*, 2023.

- Universal BlackMarks: Key-Image-Free Blackbox Multi-Bit Watermarking of Deep Neural Networks.
  - Li Li, Weiming Zhang, Mauro Barni. *SPL*, 2023.

- Generative Model Watermarking Based on Human Visual System.
  [[pdf]](https://arxiv.org/pdf/2209.15268)
  - Li Zhang, Yong Liu, Shaoteng Liu, Tianshu Yang, Yexin Wang, Xinpeng Zhang, Hanzhou Wu. *IFTC*, 2023.

- Mixer: DNN Watermarking using Image Mixup.
  [[pdf]](https://arxiv.org/pdf/2212.02814)
  - Kassem Kallas, Teddy Furon. *ICASSP*, 2023.









## No-box Watermark
- Adversarial watermarking transformer: Towards tracing text provenance with data hiding.  
  [[pdf]](https://arxiv.org/pdf/2009.03015.pdf)
  - Abdelnabi S,Fritz M. IEEE Symp on Security and Privacy, 2021.

- Watermarking neural networks with watermarked images. 
  [[link]](https://ieeexplore.ieee.org/abstract/document/9222304)
  - Hanzhou Wu, Gen Liu, Yuwei Yao, Xinpeng Zhang. IEEE Transactions on Circuits and Systems for Video Technology, 2021.

- Deep model intellectual property protection via deep watermarking. 
  [[pdf]](https://arxiv.org/pdf/2103.04980.pdf)
  - Jie Zhang, Dongdong Chen, Jing Liao, Weiming Zhang, Huamin Feng, Gang Hua, Nenghai Yu. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

- Protecting intellectual property of language generation apis with lexical watermark. 
  [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/download/21321/21070)
  - Xuanli He, Qiongkai Xu, Lingjuan Lyu, Fangzhao Wu, Chenguang Wang. Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

- Supervised gan watermarking for intellectual property protection. 
  [[pdf]](https://arxiv.org/pdf/2209.03466.pdf)
  - Jianwei Fei, Zhihua Xia, Benedetta Tondi, Mauro Barni. IEEE International Workshop on Information Forensics and Security (WIFS), 2022.


## Attack on Watermark
- Attacks on digital watermarks for deep neural networks. 
  [[pdf]](https://scholar.harvard.edu/files/tianhaowang/files/icassp.pdf)
  - Tianhao Wang, Florian Kerschbaum. Proc of IEEE Int Conf on Acoustics,Speech and Signal Processing.Piscataway, 2019.

- Leveraging unlabeled data for watermark removal of deep neural networks. 
  [[pdf]](https://ruoxijia.info/wp-content/uploads/2020/03/watermark_removal_icml19_workshop.pdf)]
  - Xinyun Chen, Wenxiao Wang, Yiming Ding, Chris Bender, Ruoxi Jia, Bo Li, Dawn Song. ICML workshop on Security and Privacy of Machine Learning, 2019.

- Fine-tuning is not enough: A simple yet effective watermark removal attack for DNN models. 
  [[pdf]](https://arxiv.org/pdf/2009.08697.pdf)
  - Shangwei Guo, Tianwei Zhang, Han Qiu, Yi Zeng, Tao Xiang and Yang Liu. International Joint Conferences on Artificial Intelligence Organization (IJCAI), 2021.

- Refit:A unified watermark removal framework for deep learning systems with limited data. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3433210.3453079)
  - Xinyun Chen, Wenxiao Wang, Chris Bender, Yiming Ding, Ruoxi Jia, Bo Li, Dawn Song. ACM Asia Conf on Computer and Communications Security, 2021.

- On the robustness of backdoor-based watermarking in deep neural networks. [[pdf]](https://arxiv.org/pdf/1906.07745.pdf)
  - Masoumeh Shafieinejad, Jiaqi Wang, Nils Lukas, Xinda Li, Florian Kerschbaum. ACM Workshop on Information Hiding and Multimedia Security, 2021.

- NeuNAC: A novel fragile watermarking algorithm for integrity protection of neural networks. [[link]](https://www.sciencedirect.com/science/article/pii/S0020025521006642)
  - Marco Botta, Davide Cavagnino, Roberto Esposito. Information Sciences, 2021.

- Removing backdoor-based watermarks in neural networks with limited data. [[pdf]](https://arxiv.org/pdf/2008.00407.pdf)
  - Xuankai Liu, Fengting Li, Bihan Wen, Qi Li. International Conference on Pattern Recognition (ICPR), 2021.

- Effective ambiguity attack against passport-based dnn intellectual property protection schemes through fully connected layer substitution. [[pdf]](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Effective_Ambiguity_Attack_Against_Passport-Based_DNN_Intellectual_Property_Protection_Schemes_CVPR_2023_paper.pdf)
  -  Yiming Chen, Jinyu Tian, Xiangyu Chen and Jiantao Zhou. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

- Detect and remove watermark in deep neural networks via generative adversarial networks. [[pdf]](https://arxiv.org/pdf/2106.08104.pdf)
  - Haoqi Wang, Mingfu Xue, Shichang Sun, Yushu Zhang, Jian Wang and Weiqiang Liu. Information Security: 24th International Conference, ISC 2021.

- Attention Distraction: Watermark Removal Through Continual Learning with Selective Forgetting. [[pdf]](https://arxiv.org/pdf/2204.01934.pdf)
  - Qi Zhong, Leo Yu Zhang, Shengshan Hu, Longxiang Gao, Jun Zhang, Yong Xiang. IEEE International Conference on Multimedia and Expo (ICME), 2022.

- Rethinking White-Box Watermarks on Deep Learning Models under Neural Structural Obfuscation. [[pdf]](https://www.usenix.org/system/files/sec23fall-prepub-444-yan-yifan.pdf)
  - Yifan Yan, Xudong Pan, Mi Zhang, Min Yang. USENIX security symposium (USENIX Security 23), 2023.

- Linear Functionality Equivalence Attack Against Deep Neural Network Watermarks and a Defense Method by Neuron Mapping. [[link]](https://ieeexplore.ieee.org/abstract/document/10077406)
  - Fang-Qi Li, Shi-Lin Wang and Alan Wee-Chung Liew. IEEE Transactions on Information Forensics and Security, 2023.

- Rethinking the Vulnerability of DNN Watermarking: Are Watermarks Robust against Naturalness-aware Perturbations? [[pdf]](https://dl.acm.org/doi/abs/10.1145/3503161.3548390)
  - ACM International Conference on Multimedia, 2022.

- Watermark Removal Scheme Based on Neural Network Model Pruning.
  - International Conference on Machine Learning and Natural Language Processing, 2022.

- Removing Watermarks For Image Processing Networks Via Referenced Subspace Attention. [[link]](https://dl.acm.org/doi/abs/10.1145/3503161.3548390)
  - Run Wang, Haoxuan Li, Lingzhou Mu, Jixing Ren, Shangwei Guo, Li Liu, Liming Fang, Jing Chen, Lina Wang. The Computer Journal, 2022.

- Attacks on Recent DNN IP Protection Techniques and Their Mitigation. [[link]](https://ieeexplore.ieee.org/abstract/document/10115275)
  -  Rijoy Mukherjee and Rajat Subhra Chakraborty. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems ,2023.



## Evaluation
- Sok: How robust is image classification deep neural network watermarking? [[pdf]](https://arxiv.org/pdf/2108.04974.pdf)
  - Nils Lukas, Edward Jiang, Xinda Li, Florian Kerschbaum. IEEE Symposium on Security and Privacy (SP), 2022.

- Evaluating the robustness of trigger set-based watermarks embedded in deep neural networks. [[pdf]](https://arxiv.org/pdf/2106.10147.pdf)
  - Suyoung Lee, Wonho Song, Suman Jana, Meeyoung Cha, and Sooel Son. IEEE Transactions on Dependable and Secure Computing, 2022.

## Other Model Protection Methods

### Fragile watermark

-  Sensitive-sample fingerprinting of deep neural networks. [[pdf]](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.pdf)
  - Zecheng He, Tianwei Zhang, Ruby Lee. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

- Fragile neural network watermarking with trigger image set. [[pdf]](https://drive.google.com/file/d/1RoUfRSsGrGyOhC1OohdmtDdf_ardyMCL/view)
  - Renjie Zhu, Ping Wei, Sheng Li, Zhaoxia Yin, Xinpeng Zhang and Zhenxing Qian. Int Conf on Knowledge Science, 2021.

- DeepiSign: Invisible fragile watermark to protect the integrity and authenticity of CNN. [[pdf]](https://arxiv.org/pdf/2101.04319.pdf)
  - Alsharif Abuadbba, Hyoungshick Kim, Surya Nepal. Annual ACM Symposium on Applied Computing, 2021.

- NeuNAC: A novel fragile watermarking algorithm for integrity protection of neural networks. [[link]](https://www.sciencedirect.com/science/article/pii/S0020025521006642)
  - Marco Botta, Davide Cavagnino, Roberto Esposito. Information Sciences, 2021.

- Neural network fragile watermarking with no model performance degradation. [[pdf]](https://arxiv.org/pdf/2208.07585.pdf)
  - Zhaoxia Yin, Heng Yin, Xinpeng Zhang. IEEE International Conference on Image Processing (ICIP), 2022.

- Deepauth: A dnn authentication framework by model-unique and fragile signature embedding. [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/download/21193/20942)
  - Yingjie Lao1, Weijie Zhao, Peng Yang, Ping Li. AAAI Conference on Artificial Intelligence, 2022.

- Verifying integrity of deep ensemble models by lossless black-box watermarking with sensitive samples. [[pdf]](https://arxiv.org/pdf/2205.04145.pdf)
  - Lina Lin and Hanzhou Wu. International Symposium on Digital Forensics and Security (ISDFS), 2022.

### Hardware-based

- Hardware-assisted intellectual property protection of deep learning models. [[pdf]](https://eprint.iacr.org/2020/1016.pdf)
  - Abhishek Chakraborty, Ankit Mondal, and Ankur Srivastava. ACM/IEEE Design Automation Conference (DAC), 2020.

- Ownership Verification of DNN Architectures via Hardware Cache Side Channels. [[pdf]](https://arxiv.org/pdf/2102.03523.pdf)
  - Xiaoxuan Lou, Shangwei Guo, Jiwei Li, and Tianwei Zhang. IEEE Transactions on Circuits and Systems for Video Technology, 2022.

- DeepHardMark: Towards watermarking neural network hardware. [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/20367/20126)
  - Joseph Clements, Yingjie Lao. AAAI Conference on Artificial Intelligence, 2022.

- PUF-Based Intellectual Property Protection for CNN Model. [[link]](https://link.springer.com/chapter/10.1007/978-3-031-10989-8_57)
  - Dawei Li, Yangkun Ren, Di Liu, Zhenyu Guan, Qianyun Zhang, Yanzhao Wang and Jianwei Liu. Knowledge Science, Engineering and Management: 15th International Conference(KSEM), 2022.



### Model Fingerprinting

- DeepAttest: An end-to-end attestation framework for deep neural networks. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3307650.3322251)

  - Huili Chen, Cheng Fu, Bita Darvish Rouhani, Jishen Zhao, Farinaz Koushanfar. International Symposium on Computer Architecture, 2019.

- AFA: Adversarial fingerprinting authentication for deep neural networks. [[link]](https://www.sciencedirect.com/science/article/abs/pii/S014036641931686X)

  - Jingjing Zhao, Qingyue Hu, Gaoyang Liu, Xiaoqiang Ma, Fei Chen, Mohammad Mehedi Hassan. Computer Communications, 2020.

- IPGuard: Protecting intellectual property of deep neural networks via fingerprinting the classification boundary. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3433210.3437526)

  - Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong. ACM Asia Conference on Computer and Communications Security, 2021.

- Characteristic Examples: High-Robustness, Low-Transferability Fingerprinting of Neural Networks. [[pdf]](https://par.nsf.gov/servlets/purl/10297119)

  - Siyue Wang , Xiao Wang , Pin-Yu Chen , Pu Zhao and Xue Lin. International Joint Conferences on Artificial Intelligence Organization (IJCAI), 2021.

- ModelDiff: testing-based DNN similarity comparison for model reuse detection. [[pdf]](https://arxiv.org/pdf/2106.08890.pdf)

  - Yuanchun Li, Ziqi Zhang, Bingyan Liu, Ziyue Yang, Yunxin Liu. ACM SIGSOFT International Symposium on Software Testing and Analysis, 2021

- Deep neural network fingerprinting by conferrable adversarial examples. [[pdf]](https://arxiv.org/pdf/1912.00888.pdf)

  - Nils Lukas, Yuxuan Zhang, Florian Kerschbaum. International Conference on Learning Representations, 2021

- Intrinsic examples: Robust fingerprinting of deep neural networks. [[pdf]](https://par.nsf.gov/servlets/purl/10300680)

  - Siyue Wang, Pu Zhao, Xiao Wang, Sang Chin, Thomas Wahl, Yunsi Fe, Qi Alfred Chen, Xue Lin. British Machine Vision Conference (BMVC), 2021.

- Tafa: A task-agnostic fingerprinting algorithm for neural networks. [[pdf]](https://drive.google.com/file/d/1cMOlTCMfFTW1iPjhClBt6GWpc-jZrHVk/view)

  - Xudong Pan, Mi Zhang, Yifan Lu, and Min Yang. Computer Security–ESORICS 2021: 26th European Symposium on Research in Computer Security, 2021.

- Fingerprinting deep neural networks-a deepfool approach. [[link]](https://ieeexplore.ieee.org/abstract/document/9401119)

  - Si Wang, Chip-Hong Chang. IEEE International Symposium on Circuits and Systems (ISCAS), 2021 .

- MetaV: A Meta-Verifier Approach to Task-Agnostic Model Fingerprinting. [[link]](https://dl.acm.org/doi/abs/10.1145/3534678.3539257)

  - Xudong Pan, Yifan Yan, Mi Zhang, Min Yang. ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022.

- Metafinger: Fingerprinting the deep neural networks with meta-training. [[pdf]](https://www.ijcai.org/proceedings/2022/0109.pdf)

  - Kang Yang, Run Wang, Lina Wang. International Joint Conference on Artificial Intelligence (IJCAI), 2022.

- Fingerprinting deep neural networks globally via universal adversarial perturbations. [[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Peng_Fingerprinting_Deep_Neural_Networks_Globally_via_Universal_Adversarial_Perturbations_CVPR_2022_paper.pdf)

  - Zirui Peng, Shaofeng Li, Guoxing Chen,  Cheng Zhang, Haojin Zhu,  Minhui X. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

- A DNN Fingerprint for Non-Repudiable Model Ownership Identification and Piracy Detection. [[link]](https://ieeexplore.ieee.org/abstract/document/9854806)

  -  Yue Zheng, Si Wang, Chip-Hong Chang. IEEE Transactions on Information Forensics and Security, 2022.

- Copy, right? A testing framework for copyright protection of deep learning models. [[pdf]](https://arxiv.org/pdf/2112.05588.pdf)

  - Jialuo Chen, Jingyi Wang, Tinglan Peng, Youcheng Sun, Peng Cheng, Shouling Ji, Xingjun Ma, Bo Li and Dawn Song. IEEE Symposium on Security and Privacy (SP), 2022.

- Mitigating Query-based Neural Network Fingerprinting via Data Augmentation. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3597933)

  -  MEIQI WANG, HAN QIU, TIANWEI ZHANG, MEIKANG QIU, BHAVANI THURAISINGHAM. ACM Transactions on Sensor Networks, 2023.

  

### Model Hashing

- Perceptual hash of neural networks. [[pdf]](https://www.mdpi.com/2073-8994/14/4/810/pdf)

  -  Zhiying Zhu, Hang Zhou, Siyuan Xing, Zhenxing Qian, Sheng Li and Xinpeng Zhang. Symmetry, 202.
- Neural Network Model Protection with Piracy Identification and Tampering Localization Capability. [[link]](https://dl.acm.org/doi/abs/10.1145/3503161.3548247)

  -  Cheng Xiong, Guorui Feng, Xinran Li, Xinpeng Zhang, Chuan Qin. ACM International Conference on Multimedia, 2022.
- DNN self-embedding watermarking: Towards tampering detection and parameter recovery for deep neural  network. [[link]](https://www.sciencedirect.com/science/article/abs/pii/S0167865522003063)

  -  Gejian Zhao, Chuan Qin, Heng Yao, Yanfang Han. Pattern Recognition Letters, 2022.
- Graph-based Robust Model Hashing. [[link]](https://ieeexplore.ieee.org/abstract/document/9975424)

  - Yitong Tao, Chuan Qin. IEEE International Workshop on Information Forensics and Security (WIFS), 2022.
- Perceptual Hashing of Deep Convolutional Neural Networks for Model Copy Detection. [[link]](https://dl.acm.org/doi/abs/10.1145/3572777)

  -  Haozhe Chen, Hang Zhou, Jie Zhang, Dongdong Chen, Weiming Zhang, Kejiang Chen, Gang Hua, Nenghai Yu. ACM Transactions on Multimedia Computing, Communications and Applications, 2023.
- Perceptual Model Hashing: Towards Neural Network Model Authentication. [[link]](https://ieeexplore.ieee.org/abstract/document/9949087)

  - Xinran Li, Zichi Wang, Guorui Feng, Xinpeng Zhang, Chuan Qin. IEEE International Workshop on Multimedia Signal Processing (MMSP), 2022.

### Active Control

- Active DNN IP protection: A novel user fingerprint management and DNN authorization control technique. [[link]](https://ieeexplore.ieee.org/abstract/document/9343023)

  - Mingfu Xue, Zhiyu Wu, Can He, Jian Wang, Weiqiang Liu. IEEE International Conference on Trust, 2020.
- Training DNN model with secret key for model protection. [[pdf]](https://arxiv.org/pdf/2008.02450.pdf)

  - MaungMaung AprilPyone and Hitoshi Kiya. IEEE Global Conference on Consumer Electronics (GCCE), 2020.
- Transfer learning-based model protection with secret key. [[pdf]](https://arxiv.org/pdf/2103.03525.pdf)

  - MaungMaung AprilPyone and Hitoshi Kiya. IEEE International Conference on Image Processing (ICIP), 2021.
- Sample-Specific Backdoor based Active Intellectual Property Protection for Deep Neural Networks. [[link]](https://ieeexplore.ieee.org/abstract/document/9869927)

  - Yinghao Wu, Mingfu Xue, Dujuan Gu, Yushu Zhang, Weiqiang Liu. IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS), 2022.
- Active intellectual property protection for deep neural networks through stealthy backdoor and users' identities authentication. [[link]](https://link.springer.com/article/10.1007/s10489-022-03339-0)

  -  Mingfu Xue, Shichang Sun, Yushu Zhang, Jian Wang and Weiqiang Liu. Applied Intelligence, 2022.
- AdvParams: An active DNN intellectual property protection technique via adversarial perturbation based parameter encryption. [[pdf]](https://arxiv.org/pdf/2105.13697.pdf)

  -  Mingfu Xue, Zhiyu Wu, Jian Wang, Yushu Zhang, and Weiqiang Liu. IEEE Transactions on Emerging Topics in Computing, 2022.
- Access control of semantic segmentation models using encrypted feature maps. [[pdf]](https://www.nowpublishers.com/article/OpenAccessDownload/SIP-2022-0013)

  -  Hiroki Ito, MaungMaung AprilPyone, Sayaka Shiota and Hitoshi Kiya. APSIPA Transactions on Signal and Information Processing, 2022.
- Image and model transformation with secret key for vision transformer. [[pdf]](https://arxiv.org/pdf/2207.05366.pdf)

  - Hitoshi KIYA, Ryota IĲIMA, MaungMaung APRILPYONE and Yuma KINOSHITA. IEICE TRANSACTIONS on Information and Systems, 2023.
- ActiveGuard: An active intellectual property protection technique for deep neural networks by leveraging adversarial examples as users' fingerprints. [[pdf]](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/cdt2.12056)

  -  Mingfu Xue, Shichang Sun, Can He, Dujuan Gu, Yushu Zhang, Jian Wang, Weiqiang Liu. IET Computers & Digital Techniques, 2023.
- Active Authorization Control of Deep Models Using Channel Pruning. [[link]](https://link.springer.com/chapter/10.1007/978-981-99-2385-4_40)

  - Linna Wang, Yunfei Song, Yujia Zhu and Daoxun Xia. Computer Supported Cooperative Work and Social Computing, ChineseCSCW 2022. Springer Nature Singapore, 2023.

### Reversible Watermarking
- Reversible watermarking in deep convolutional neural networks for integrity authentication. [[pdf]](https://arxiv.org/pdf/2104.04268.pdf)

  - Xiquan Guan, Huamin Feng, Weiming Zhang, Hang Zhou, Jie Zhang, Nenghai Yu. ACM International Conference on Multimedia, 2020.


### Model Encryption


- Chaotic weights: A novel approach to protect intellectual property of deep neural networks. [[pdf]](https://luhang-ccl.github.io/files/chaotic-TCAD.pdf)

  - Ning Lin, Xiaoming Chen, Hang Lu  and Xiaowei L. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020.

## Competition

  
















