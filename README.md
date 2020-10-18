## AI for Wireless (Challenges)

此项目是参加AI + 无线通信挑战的原始代码以及保存的模型

## 模型介绍

请介绍初赛作品方案，如实现方式、特点等

方案说明：
AIGroup_WMCT团队初赛作品实验主要采用Tensorflow（keras）框架实现，部分实验采用Pytorch和Tensorflow。最终审核作品为Tensorflow（keras）版本。实验环境为NVIDIA DGX-1（Tesla V100），相关软件为官方指定版本。
作品特点：
所设计Encoder模型较小，Decoder模型较大，适用于客户端运算能力较弱，中心运算能力强的无线通信环境。
1. Encoder：采用多尺度特征融合方式完成非线性映射，其中一路将Input直接叠加到Encoder的dense层输出之前，防止梯度弥散。
2. Decoder：借鉴残差网络和Inception的思想  结合GoogleNet和ResNet的优点，利用多卷积核（3*3,2*2等）学习样本更丰富的特征；综合利用Swish与ReLU激活函数，通过多epochs重复Reload模型进行模型再训练，进一步结合LearningRate调整，训练Decoder来实现信号的精确重建。
 
## 实验设置

待完善