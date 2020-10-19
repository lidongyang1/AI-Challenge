## AI for Wireless (Challenges)

此项目是参加AI + 无线通信挑战的原始代码以及保存的模型
项目参与组最终初赛排名为14名，初赛分数：0.99175786320
其中final_code文件夹下保存了参数提交得最终原始代码，最终提交网络模型可以在model_history中20201017_0.99175786320文件夹下Model_save文件夹下找到。
model_history文件夹下上传了部分比赛上传代码，文件夹命名分别表示提交时间和对应得分数。
有任何问题可以随时跟我们联系，感谢各位审核指导！
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


## 如何上传到github

- 1 首先克隆本项目到本地电脑
- 2 按照自己的需要更改代码
- 3 推送到GitHub
  - 3.1 git lfs install （由于本项目涉及到大文件，所以需要git lfs，第一步是初始化）
  - 3.2 git track "*.zip" "*.h5" （添加要追踪的大文件，此处涉及到两种类型，强哥的zip和东阳的h5）
  - 3.3 git add 文件名或者文件夹名或者一个“.” （当要添加一个或者单独几个文件时，写文件名简介；当要添加整个文件夹是，用文件夹名称；当不知道文件在哪里时，就用一个“.”吧，就是全部的文件都加进来）
  - 3.4 git commit -m "此次更新的简单或者详细说明" （双引号的内容会显示到log上）
  - 3.5 git push origin master （开始推送）

以上的前提是你对此工程有访问和更新权限，并且进行了git add remote 相关的配置，具体请Google
