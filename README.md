# CSU-ML-Inclass-Project
中南大学智能科学与技术专业机器学习课程设计。
### 包含&可支持算法
| \ |FNN    |RBF-NN        |Conv2d-NN                 |
|---|-------|--------------|--------------------------|
|算法支持情况说明| 支持  |  支持高斯核    |     支持ResNet和VGG架构    | 
### 内置的任务
|\      |BCI-Competition| Mnist | CIFAR-10|
|-------|---------------|-------|---------|
|任务支持情况说明|  BCI-Competition III 中的[Dataset I](https://www.bbci.de/competition/iii/)| 内置Mnist同时也可更换成Mnist-Fashion| 支持 |
|采用算法       | RBF-NN |   CNN | CNN |
### 支持的优化器(Optimizer)
|\         | SGD | SGD + Momentum | Adam | RMSprop | RMSprop + Momentum|
|----------|-----|----------------|------|---------|-------------------|
|支持情况说明| 支持 |    支持        | 已实现但效果不好| 支持 | 支持 | 
### 其他注意事项
- 支持使用GPU进行加速，请自行根据显存大小控制batch_size以免OOM
- FNN使用GPU加速时，训练时间较短；CNN使用GPU加速时，仍然需要很多时间（算法写的不好别骂了），个人推测是卷积操作的正向和反向传播过程中reshape的transpose等算子花费的时间较长
- 关于最终指标，在Mnist上可以达到96%的准确率，在CIFAR-10上可以达到64%+（具体为啥这么差俺也不清楚，有兴趣的可以研究一下）
- 不要直接clone过去当自己的作业！
