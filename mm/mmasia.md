题目：SpikMamba: When SNN meets Mamba in Event-based Human Action Recognition

发表会议：MMASIA '24

本文是MMASIA 2024入选论文《SpikMamba: When SNN meets Mamba in Event-based Human Action Recognition》的中文解读。本文的第一作者陈佳祺为东北大学研究生，研究方向为Human Action Recognition & Few-shot Learning。本文的其他合作者分别来自东北大学、澳大利亚国立大学和北京理工大学。

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/title.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
    title
  </figcaption>
</figure>

# 1.introduction

人体动作识别（Human Activity Recognition, HAR）在视频分析、监控、自动驾驶、机器人技术和医疗保健等领域具有至关重要的作用。目前，大多数 HAR 算法基于 RGB 图像展开研究，借助其丰富的视觉细节。然而，在隐私敏感场景中，基于 RGB 的方法因会记录可识别特征而引发担忧。与此相对，事件相机通过仅在像素级捕捉场景亮度变化来呈现信息，而不生成完整的图像，在保护用户隐私方面展现了良好前景。此外，事件相机具有高动态范围，能有效应对低光或强对比度等复杂光照环境。但是，基于事件相机的 HAR 还面临空间稀疏和高时间分辨率带来的建模挑战。鉴于此，本文围绕基于事件相机的人体动作识别展开研究，重点探讨：是否能设计出一种框架，既能在保护用户隐私的同时，又准确识别人体动作？

主要贡献：

① 本文提出了一个名为 SpikMamba 的框架，利用事件数据有效地、准确地识别人类行为。

② 本文探索了 Mamba 和基于窗口的线性注意脉冲基机制，用于模拟事件数据中的全局和局部时间依赖性。

③ 本文使用常见的基于事件的 HAR 数据进行实验，展示了 SpikMamba 相对于现有最先进算法具有优越性能。

# 2.Background

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/background.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
图1. 基于事件的人体动作识别方法概览
  </figcaption>
</figure>

① ANN方法：ANN方法通过时间下采样事件数据来减少计算量，并使用注意力机制、卷积神经网络(CNN)和图神经网络(CNN)从空间稀疏的事件中提取特征，从而实现高性能。然而，这种方法会丢失关于人体动作的细粒度信息，而这些信息可能进一步提升模型性能。

② SNN方法：SNN方法利用SpikRNN和SpikGCN从空间稀疏的事件中有效地提取特征。然而，这类方法的计算通常仅限于局部时间上下文，这导致对全局时间依赖性的捕捉不足，从而无法准确识别人体动作。

③ 本文方法(SpikMamba)：我们将Mamba和基于窗口的线性注意力机制结合到SNN中，以高效建模事件数据的全局和局部时间依赖性，从而准确识别人体动作。

# 3.Related Works

## 3.1 基于事件相机的人体动作识别

基于ANN的方法通常使用CNN [1]、ViTs [2] 和GCNs [3] 提取稀疏事件数据的特征。例如，EV-ACT [1] 利用带有时空注意力机制的CNN进行动作识别。ViTs [2] 使用基于patch和体素的Transformer编码器高效提取时空特征，而GCNs [3] 则管理稀疏和异步的数据结构。然而，大多数基于ANN的HAR方法忽视了事件数据的空间稀疏性和高时间分辨率的问题，导致细节信息的丢失。

相比之下，基于SNN的方法更适合处理时间序列数据。脉冲神经网络（SNNs）[4, 5, 6] 使用离散脉冲序列而非连续小数值，天然适用于处理事件驱动的HAR任务。然而，这些方法往往因事件数据的下采样丢失细粒度动作细节，同时在捕捉全局时间依赖性方面存在不足。


## 3.2 状态空间模型(SSM)

状态空间模型（State-Space Model, S4）[7] 是一种用于长距离依赖建模的替代方法，相较于CNN和Transformer。Mamba已经被应用于事件数据 [8, 9]，其中[8] 将脉冲前端集成到Mamba中以进行时间处理，[9] 使用线性复杂度的状态空间模型进行跟踪。

# 4.本文方法

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/framework1.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
图2. SpikMamba的框架
  </figcaption>
</figure>

本文采用[11]中的方法将事件数据转换为三通道事件图像  $X \in \mathbb{R}^{3 \times T \times H \times W} $，其中  $T 、 H 、 W  $分别表示事件图像的时间维度、高度和宽度。我们使用SpikMamba（如图2所示）预测事件图像  X  的动作类别。SpikMamba包含两个主要模块：

① Spiking 3D Patch Embedding
该模块将事件帧$  X  $划分为小块（patch），并通过SNN（Spiking Neural Network）计算这些patch的嵌入表示  $P $。

② SpikMamba Block
此模块结合了基于窗口的线性注意力机制和Mamba模块，并嵌入到SNN中，用于建模事件数据的局部和全局时间依赖性，以从patch嵌入中提取特征，从而实现人体动作识别（HAR）。

最后，SpikMamba块生成的嵌入通过池化操作进行汇总，并通过一个最终的线性层将嵌入投影到动作类别，用于分类任务。

# 5.实验设置

## 5.1 数据集

我们使用四个数据集来评估我们的模型SpikMamba的性能。这些数据集包括PAF、HARDVS、DVSGesture和E-FAction。具体而言：

①	PAF是一个使用DVSIS346事件相机采集的人体动作数据集，包含10个动作类别，每个类别有45个样本。

②	HARDVS是最近发布的一个数据集，拥有最多的动作类别和样本，总计300个类别和107,646条记录。

③	DVSGesture捕捉手部和手臂的运动，包含11个动作类别，分辨率为128×128。

④	E-FAction数据集包含128个人体动作类别，总计1024条记录，分辨率为346×260。

## 5.2 实现细节

我们采用AFE表示法 [11] 将事件流压缩为事件帧。我们的SpikMamba模型包括一个用于脉冲3D补丁嵌入的层，以及两层SpikMamba模块用于特征提取。我们使用256的隐藏状态维度，并通过Linearm(·)将状态空间方程的状态维度扩展到256。状态空间方程在2048的维度上运行，前馈网络的隐藏维度为1024。在训练过程中，我们使用Adam优化器，权重衰减设置为 $2\times10^{-4}$。学习率初始化为 $1 \times 10^{-5}$，并采用CosineAnnealingLR学习率调度器，最低学习率设置为 $1 \times 10^{-6}$。我们的模型在两块NVIDIA 4090 GPU上进行了训练，训练100个epoch，批量大小为32。

# 6.实验与分析（完整实验分析见论文）

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/table1.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
    表1：在PAF、HARDVS、DVSGesture 和E-FAction数据集上，对基于事件的动作识别模型与当前最先进(state-of-the-art)方法进行对比。这些模型使用准确率(ACC)进行评估，同时列出了模型类型(即ANN或SNN)。准确率最高的方法以加粗字体显示。
  </figcaption>
</figure>

## 6.1 主要结果

① 在四个数据集上，本文的方法分别取得了96.28%、97.32%、99.01%和71.02%的最优准确率。

② 与基于ANN的ExACT(第二高准确率)相比，本文的方法在准确率上分别提高了1.45%、7.22%、0.33%和3.09%。

③ 在HARDVS数据集上，本文的SpikMamba和ExACT在与其他方法相比表现出显著的改进，准确率提高了35%以上。此外，本文的SpikMamba还通过提高ExACT的准确率额外提高了7.22%。

④ 在DVGesture数据集上，最先进方法的最高准确率已经是98.86%，但本文的SpikMamba将其提高到了99.01%。

⑤ 与四个数据集上第二高的SNNs(次高准确率)相比，SpikMamba在PAF和DVSGesture数据集上的准确率分别提高了6.14%和2.81%，这是第一个比ANN方法更好的SNN方法。

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/table2.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
    表2：对SpikeSLA与SpikMamba层进行的消融实验。加粗字体表示最高准确率。
  </figcaption>
</figure>

## 6.2 消融实验

在表2中，本文消融了模型中的SpikeSLA和SpikMamba层。得到以下结果：
① 仅使用本文模型的SpikeSLA层，网络在捕捉高时间分辨率事件数据中的长期/全局信息方面的能力显著下降，准确率分别为97.12%，95.33%，98.17%和70.66%。

② 当从模型中移除SpikeSLA层时，观察到准确率的显著下降。平均下降为19.73%。考虑到四个数据集中的动作持续时间主要在5到7秒之间，动作的关键帧很可能是短期，构成了动作的主要特征。因此，当从的模型中移除SpikeSLA时，网络无法有效地增强HAR的特征局部性。

③ 具有SpikeSLA和SpikMamba层的模型高效且准确地模拟了事件数据的全球和局部时间依赖性，并具有最佳性能。

## 6.3 讨论

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/attention.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
    图3：SpikMamba 的注意力图示例。高注意力区域以白色标记，低注意力区域以黑色标记。该注意力图显示出SpikMamba能够有效关注图像中包含人类动作的区域。
  </figcaption>
</figure>

① 注意力图。在图3中，本文展示了最后一个时间步中SpikMamba模块的注意力图。为方便理解，提供了由SpikMamba生成的RGB图像上的注意力图。高注意力区域以白色标记，低注意力区域以黑色标记。SpikMamba能够有效地捕捉图像中包含人类动作的区域。

<figure style="text-align:center; margin:0 auto;">
  <img src="/Users/chenjiaqi/Documents/master/code/acmmmasia/images/table3.png"
       alt="示例图片"
       style="display:block; margin:0 auto;" />
  <figcaption style="font-size:0.9em; text-align:center;">
   表3：我们将SpikMamba与ExACT与EVT的计算效率进行了对比，这两者分别是最先进的ANN与SNN算法。
  </figcaption>
</figure>

② 计算效率。我们在表3中将SpikMamba与最新的ANN和SNN方法(ExACT和EvT)在计算效率方面进行了比较。SpikMamba结合了SNN和Mamba的优势，高效捕捉事件数据中的全局依赖性，并通过基于尖峰窗口的线性注意力机制建模事件数据的局部依赖性，在计算效率和性能之间达到了平衡。在参数数量和FLOPs方面，我们的方法最少，同时在人类行为识别(HAR)任务中表现优于现有最先进的ANN和SNN方法。

# 7.讨论与总结

本文提出了一种名为SpikMamba的基于事件数据的人体动作识别(HAR)方法。使用事件数据进行人体动作识别时，如何从空间稀疏且具有高时间分辨率的事件数据中有效提取有意义的特征是一大挑战。本文利用脉冲神经网络(SNN)的计算优势以及Mamba在长序列建模方面的能力，使得SpikMamba能够从稀疏且高时间分辨率的事件流中有效地捕捉全局依赖关系。此外，我们还提出了一种基于脉冲窗口的线性注意力机制，以增强在HAR任务中对事件数据进行建模时的局部性。在常见的基于事件数据的 HAR数据集上进行的实验表明，与现有的最先进的ANN和SNN方法相比，SpikMamba展现出了更出色的性能。

# 引用
[1] Yue Gao, et al. 2023. Action recognition and benchmark using event cameras. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).
[2] Alberto Sabater, et al. 2022. Event transformer. asparse-aware solution for efficient event data processing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2677–2686.
[3] Enrico Calabrese, et al. 2019. DHP19: Dynamic vision sensor 3D human pose dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 0–0.
[4] Tong Bu, et al. 2023. Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks. arXiv preprint arXiv:2303.04347 (2023).
[5] Yongqiang Cao, et al. 2015. Spiking deep convolutional neural networks for energy-efficient object recognition. International Journal of Computer Vision 113 (2015), 54–66.
[6] Wei Fang, et al. 2021. Incorporating learnable membrane time constant to enhance learning of spiking neural networks. In Proceedings of the IEEE/CVF international conference on computer vision. 2661–2671.
[7] Albert Gu, et al. 2021. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396 (2021).
[8] Jiahao Qin and Feng Liu. 2024. Mamba-Spike: Enhancing the Mamba Architecture with a Spiking Front-End for Efficient Temporal Data Processing. arXiv preprint arXiv:2408.11823 (2024).
[9] Xiao Wang, et al. 2024. MambaEVT: Event Stream based Visual Object Tracking using State Space Model. arXiv preprint arXiv:2408.10487 (2024).
[10] Jiazhou Zhou, et al. 2024. ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action Recognition and More. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18633–18643.
