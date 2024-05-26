# 肌电信号（EMG）在力学方程充当损失函数下的模型训练效果研究



本实验在EMG-肌肉力联合数据集上做了一定的探索。早期未加入力学方程损失函数（下称 newloss），以CNN-均方差这一常见组合对数据做了测试，拟合效果良好，数据有效。虽然本工作主要是对newloss 的研究，但前期对数据的处理相信也具有一定参考价值，因此对早期的文件不做删除。

所有文件列出如下表，研究newloss 的文件已用备注注明，也可以从文件名含有`newloss`,`nl`来判断。`kmmf`是第一版数据集的相关代码，`emgmf`是第二版的数据集，后者最全面，newloss 研究以第二版数据集为主，后文不再特殊说明。

```python
cnn_emgmf_try.ipynb #以CNN和全连接等神经网络对第二版数据的测试
cnn_emgmf_try_newloss.ipynb #初代newloss的测试
cnn_emgmf_try_nlv2_409.ipynb #第二代newloss的测试
cnn_kmmf.ipynb #数据集（第一版）的模型训练与测试
emgmf_dataprocess.ipynb #数据集（第二版）的数据处理
emgmf_exportresult .ipynb #重播训练记录，用于寻找合适点和研究问题
emgmf_intersubject_try_426.ipynb #测试newloss 的迁移性能（于不同个体的数据之间）
emgmf_nltry.ipynb #编写损失函数时用于单独测试newloss的代码
emgmf_nlv2_export_0501_serverdata.ipynb
kmmfDataprocess.ipynb #数据集（第一版）的数据处理
```

## 研究过程概述

接收到的第一版数据集数据量较小，不论是肌电采集点的个数还是时间序列的长度均比较小，其相关研究以 kmmf 做简称。为此，首先编写了`kmmfDataprocess.ipynb`,`cnn_kmmf.ipynb` 完成了基于神经网络的拟合。不过由于数据量较小，无法很好地体现出方法的缺陷（如果有）。

第二版数据集数量充分，可以较好的展开研究。`emgmf_dataprocess.ipynb`，`cnn_emgmf_try.ipynb`两个文件完成了对数据的处理和模型训练工作。这一工作得以为之后进行newloss 的研究打下基础，之后如果需要改动损失函数，也只需要更新损失函数的那一部分的代码即可。

初代newloss 的设计思路如下：在正常的计算均方误差后，还需要再计算一个动力学平衡方程的解。由于方程涉及时间上的求导和二次求导，为了保证每次计算时都能成功运行，它被设计成了必须依靠上一刻的数据来获得这一刻的数据的形式。因此，这个方程无法兼容并行数据（也就是batch size 大于1）。代码的编写首先在`emgmf_nltry.ipynb`完成并测试，然后加入到`cnn_emgmf_try.ipynb`成为`cnn_emgmf_try_newloss.ipynb`。

上一代的newloss 在实验中发现了一些问题，新的公式在`cnn_emgmf_try_nlv2_409.ipynb`完成测试。

## 代码解释

`cnn_emgmf_try_newloss.ipynb`和`cnn_emgmf_try_nlv2_409.ipynb`是本研究的主要代码，具有充分的代码注释。其他的文件虽然也有注释，但由于未经过补充，可读性没有这两个文件高。不过，这些文件可以理解为研究做到一定程度后的备份，因此实际上并非是完全独立的。这两个主要代码文件实际上已经包括了所有需要重点关注的代码部分。