# MY 2023/3/27（必填）
# A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery（论文题目必填）
## 原文地址（必填）： https://arxiv.org/abs/2109.08937
## 代码地址（必填，没有写无）：https://github.com/WangLibo1995/GeoSeg
## 好的解读（可有可无）：https://zhuanlan.zhihu.com/p/551978086
## 我的看法（必填）：
这是一篇语义分割的文章（***需要说明！！！***）。CNN的encoder+Transformer的decoder，网络参数量非常小。这个网络是目前所有遥感项目里的SOTA，这说明unet结构依然大有可为。文章对transformer的结构进行了进一步的修正，使他的计算量很小。这篇文章的代码非常好复现，所用的timm库提供了很多cnn的backbone。
