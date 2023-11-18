# MMMViT: Multiscale Multimodal Vison Transformer for Brain Tumor Segmentation with Missing Modalities

## Abstract

Accurate segmentation of brain tumors from multimodal MRI sequences is a critical prerequisite for brain tumor diagnosis, prognosis, and surgical treatment. While one or more modalities are often missing in clinical practice, which can collapse most previous methods that rely on all modality data. To deal with this problem, the current state-of-the-art Transformer-related approach directly fuses available modality-specific features to learn a shared latent representation, with the aim of extracting common features that are robust to any combinatorial subset of all modalities. However, it is not trivial to directly learn a shared latent representation due to the diversity of combinatorial subsets of all modalities. Furthermore, correlations across modalities as well as global multiscale features are not exploited in this Transformer-related approach. In this work, we propose a Multiscale Multimodal Vison Transformer (MMMViT), which not only leverages correlations across modalities to decouple the direct fusing procedure into two simple steps but also innovatively fuses local multiscale features as the input of the intra-modal Transformer block to implicitly obtain the global multiscale features to adapt to brain tumors of various sizes. We experiment on the BraTs 2018 dataset for all modalities and various missing-modalities as input, and the results demonstrate that the proposed method achieves the state-of-the-art performance.

![image](https://github.com/qiuchengjian/MMMViT/blob/main/fig.png)
## Usage
please refer to [mmFormer](https://github.com/YaoZhang93/mmFormer)


## Reference

* [mmFormer](https://github.com/YaoZhang93/mmFormer)
* [TransBTS](https://github.com/Wenxuan-1119/TransBTS)
* [RFNet](https://github.com/dyh127/RFNet)

