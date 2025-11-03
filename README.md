<!-- # GUI-Actor -->
## GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding
[Shijie Zhou](https://scholar.google.com/citations?user=eEjr-isAAAAJ&hl=en)<sup>*1</sup>&nbsp;
[Viet Dac Lai](https://laiviet.github.io/)<sup>2</sup>&nbsp;
[Hao Tan](http://www.cs.unc.edu/~airsplay/)<sup>2</sup>&nbsp;
[Jihyung Kil](https://heendung.github.io/)<sup>2</sup>&nbsp;
[Wanrong Zhu](https://wanrong-zhu.com/)<sup>2</sup>&nbsp;<br>
[Changyou Chen](https://cse.buffalo.edu/~changyou/)<sup>1</sup>
[Ruiyi Zhang](https://zhangry868.github.io/)<sup>2</sup><sup>†</sup> 

<sup>1</sup> University at Buffalo&nbsp;&nbsp;<sup>2</sup> Adobe Research<br>
<sup>*</sup> Majority work done while SZ is at University at Buffalo&nbsp;&nbsp;<sup>†</sup> Leadership  

<h4>
<a href="">📄 arXiv Paper</a> &nbsp;<a href="https://huggingface.co/smz8599/GUI-AIMA-3B">🤗 GUI-AIMA-3B</a>&nbsp;
<a href="https://huggingface.co/smz8599/GUI-AIMA-3B-kl">🤗 GUI-AIMA-3B (soft)</a> &nbsp;
</h4>

</div>
<div align="center">
<img src="assets/images/comparison.png?raw=true" width="85%">
</div>

Figure 1. **GUI-AIMA** utilize the inherent attention of MLLMs for patch-wise GUI grounding. It simplifies the vanilla attention grounding requiring proper aggregation between all query tokens' grounding vectors by adding a learnable ANCHOR token as the context anchor of query. The multi-head aggregation on attention vectors between ANCHOR and visual tokens is adequate for grounding.

<div align="center">
<img src="assets/images/main_fig.png?raw=true" width="85%">
</div>

Figure 2. **GUI-AIMA** proposes an effective multi-head weighting approach by measuring the uniformity between global query-visual pattern and head-wise query-visual pattern. 

## Table of Contents
- [Main Results](#main-results)
- [Installation](#installation)
- [Model Training](#model-training)
  - [Data preparation](#data-preparation)
  - [Training](#training)
- [Evaluation on GUI Grounding Benchmarks](#evaluation-on-gui-grounding-benchmarks)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Main Results
There are two variants of GUI-AIMA: [GUI-AIMA-3B](https://huggingface.co/smz8599/GUI-AIMA-3B) and [GUI-AIMA-3B(soft)](https://huggingface.co/smz8599/GUI-AIMA-3B-kl) with slight differences of multihead weighting.

GUI-AIMA achieves **47.1%** and **56.9%** on ScreenSpot-pro and OSWorld-G. With 2-step zoom-in inference, it can achieve **58.6%** and **62.2%** on ScreenSpot-pro and OSWorld-G.

We trained GUI-AIMA for one-step center points predictions. However, **GUI-AIMA can be inferenced in the 2-step fashion without further fine-tuning**: (step 1) 1st inferece to determine rough grounding areas; (step 2) crop and zoom-in the rough grounding areas for 2nd preciser grounding inference.  The 2-step inference is very helpful for GUI grounding on high-resolution screenshots, such as samples in ScreenSpot-pro and OSWorld-G.

</div>
<div align="left">
<img src="assets/images/ss_pro.png?raw=true" width="100%">
</div>

<div align="left">
<img src="assets/images/osworld-g.png?raw=true" width="80%">
</div>

<div align="left">
<img src="assets/images/ss_v2.png?raw=true" width="85%">
</div>

## Installation
1. Environment:
```bash
git clone https://github.com/sjz5202/GUI-AIMA
cd GUI-AIMA
conda create -n gui_aima python=3.10
conda activate gui_aima
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```
## Model Training
### Data preparation
1. Download the GUI-Actor data from [here](https://huggingface.co/datasets/cckevinn/GUI-Actor-Data).
2. Download the UGround single-round dialogue json data from [here](https://huggingface.co/datasets/smz8599/UGround-single).
2. Download the GTA1 data without the web part from [here](https://huggingface.co/datasets/smz8599/GTA_data_no_web).

### Training
1. Single-node training:
```bash
bash scripts/sft_single_node.sh
```
2. Multi-node training (for reference, need adjusted for your environment):
```bash
bash scripts/sft_multi_node.sh
```

## Evaluation on GUI Grounding Benchmarks
We provide evaluation scripts on ScreenSpot-Pro, ScreenSpot-v2 and OSWorld-G under the `eval/` folder: `eval_ss_pro.sh`, `eval_ss_v2.sh`, `eval_osworld_g.sh`. 

For ScreenSpot-Pro and OSWorld-G, we provide 2-step inference in `eval_ss_pro.sh` and `eval_osworld_g.sh`, which determines the focusing area at the 1st step and zoom-in the focusing area for grounding at the 2nd step without extra model training.

For ScreenSpot-Pro and OSWorld-G, you need to download the data from [here](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) and [here](https://github.com/xlang-ai/OSWorld-G), then adjust the data path in `eval_ss_pro.sh` and `eval_osworld_g.sh`.

Single sample example usage is available in `eval/example_inference.py`.


## Acknowledgements

GUI-AIMA is built upon the following projects.
- [GUI-Actor](https://github.com/microsoft/GUI-Actor)
- [TAG](https://github.com/HeimingX/TAG.git)
- [GTA1](https://github.com/Yan98/GTA1)
- [Transformers](https://github.com/huggingface/transformers)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [AGUVIS](https://github.com/xlang-ai/aguvis)
- [OS-Atlas](https://github.com/OS-Copilot/OS-Atlas)
- [SeeClick](https://github.com/njucckevin/SeeClick)

Thanks for their great work!

## Citation
```bibtex
```
