# Progressive Spatio-temporal Perception for Audio-Visual Question Answering (ACMMM'23) [[arXiv](https://arxiv.org/abs/2308.05421)]
PyTorch code accompanies our PSTP-Net.

[Guangyao Li](https://ayameyao.github.io/), [Wenxuan Hou](https://hou9612.github.io/),  [Di Hu](https://dtaoo.github.io/index.html)

---
## Requirements

```python
python3.6 +
pytorch1.6.0
tensorboardX
ffmpeg
numpy
```



## Usage

1. **Clone this repo**

   ```python
   git clone https://github.com/GeWu-Lab/PSTP-Net.git
   ```

2. **Download data**

   MUSIC-AVQA: https://gewu-lab.github.io/MUSIC-AVQA/

   AVQA: http://mn.cs.tsinghua.edu.cn/avqa/

3. **Feature extraction**

   ```python
   feat_script/extract_clip_feat
   python extract_patch-level_feat.py
   ```

4. Training

   ```python
   python main_train.py \
   --temp_select True --segs 12 --top_k 2 \
   --spat_select True --top_m 25 \
   --a_guided_attn True \
   --global_local True \
   --batch-size 64 --epochs 30 --lr 1e-4 --gpu 0 \
   --checkpoint PSTP_Net \
   --model_save_dir models_pstp
   ```

5. Testing

   ```python
   python main_test.py
   ```




## Citation

If you find this work useful, please consider citing it.

```
coming soon!
```



## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.