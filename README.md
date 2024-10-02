# Debug Face-TTS Training with temporal Dataset TCD-TIMIT

---
## Setting up the Development Environment

1. Create and acitivate the environment
   ```bash
   conda env create -f environment.yml
   conda activate lab_environment
   ```
2. Build monotonic align module
   ```
   cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
   ```

## Preparation
1. Download trained model weights from <a href="https://drive.google.com/file/d/18ERr-91Z1Mnc2Aq9n1nBPijzb5gSymLq/view?usp=sharing">here</a>
2. Request and Download TCD-TIMIT Data <a href="https://sigmedia.tcd.ie/">here</a>
3. Create Data Structure with TCD-Timit Data
   ```
   python createdata.py
   ```
---
## Test

:exclamation: Face should be cropped and aligned for LRS3 distribution. You can use <a href="https://github.com/joonson/syncnet_python/tree/master/detectors">'syncnet_python/detectors'</a>.

1. Prepare text description in txt file.
```
echo "This is test" > test/text.txt
```


2. Inference Face-TTS.
```
python inference.py
```

3. Result will be saved in `'test/'`. 

:zap: To make MOS test set, we use 'test/ljspeech_text.txt' to randomly select text description.

--- 
## Training

1. Check config.py 

2. Run
```
python run.py
```

---
## Reference
This Laboratory Internship is based on FACE-TTS
<a href="https://arxiv.org/abs/2302.13700"><img src="https://img.shields.io/badge/arXiv-2302.13700-%23B31B1B"></a>
<a href="https://facetts.github.io/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
Thanks!

FACE-TTS is based on 
<a href="https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS">Grad-TTS</a>, 
<a href="https://github.com/bshall/hifigan">HiFi-GAN-16k</a>, 
<a href="https://github.com/joonson/syncnet_trainer">SyncNet</a>. 

---
## Citation

```
@inproceedings{lee2023imaginary,
  author    = {Lee, Jiyoung and Chung, Joon Son and Chung, Soo-Whan},
  title     = {Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year      = {2023},
}
```


