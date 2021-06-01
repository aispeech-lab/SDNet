# ICASSP 2021: SDNet:Speaker and Direction Inferred Dual-channel Speech Separation

If you have the interest in our work, or use this code or part of it, please cite us!  
Consider citing:
```bash
@inproceedings{li2021speaker,
  title={Speaker and Direction Inferred Dual-Channel Speech Separation},
  author={Li, Chenxing and Xu, Jiaming and Mesgarani, Nima and Xu, Bo},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5779--5783},
  year={2021},
  organization={IEEE}
}
```
For more detailed descirption, you can further explore the whole paper with [this link](https://doi.org/10.1109/ICASSP39728.2021.9413818).  

# Requirements:
Pytorch>=1.1.0 <br>
resampy <br>
soundfile <br>

# Model Descriptions:
![](https://github.com/aispeech-lab/SDNet/blob/main/jpg/sdnet.jpeg)  



# Data Preparation

Please refer to predata_WSJ_lcx.py
A more detailed dataset preparation procedure will be updated soon.

# Train and Test

For train: <br>
python train_WSJ0_SDNet.py <br>

For test: <br>
python test_WSJ0_SDNet.py <br>

Please Modify the model path in test_WSJ0_SDNet.py.

# Contact
If you have any questions please contact: <br>
Email:lichenxing007@gmail.com

# TODO
1. A brief implemention of SDNet
2. pretrained models.
3. separated samples.




