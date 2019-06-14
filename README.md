# SSL with TensorFlow
Implementation of "Semi-Supervised Learning with Deep Generative Models"
<br>
<br>

### References
D. P. Kingma, D. J Rezende, S. Mohamed, M. Welling 
"Semi-Supervised Learning with Deep Generative Models," 
Advances in Neural Information Processing Systems 27(NIPS 2014). 
[PDF](https://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf)

### Environment
- CPU: Intel(R) Core(TM) i7-8700K @ 3.70GHz
- GPU: NVIDIA GeForce GTX 760
- TensorFlow 1.10.0

### Result
M1-Loss=94.321: 100%|██████████| 100/100 [03:32<00:00,  2.12s/it] <br>
M2-Loss=2706.5: 100%|██████████| 1000/1000 [48:20<00:00,  2.92s/it] <br>
100%|██████████| 100/100 [00:00<00:00, 550.92it/s] <br>
Test Accuracy:  0.9104000002145767 <br>

### Result v2
M1 loss=92.4554: 100%|██████████| 500/500 [19:04<00:00,  2.26s/it]
M2 loss=-217453, val_acc=0.864309: 100%|██████████| 500/500 [30:01<00:00,  3.62s/it] 
100%|██████████| 100/100 [00:00<00:00, 642.74it/s]
Test Accuracy:  0.8694999974966049

![M1 model](https://github.com/DonggeunKwon/ssl-2014/blob/master/img/Figure_1.png)
![M2 model](https://github.com/DonggeunKwon/ssl-2014/blob/master/img/Figure_2.png)
