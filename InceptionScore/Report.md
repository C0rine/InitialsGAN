## Inception Score

Code based on: 
* https://github.com/gstrezoski/initials_baselines
* https://github.com/sbarratt/inception-score-pytorch

Inception score calculator taken from: 
* https://github.com/openai/improved-gan/blob/master/inception_score/model.py

Which is the code corresponding to the improved GAN paper: http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans

----------

50.000 images were generated using random noise which was inputted to the DCGAN from the 04_GAN-adjustable folder. There were splitted into 10 splits (as suggested in  http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans)

Results for DCGAN:
* Mean IS: 9.99962
* Std deviation: 0.14551

These results in themselves don't tell us much yet as they cannot (yet) be compared to other inception scores. The comparison would be invalid as the scores you see in academic papers are based on the probabilities taken from a completely different classifier. 
