bert-spam-detector
====

This repository contains code for the training and deployment of a fine-tuned BERT model to detect spam. This exercise contains many other components considered necessary to put a model like this one in production at scale, and it was made for learning purposes.


## Requirements

ytt
taskfile
python 3.9



## Lessons learned

- CUDA binaries are quite big, so Docker images end up being >6GB in size. Some changes were needed to layer the image properly as to avoid uploading big layers when changing just the code
- If you wanna visualize GPU metrics in Grafana, load this dashboard, import this dashboard id: 14574