ML Playground
====

This is my personal project focused on learning MLOps concepts and infrastructure, as well as the operational aspects of deploying and serving large ML models in production. My goal was to create a platform that prioritizes scalability and applies good practices found in other non-ML projects.

In addition, I wanted to explore LLMs and general DL algorithms and provide a quick way to train and deploy models to iterate faster. The project incorporates several technologies, including:

* Kubernetes (microk8s for now) and Nvidia GPU operator for GPU resource access
* PyTorch/transformers as the main ML framework
* Torchserve for model inference
* Terraform for spinning up Lambda Cloud clusters
* Prometheus, Grafana, Loki for metrics and general observability

I have also included an example model, a BERT model with a spam detector fine-tuning, and plan to add more in the future. It's worth noting that this list of technologies will grow as the project evolves.

While my main objective is not to generate my own DL algorithms or training/inference code, I plan to compile from other sources and keep the project up-to-date with the latest LLM models trained with a smaller budget. This project is a personal endeavor, and my primary aim is to learn and improve my knowledge of MLOps and machine learning.

## Requirements

* ytt for kubernetes yaml quick-and-dirty templating
* Taskfile to launch different tasks
* Python 3.8
* Terraform to manage the infrastructure
* kubectl
* Docker

## Getting started

TODO

## Lessons learned (for now)

- CUDA binaries are quite big, so Docker images end up being >6GB in size. Some changes were needed to layer the image properly as to avoid uploading big layers when changing just the training / inference code
- If you wanna visualize GPU metrics in Grafana, load this dashboard, import this dashboard id: 14574