## BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

## Announcement: BLIP is now officially integrated into [LAVIS](https://github.com/salesforce/LAVIS) - a one-stop library for language-and-vision research and applications!

<img src="BLIP.gif" width="700">

This is the PyTorch code of the <a href="https://arxiv.org/abs/2201.12086">BLIP paper</a> [[blog](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)]. The code has been tested on PyTorch 1.10.
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

Catalog:
- [x] Inference demo
- [x] Pre-trained and finetuned checkpoints
- [x] Finetuning code for Image-Text Retrieval, Image Captioning, VQA, and NLVR2
- [x] Pre-training code
- [x] Zero-shot video-text retrieval
- [x] Download of bootstrapped pre-training datasets 


### Inference demo:
Run our interactive demo using [Colab notebook](https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb) (no GPU needed).
The demo includes code for: 
1. Image captioning
2. Open-ended visual question answering
3. Multimodal / unimodal feature extraction
4. Image-text matching

Try out the [Web demo](https://huggingface.co/spaces/Salesforce/BLIP), integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). 

Replicate web demo and Docker image is also available at [![Replicate](https://replicate.com/salesforce/blip/badge)](https://replicate.com/salesforce/blip)

### Pre-trained checkpoints:
Num. pre-train images | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---: 
14M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth">Download</a>| - | -
129M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth">Download</a> | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth">Download</a>

### Finetuned checkpoints:
Task | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---:
Image-Text Retrieval (COCO) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth">Download</a>| - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth">Download</a>
Image-Text Retrieval (Flickr30k) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth">Download</a>|  - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth">Download</a>
Image Captioning (COCO) | - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a> | 
VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth">Download</a> | - 
NLVR2 | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth">Download</a>| - | - 


### Image-Text Retrieval:
1. Download COCO and Flickr30k datasets from the original websites, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.
2. To evaluate the finetuned BLIP model on COCO, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco \
--evaluate</pre> 
3. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco </pre> 

### Image-Text Captioning:
1. Download COCO and NoCaps datasets from the original websites, and set 'image_root' in configs/caption_coco.yaml and configs/nocaps.yaml accordingly.
2. To evaluate the finetuned BLIP model on COCO, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py --evaluate</pre> 
3. To evaluate the finetuned BLIP model on NoCaps, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_nocaps.py </pre> 
4. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/caption_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py \
--config ./configs/caption_coco.yaml \
--output_dir output/caption_coco </pre> 

### VQA:
1. Download COCO and VQA datasets from the original websites, and set 'image_root' in configs/vqa.yaml accordingly.
2. To evaluate the finetuned BLIP model on VQA, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_vqa.py \
--config ./configs/vqa.yaml \
--output_dir output/vqa \
--evaluate</pre> 
3. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/vqa.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py \
--config ./configs/vqa.yaml \
--output_dir output/vqa </pre> 

### NLVR2:
1. Download NLVR2 datasets from the original websites, and set 'image_root' in configs/nlvr2.yaml accordingly.
2. To evaluate the finetuned BLIP model on NLVR2, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_nlvr2.py \
--config ./configs/nlvr2.yaml \
--output_dir output/nlvr2 \
--evaluate</pre> 
3. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/nlvr2.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_nlvr2.py \
--config ./configs/nlvr2.yaml \
--output_dir output/nlvr2 </pre> 

### Pre-training:
1. Download the 14M or 129M pre-training dataset from [here](https://github.com/salesforce/BLIP#download-of-bootstrapped-pre-training-datasets).
2. To pre-train BLIP from scratch using 8 A100 GPUs, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_pretrain.py \
--config ./configs/pretrain.yaml \
--output_dir output/pretrain </pre> 

### Download of bootstrapped pre-training datasets:
We release the bootstrapped pre-training datasets (image-text pairs) used in the paper. 
- [14M dataset (clean)](https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/14M_clean.tar.gz): 14M image-text pairs from COCO, VisualGenome, and Conceptual Captions, with noise filtering applied.
- [14M dataset (noisy)](https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/14M_noisy.tar.gz): 14M image-text pairs from COCO, VisualGenome, and Conceptual Captions, without noise filtering.
- [129M dataset](https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/129M.tar.gz): 129M image-text pairs from COCO, VisualGenome, Conceptual Captions, SBU, and LAION-en, with noise filtering applied.

Note: The datasets are provided for research purposes only. Please refer to the original datasets for license terms.

### Zero-shot video-text retrieval:
We provide a script to perform zero-shot video-text retrieval using BLIP. Run the following command:
<pre>python demo_video.py --video_path your_video.mp4 --text "a description of the video"</pre>

### Citation:
If you find this code useful for your research, please cite our paper:
<pre>
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven C. H.},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
</pre>

### License:
This project is released under the BSD 3-Clause License.

### Related project: [LAVIS](https://github.com/salesforce/LAVIS)