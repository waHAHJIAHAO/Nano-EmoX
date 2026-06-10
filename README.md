<div align="center">

<h2>[CVPR 2026] Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy</h2>

<a href="https://arxiv.org/pdf/2603.02123">
<img src='https://img.shields.io/badge/Paper-Arxiv-orange' alt='Paper PDF'></a>
<a href="https://huggingface.co/datasets/">
<img src='https://img.shields.io/badge/Dataset-HuggingFace-yellow' alt='Dataset'></a>
<a href="#">
<img src='https://img.shields.io/badge/Model-ModelScope-blue' alt='Model'></a>

</div>

## Todo List

- [x] release paper
- [x] release project codes
- [ ] training and evaluation scripts
- [ ] model weights

## Unified Emotion Intelligence

our model is first compact (2.2B) emotion intelligence videoLM. It integrates a pretrained LLM with modality-specific encoders and experts-based fusion encoder to handle a broad spectrum of affective tasks in one model.

Nano-EmoX supports six core tasks within one model: 1), Multimodal Sentiment Analysis. 2), Multimodal Emotion Recognition. 3), Open-Vocabulary Multimodal Emotion Recognition. 4), Multimodal Intention Recognition. 5), Emotion Reason Inference. 6), Empathic Response Generation

<img src="./assets/nanoemox-arch.png" width="800" />

- **Visual Encoder**: a visual encoder (CLIP-Large) produces frame-level representations, followed by Q-Former, then a projection to the LLM hidden space.
- **Audio Encoder**: an acoustic encoder (HuBERT-Large) is paired with an Q-Former and projected into the LLM hidden space.
- **Facial Encoder**: a facial encoder (FaceXFormer encoder only + temporal modeling) extracts face–aware features and maps them into the LLM space.
- **Fusion Encoder**: It consists of three independent fusion experts and a gating network. Fusion encoder fuses video and audio features before injecting them into the LLM.
- **LLM backbone**: a frozen causal samll scale LM (Qwen-2.5-1.5B) is adapted with lightweight LoRA layers for efficient fine-tuning.
- **Unified prompt injection**: modality tokens are replaced by learned embeddings so that all modalities align in the LLM embedding space.

## P2E training framework (Three Phase)

We train Nano-EmoX with a three-phase curriculum that gradually increases emotional intelligence:

<img src="./assets/p2e-arch.png" width="800" />

This staged curriculum progressively strengthens the model’s perception, fusion, and reasoning over multimodal affective cues.

## Performance

<img src="./assets/visualize.png" width="800" />
<img src="./assets/merunibench.png" width="800" />
<img src="./assets/avamerg.png" width="600" />

## Quick Start

## Datasets

- **For training**
- **For evaluation**
  
## Training
- **Phase 1**: config file: `xemo_phase1.yaml` and `xemo_phase2.yaml` (modality alignment)
- **Phase 2**: config file: `xemo_phase3.yaml` (train fusion encoder)
- **Phase 3**: config file: `xemo_phase4.yaml`

## Evaluation


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If this work has been helpful or inspiring to your research, please consider cite our article:

```bibtex
@InProceedings{Huang_2026_CVPR,
    author    = {Huang, Jiahao and Lin, Fengyan and Yang, Xuechao and Feng, Chen and Zhu, Kexin and Yang, Xu and Chen, Zhide},
    title     = {Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {22986-22997}
}

