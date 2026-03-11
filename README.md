# ✨Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy (CVPR2026)

**Nano-EmoX** is first compact (2.2B) emotion intelligence videoLM. It integrates a pretrained LLM with modality-specific encoders and experts-based fusion encoder to handle a broad spectrum of affective tasks in one model. 

- The code will be released.

### Architecture Overview

<img src="nanoemox-arch.png" width="800" />

- **Visual Encoder**: a visual encoder (CLIP-Large) produces frame-level representations, followed by Q-Former, then a projection to the LLM hidden space.
- **Audio Encoder**: an acoustic encoder (HuBERT-Large) is paired with an Q-Former and projected into the LLM hidden space.
- **Facial Encoder**: a facial encoder (FaceXFormer encoder only + temporal modeling) extracts face–aware features and maps them into the LLM space.
- **Fusion Encoder**: It consists of three independent fusion experts and a gating network. Fusion encoder fuses video and audio features before injecting them into the LLM.
- **LLM backbone**: a frozen causal LLM is adapted with lightweight LoRA layers for efficient fine-tuning.
- **Unified prompt injection**: modality tokens are replaced by learned embeddings so that all modalities align in the LLM embedding space.

### Unified Emotion Intelligence

Nano-EmoX supports six core tasks within one model:

1. Multimodal Sentiment Analysis
2. Multimodal Emotion Recognition
3. Open-Vocabulary Multimodal Emotion Recognition
4. Multimodal Intention Recognition
5. Emotion Reason Inference
6. Empathic Response Generation

### P2E Curriculum Learning (Three Phase)

We train Nano-EmoX with a three-stage curriculum that gradually increases emotional intelligence:

<img src="p2e-arch.png" width="800" />

- **Phase 1**: `xemo_phase1.yaml` and `xemo_phase2.yaml` (modality alignment)
- **Phase 2**: `xemo_phase3.yaml` (train fusion encoder)
- **Phase 3**: `xemo_phase4.yaml`

This staged curriculum progressively strengthens the model’s perception, fusion, and reasoning over multimodal affective cues.

### Performance
<img src="visualize.png" width="800" />
<img src="merunibench.png" width="800" />
<img src="avamerg.png" width="600" />
