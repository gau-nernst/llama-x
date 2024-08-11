# Llama-X

Exploration work on fine-tuning Llama3.1 for multi-modal understanding.

I believe transformers are universal machines. In a past experiment, I found that a BERT-initialized transformer can be used as an effective fusion module for multi-modal models. Motivated by this observation, I aim to explore fine-tuning Llama3.1 directly to understand other modalities without modal-specific encoders. Architecture-wise, this is also known as early-fusion.

Inspirations:
- Fuyu: [[blog](https://www.adept.ai/blog/fuyu-8b)]
- PaliGemma: [[paper](https://arxiv.org/abs/2407.07726)]
- Chameleon: [[paper](https://arxiv.org/abs/2405.09818)]
- Qwen-Audio: [[paper v1](https://arxiv.org/abs/2311.07919)] [[paper v2](https://arxiv.org/abs/2407.10759)]

Plan:
- Image understanding: use PatchEmbed -> ViT style.
- Speech understanding: use mel-spectrogram + 2 Conv1D layers -> Whisper Encoder style. An alternative is Wav2Vec2 style (originally called feature encoder).
- Attention: Prefix-LM, leveraging the newly released [FlexAttention](https://pytorch.org/blog/flexattention/).
