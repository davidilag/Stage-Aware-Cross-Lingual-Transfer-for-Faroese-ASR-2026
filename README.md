# SPEAKABLE 2026
# Stage Aware Cross-Lingual Transfer for Faroese ASR

Files related to a paper on ASR for Fareose for the ICASSP 2026 conference. Continuous pre-training and fine-tuning of Wav2Vec2 models and fine-tuning of Whisper models with Fareose and other Scandinavian data.

FairSeq installation guide is found here: https://github.com/facebookresearch/fairseq/tree/main

FairSeq code base for unsupervised Wav2Vec2 models is found here: https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

The Facebook Wav2Vec2 XLS-R 300M model configuration file is found here: https://huggingface.co/facebook/wav2vec2-xls-r-300m/raw/main/config.json 

Starting CPT is done like so:
python /work/Model/fairseq/fairseq_cli/hydra_train.py \
    --config-dir /work/Model/configuration/FO_1000h \
    --config-name cpt_xlsr_300m

Converstion script is found here: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py

Conversion is done like so:
python convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py \
  --pytorch_dump_folder_path "xlsr2_300m_1000h_FO/checkpoint_best" \
  --checkpoint_path "/1000h-FO/checkpoint_best.pt" \
  --config_path "config.json" \
  --not_finetuned

