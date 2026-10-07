"""Checkpoint-compatible text/context/fusion model for the inference demo."""

from pathlib import Path

import torch
import torch.nn as nn


def _checkpoint_state(path):
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    return state.get("model_state_dict", state)


def filter_ttm_checkpoint_state(state, separate_encoder=False):
    """Match the legacy loader when RoBERTa is supplied separately."""

    if not separate_encoder:
        return state
    return {key: value for key, value in state.items() if not key.startswith("encoder.roberta.")}


def load_separate_roberta(encoder, checkpoint_path):
    state = _checkpoint_state(checkpoint_path)
    roberta_state = {}
    for key, value in state.items():
        if key.startswith("encoder.roberta."):
            roberta_state[key[len("encoder.roberta.") :]] = value
        elif key.startswith("roberta."):
            roberta_state[key[len("roberta.") :]] = value
        elif key.startswith(("embeddings.", "encoder.", "pooler.")):
            roberta_state[key] = value
    missing, unexpected = encoder.load_state_dict(roberta_state, strict=False)
    if missing:
        print("[warning] missing separate RoBERTa keys:", len(missing))
    if unexpected:
        print("[warning] unexpected separate RoBERTa keys:", len(unexpected))


class SpeakerAwareTextEncoder(nn.Module):
    def __init__(
        self,
        tokenizer_name,
        hidden_dim=1024,
        speaker_count=30,
        max_rounds=500,
        initialize_from_pretrained=True,
    ):
        super().__init__()
        from transformers import RobertaModel

        if initialize_from_pretrained:
            self.roberta = RobertaModel.from_pretrained(tokenizer_name)
        else:
            from transformers import RobertaConfig

            self.roberta = RobertaModel(RobertaConfig.from_pretrained(tokenizer_name))
        self.hidden_dim = self.roberta.config.hidden_size
        if self.hidden_dim != hidden_dim:
            hidden_dim = self.hidden_dim
        self.speaker_emb = nn.Embedding(speaker_count, 32, padding_idx=29)
        self.pos_emb = nn.Embedding(max_rounds, 32, padding_idx=max_rounds - 1)
        self.proj = nn.Linear(hidden_dim + 64, hidden_dim)

    def forward(self, input_ids, attention_mask, speaker_ids, round_ids):
        batch, utterances, tokens = input_ids.shape
        outputs = self.roberta(
            input_ids.reshape(batch * utterances, tokens),
            attention_mask=attention_mask.reshape(batch * utterances, tokens),
        )
        text = outputs.last_hidden_state[:, 0, :].reshape(batch, utterances, -1)
        speaker = self.speaker_emb(speaker_ids)
        rounds = self.pos_emb(round_ids)
        return self.proj(torch.cat([text, speaker, rounds], dim=-1))


class UtteranceLevelTransformer(nn.Module):
    """Keep the legacy ``utt_transformer.transformer.*`` checkpoint namespace."""

    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask=None):
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class TTMDialogModel(nn.Module):
    """The paper's SSE -> CCM -> GVE -> fusion path.

    ``forward_from_memory`` is the inference path: it starts from cached SSE
    embeddings and therefore skips the frozen PLM and projection computation.
    """

    def __init__(
        self,
        tokenizer_name,
        hidden_dim=1024,
        lam_dim=256,
        max_rounds=500,
        initialize_from_pretrained=True,
    ):
        super().__init__()
        self.encoder = SpeakerAwareTextEncoder(
            tokenizer_name,
            hidden_dim,
            max_rounds=max_rounds,
            initialize_from_pretrained=initialize_from_pretrained,
        )
        hidden_dim = self.encoder.hidden_dim
        self.utt_transformer = UtteranceLevelTransformer(hidden_dim, num_layers=2)
        self.lam_proj = nn.Linear(lam_dim, hidden_dim)
        self.mm_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.fusion_proj = nn.Linear(hidden_dim + lam_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),
        )

    @staticmethod
    def masked_average(features, mask):
        weights = mask.to(features.dtype).unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def forward_from_memory(self, sse_embeddings, lengths, lam_features, lam_mask):
        max_length = sse_embeddings.size(1)
        if torch.all(lengths == max_length):
            # Memory-bank inference supplies an exact, unpadded context window.
            # Keep this path identical to the legacy model, which did not pass a
            # padding mask to the utterance Transformer.
            context = self.utt_transformer(sse_embeddings)
        else:
            positions = torch.arange(max_length, device=sse_embeddings.device).unsqueeze(0)
            padding_mask = positions >= lengths.unsqueeze(1)
            context = self.utt_transformer(sse_embeddings, src_key_padding_mask=padding_mask)
        batch_indices = torch.arange(context.size(0), device=context.device)
        text_feature = context[batch_indices, lengths - 1]
        visual_feature = self.lam_proj(self.masked_average(lam_features, lam_mask))
        tokens = torch.stack([text_feature, visual_feature], dim=1)
        attended, _ = self.mm_attn(tokens, tokens, tokens)
        fused = text_feature + attended[:, 0, :]
        return self.mlp(fused)

    def forward(self, input_ids, attention_mask, speaker_ids, round_ids, lam_features, lengths, lam_mask):
        embeddings = self.encoder(input_ids, attention_mask, speaker_ids, round_ids)
        return self.forward_from_memory(embeddings, lengths, lam_features, lam_mask)


def load_ttm_model(tokenizer_name, checkpoint_path, device, encoder_checkpoint=None):
    model = TTMDialogModel(tokenizer_name, initialize_from_pretrained=encoder_checkpoint is None)
    if encoder_checkpoint:
        load_separate_roberta(model.encoder.roberta, encoder_checkpoint)
    state = _checkpoint_state(Path(checkpoint_path))
    state = filter_ttm_checkpoint_state(state, separate_encoder=bool(encoder_checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[warning] missing checkpoint keys:", len(missing))
    if unexpected:
        print("[warning] unexpected checkpoint keys:", len(unexpected))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
