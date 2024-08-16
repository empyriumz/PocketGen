import torch
import torch.nn as nn
from .PD import (
    ProteinFeature,
    RES_ATOM_TYPE,
    residue_atom_mask,
    interpolation_init_new,
    sample_from_categorical,
)
from .protein_features import PositionalEncodings
from .esmadapter import ProteinBertModelWithStructuralAdatper
from .esm2adapter import ESM2WithStructuralAdatper
from .encoders import get_encoder


class Pocket_Design(nn.Module):
    def __init__(
        self, config, protein_atom_feature_dim, ligand_atom_feature_dim, device
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_channels = config.hidden_channels

        self._init_embeddings(protein_atom_feature_dim, ligand_atom_feature_dim)
        self._init_encoder()
        self._init_loss_functions()
        self._init_esmadapter()
        self._init_misc()

    def _init_embeddings(self, protein_atom_feature_dim, ligand_atom_feature_dim):
        self.protein_atom_emb = nn.Embedding(
            protein_atom_feature_dim, self.hidden_channels // 2 - 8
        )
        self.pos_embed = PositionalEncodings(16)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, self.hidden_channels)
        self.atom_pos_embedding = nn.Embedding(14, 8)
        self.residue_embedding = nn.Embedding(21, self.hidden_channels // 2 - 16)

    def _init_encoder(self):
        self.encoder = get_encoder(self.config.encoder, self.device)

    def _init_loss_functions(self):
        self.huber_loss = nn.SmoothL1Loss(reduction="mean")
        self.pred_loss = nn.CrossEntropyLoss(reduction="mean")
        self.proteinloss = ProteinFeature()

    def _init_esmadapter(self):
        if self.config.encoder.esm[:4] == "esm2":
            encoder_args = {
                "_target_": "esm2_adapter",
                "encoder": {"d_model": 128, "use_esm_alphabet": True},
                "dropout": 0.1,
                "adapter_layer_indices": [6, 20, 32],
            }
            self.esmadapter = ESM2WithStructuralAdatper.from_pretrained(
                args=encoder_args, name=self.config.encoder.esm
            ).to(self.device)
        else:
            encoder_args = {
                "_target_": "esm_adapter",
                "encoder": {
                    "d_model": 128,
                    "n_enc_layers": 3,
                    "n_dec_layers": 3,
                    "use_esm_alphabet": True,
                },
                "adapter_layer_indices": [6, 20, 32],
            }
            self.esmadapter = ProteinBertModelWithStructuralAdatper.from_pretrained(
                args=encoder_args
            ).to(self.device)

    def _init_misc(self):
        self.standard2alphabet = torch.tensor(
            [1, 6, 13, 9, 19, 12, 5, 2, 17, 8, 0, 11, 16, 14, 10, 4, 7, 18, 15, 3]
        ).to(self.device)
        self.alphabet2standard = torch.tensor(
            [10, 0, 7, 19, 15, 6, 1, 16, 9, 3, 14, 11, 5, 2, 13, 18, 12, 8, 17, 4]
        ).to(self.device)
        self.residue_atom_mask = residue_atom_mask.to(self.device)
        self.res_atom_type = torch.tensor(RES_ATOM_TYPE).to(self.device)

    def init(self, batch):
        residue_mask = batch["small_pocket_residue_mask"]
        label_ligand, pred_ligand = batch["ligand_pos"], batch["ligand_pos"]
        pred_ligand = (
            label_ligand + torch.randn_like(label_ligand).to(self.device) * 0.5
        )

        res_X = self._initialize_res_X(batch, residue_mask)
        res_S = batch["amino_acid_processed"]

        ligand_feat = self.ligand_atom_emb(batch["ligand_feat"])
        res_H = self._compute_res_H(res_S, batch)

        self.full_seq_with_masked_tokens = batch["full_seq_with_masked_tokens"]
        self.full_seq_mask = batch["full_seq_mask"]
        self.large_pocket_mask = batch["large_pocket_mask"]

        return (
            res_H,
            res_X,
            res_S,
            batch["amino_acid_batch"],
            pred_ligand,
            ligand_feat,
            batch["ligand_mask"],
            batch["edit_residue_num"],
            residue_mask,
        )

    def _initialize_res_X(self, batch, residue_mask):
        res_X = batch["residue_pos"]
        res_X = interpolation_init_new(
            res_X, residue_mask, batch["backbone_pos"], batch["amino_acid_batch"]
        )
        for k in range(len(batch["amino_acid"])):
            if residue_mask[k]:
                pos = res_X[k]
                pos[4:] = pos[1].repeat(10, 1) + 0.1 * torch.randn(
                    10, 3, device=self.device
                )
                res_X[k] = pos
        return res_X

    def _compute_res_H(self, res_S, batch):
        atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])
        atom_pos_emb = (
            self.atom_pos_embedding(torch.arange(14).to(self.device))
            .unsqueeze(0)
            .repeat(res_S.shape[0], 1, 1)
        )
        res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)
        res_pos_emb = self.pos_embed(batch["res_idx"]).unsqueeze(-2).repeat(1, 14, 1)
        return torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)

    def forward(self, batch):
        (
            res_H,
            res_X,
            res_S,
            res_batch,
            pred_ligand,
            ligand_feat,
            ligand_mask,
            edit_residue_num,
            residue_mask,
        ) = self.init(batch)

        res_H, res_X, ligand_pos, ligand_feat, pred_res_type = self.encoder(
            res_H,
            res_X,
            res_S,
            res_batch,
            pred_ligand,
            ligand_feat,
            ligand_mask,
            edit_residue_num,
            residue_mask,
        )
        h_residue = res_H.sum(-2)
        batch_size = res_batch.max().item() + 1
        encoder_out = {
            "feats": torch.zeros(
                batch_size,
                self.full_seq_with_masked_tokens.shape[1],
                self.hidden_channels,
            ).to(self.device)
        }
        encoder_out["feats"][self.large_pocket_mask] = h_residue.view(
            -1, self.hidden_channels
        )
        init_pred = self.full_seq_with_masked_tokens
        decode_logits = self.esmadapter(init_pred, encoder_out)["logits"]
        pred_res_type = decode_logits[self.full_seq_mask][:, 4:24]

        return res_X, ligand_pos, pred_res_type

    def compute_loss(self, res_X, ligand_pos, pred_res_type, batch):
        residue_mask = batch["small_pocket_residue_mask"]
        label_ligand = batch["ligand_pos"]
        atom_mask = self.residue_atom_mask[batch["amino_acid"][residue_mask]].bool()
        label_X = batch["residue_pos"]

        huber_loss = self._compute_huber_loss(
            res_X,
            ligand_pos,
            label_X,
            label_ligand,
            residue_mask,
            atom_mask,
            batch["ligand_mask"],
        )
        pred_loss = self._compute_pred_loss(
            pred_res_type, batch["amino_acid"], residue_mask
        )
        struct_loss = self._compute_struct_loss(res_X, label_X, batch, residue_mask)

        return huber_loss, pred_loss, struct_loss

    def _compute_huber_loss(
        self,
        res_X,
        ligand_pos,
        label_X,
        label_ligand,
        residue_mask,
        atom_mask,
        ligand_mask,
    ):
        return self.huber_loss(
            res_X[residue_mask][atom_mask], label_X[residue_mask][atom_mask]
        ) + self.huber_loss(
            ligand_pos[ligand_mask.bool()], label_ligand[ligand_mask.bool()]
        )

    def _compute_pred_loss(self, pred_res_type, amino_acid, residue_mask):
        return self.pred_loss(
            pred_res_type,
            self.standard2alphabet[amino_acid[residue_mask] - 1],
        )

    def _compute_struct_loss(self, res_X, label_X, batch, residue_mask):
        return 2 * self.proteinloss.structure_loss(
            res_X[residue_mask],
            label_X[residue_mask],
            batch["amino_acid"][residue_mask] - 1,
            batch["res_idx"][residue_mask],
            batch["amino_acid_batch"][residue_mask],
        )

    def compute_metrics(self, pred_res_type, res_X, batch):
        residue_mask = batch["small_pocket_residue_mask"]
        label_X = batch["residue_pos"]

        sampled_type, _ = sample_from_categorical(pred_res_type.detach())
        aar = self._compute_aar(sampled_type, batch["amino_acid"], residue_mask)
        rmsd = self._compute_rmsd(res_X, label_X, residue_mask)

        return aar, rmsd

    def _compute_aar(self, sampled_type, amino_acid, residue_mask):
        return (
            self.standard2alphabet[amino_acid[residue_mask] - 1] == sampled_type
        ).sum() / residue_mask.sum()

    def _compute_rmsd(self, res_X, label_X, residue_mask):
        return torch.sqrt(
            (
                (
                    res_X[residue_mask][:, :4].reshape(-1, 3)
                    - label_X[residue_mask][:, :4].reshape(-1, 3)
                )
                .norm(dim=1)
                .sum()
                / residue_mask.sum()
                / 4
            )
        )
