import torch
import numpy as np
from torch_geometric.data import Batch
from typing import Sequence, Tuple, List
import itertools

# fmt: off
proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}
# fmt: on
"""
adapted from ESM https://github.com/facebookresearch/esm
"""


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ["<eos>", "<unk>", "<pad>", "<cls>", "<mask>"]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self):
        return BatchConverter(self)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        (
                            self._tokenize(token)
                            if token not in self.unique_no_split_tokens
                            else [token]
                        )
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = (
                    self.alphabet.eos_idx
                )

        return labels, strs, tokens


class ProteinLigandData(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(large_pocket_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if large_pocket_dict is not None:
            for key, item in large_pocket_dict.items():
                instance["protein_" + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance["ligand_" + key] = item

        return instance


def batch_from_data_list(data_list):
    return Batch.from_data_list(
        data_list, follow_batch=["ligand_element", "protein_element"]
    )


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def collate_mols_block(mol_dicts, batch_converter, cls_idx=0, mask_idx=32):
    data_batch = {}
    batch_size = len(mol_dicts)

    # Concatenate tensors for common keys
    common_keys = [
        "protein_pos",
        "protein_atom_feature",
        "small_pocket_residue_mask",
        "amino_acid",
        "residue_num_atoms",
        "protein_atom_to_aa_type",
        "res_idx",
        "ligand_element",
        "ligand_bond_type",
    ]
    for key in common_keys:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)

    # Process edge and ligand atom information
    edge_num = torch.tensor(
        [len(mol_dict["ligand_bond_type"]) for mol_dict in mol_dicts]
    )
    ligand_atom_num = torch.tensor(
        [len(mol_dict["ligand_element"]) for mol_dict in mol_dicts]
    )
    data_batch["edge_batch"] = torch.repeat_interleave(
        torch.arange(batch_size), edge_num
    )
    data_batch["ligand_batch"] = torch.repeat_interleave(
        torch.arange(batch_size), ligand_atom_num
    )
    data_batch["ligand_bond_index"] = torch.cat(
        [mol_dict["ligand_bond_index"] for mol_dict in mol_dicts], dim=1
    )

    # Process protein backbone positions
    backbone_keys = ["pos_N", "pos_CA", "pos_C", "pos_O"]
    data_batch["backbone_pos"] = torch.stack(
        [
            torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
            for key in backbone_keys
        ],
        dim=1,
    )

    # Process protein residue information
    num_residues = len(data_batch["amino_acid"])
    data_batch["amino_acid_processed"] = data_batch["amino_acid"].clone()
    offset = 0  # the offset is necessary to keep track of the residue index as we have concatenated the amino acids
    for mol_dict in mol_dicts:
        mask_length = len(mol_dict["small_pocket_residue_mask"])
        data_batch["amino_acid_processed"][offset : offset + mask_length][
            mol_dict["small_pocket_residue_mask"]
        ] = cls_idx
        offset += mask_length
    data_batch["atom2residue"] = torch.repeat_interleave(
        torch.arange(num_residues), data_batch["residue_num_atoms"]
    )
    data_batch["residue_pos"] = torch.zeros(
        num_residues, 14, 3, device=data_batch["amino_acid"].device
    )

    for k in range(num_residues):
        mask = data_batch["atom2residue"] == k
        data_batch["residue_pos"][k][
            : min(data_batch["residue_num_atoms"][k].item(), 14)
        ] = data_batch["protein_pos"][mask][
            : min(data_batch["residue_num_atoms"][k].item(), 14)
        ]

    # Process batch information for residues
    repeats = torch.tensor([len(mol_dict["amino_acid"]) for mol_dict in mol_dicts])
    data_batch["amino_acid_batch"] = torch.repeat_interleave(
        torch.arange(batch_size), repeats
    )

    # Process ligand information
    data_batch["ligand_num_atoms"] = torch.tensor(
        [len(mol_dict["ligand_pos"]) for mol_dict in mol_dicts]
    )
    max_ligand_atoms = max(data_batch["ligand_num_atoms"])

    ligand_tensors = ["ligand_pos", "ligand_feat"]
    for key in ligand_tensors:
        data_batch[key] = torch.zeros(
            batch_size,
            max_ligand_atoms,
            mol_dicts[0][
                "ligand_pos" if key == "ligand_pos" else "ligand_atom_feature"
            ].shape[-1],
            device=data_batch["amino_acid"].device,
        )
    data_batch["ligand_mask"] = torch.zeros(
        batch_size, max_ligand_atoms, device=data_batch["amino_acid"].device
    )

    for b in range(batch_size):
        n_atoms = data_batch["ligand_num_atoms"][b]
        data_batch["ligand_pos"][b, :n_atoms] = mol_dicts[b]["ligand_pos"]
        data_batch["ligand_feat"][b, :n_atoms] = mol_dicts[b]["ligand_atom_feature"]
        data_batch["ligand_mask"][b, :n_atoms] = 1

    # Process edit residue information
    data_batch["edit_residue_num"] = torch.tensor(
        [mol_dict["small_pocket_residue_mask"].sum() for mol_dict in mol_dicts],
        device=data_batch["amino_acid"].device,
    )

    # Process sequence information
    data_batch["full_seq"] = [("", mol_dict["full_seq"]) for mol_dict in mol_dicts]
    _, _, data_batch["full_seq"] = batch_converter(data_batch["full_seq"])

    data_batch["full_seq_mask"] = torch.zeros_like(data_batch["full_seq"]).bool()
    data_batch["large_pocket_mask"] = torch.zeros_like(data_batch["full_seq"]).bool()

    for b in range(batch_size):
        data_batch["full_seq"][b][
            mol_dicts[b]["small_pocket_global_idx"] + 1
        ] = mask_idx
        data_batch["full_seq_mask"][b][
            mol_dicts[b]["small_pocket_global_idx"] + 1
        ] = True
        data_batch["large_pocket_mask"][b][mol_dicts[b]["large_pocket_idx"] + 1] = True

    data_batch["full_seq_with_masked_tokens"] = data_batch["full_seq"].clone()
    data_batch.pop("full_seq")
    # Add filename information
    data_batch["protein_filename"] = [
        mol_dict["protein_filename"] for mol_dict in mol_dicts
    ]
    data_batch["ligand_filename"] = [
        mol_dict["ligand_filename"] for mol_dict in mol_dicts
    ]

    return data_batch
