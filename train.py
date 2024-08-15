import argparse
import os
import wandb
from functools import partial
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from models.Pocket_Design import Pocket_Design
from utils.dataset import PocketLigandPairDataset
from utils.misc import load_config, seed_all
from utils.train import inf_iterator, get_optimizer, get_scheduler
from utils.data import Alphabet, BatchConverter, collate_mols_block
from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_model.yml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="./logs")
    return parser.parse_args()


def setup_data(config, transform, batch_converter):
    train_dataset = PocketLigandPairDataset(
        config["data"]["train_path"], transform=transform
    )
    val_dataset = PocketLigandPairDataset(
        config["data"]["val_path"], transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=partial(
            collate_mols_block,
            batch_converter=batch_converter,
            mask_idx=batch_converter.alphabet.mask_idx,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=partial(collate_mols_block, batch_converter=batch_converter),
    )
    return inf_iterator(train_loader), val_loader


def setup_model(config, protein_featurizer, ligand_featurizer, device):
    model = Pocket_Design(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=device,
    ).to(device)
    total_params = sum(p.nelement() for p in model.parameters())
    print(f"Number of parameters: {total_params / 1e6:.2f}M")
    return model


def train_step(model, batch, optimizer, config, device):
    model.train()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Forward pass
    res_X, ligand_pos, pred_res_type = model(batch)

    # Compute loss
    huber_loss, pred_loss, struct_loss = model.compute_loss(
        res_X, ligand_pos, pred_res_type, batch
    )
    loss = huber_loss + pred_loss + struct_loss

    # Backward pass
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    # Compute metrics
    aar, rmsd = model.compute_metrics(pred_res_type, res_X, batch)

    return loss, huber_loss, pred_loss, struct_loss, aar, rmsd, orig_grad_norm


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_aar = 0
    total_rmsd = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate"):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            # Forward pass
            res_H, res_X, ligand_pos, ligand_feat, pred_res_type, batch = model(batch)

            # Compute loss
            huber_loss, pred_loss, struct_loss = model.compute_loss(
                res_X, ligand_pos, pred_res_type, batch
            )
            loss = huber_loss + pred_loss + struct_loss

            # Compute metrics
            aar, rmsd = model.compute_metrics(pred_res_type, res_X, batch)

            # Accumulate results
            total_loss += loss.item()
            total_aar += aar
            total_rmsd += rmsd
            total_samples += 1

    avg_loss = total_loss / total_samples
    avg_aar = total_aar / total_samples
    avg_rmsd = total_rmsd / total_samples

    return avg_loss, avg_aar, avg_rmsd


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_all(config.train.seed)
    wandb.init(project="pocket generation", config=config)
    ckpt_dir = wandb.run.dir
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([protein_featurizer, ligand_featurizer])
    alphabet = Alphabet.from_architecture("ESM-1b")
    batch_converter = BatchConverter(alphabet)

    train_iterator, val_loader = setup_data(config, transform, batch_converter)
    model = setup_model(config, protein_featurizer, ligand_featurizer, args.device)
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    for it in range(1, config.train.max_iters + 1):
        batch = next(train_iterator)
        loss, huber_loss, pred_loss, struct_loss, aar, rmsd, grad_norm = train_step(
            model, batch, optimizer, config, args.device
        )

        if it % config.train.log_freq == 0:
            wandb.log(
                {
                    "iteration": it,
                    "loss": loss.item(),
                    "huber_loss": huber_loss.item(),
                    "pred_loss": pred_loss.item(),
                    "struct_loss": struct_loss.item(),
                    "aar": aar.item(),
                    "rmsd": rmsd.item(),
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if it % config.train.val_freq == 0 or it == config.train.max_iters:
            val_loss, val_aar, val_rmsd = validate(model, val_loader, args.device)
            wandb.log(
                {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_aar": val_aar,
                    "val_rmsd": val_rmsd,
                }
            )

            if config.train.scheduler.type == "plateau":
                scheduler.step(val_loss)
            elif config.train.scheduler.type == "warmup_plateau":
                scheduler.step_ReduceLROnPlateau(val_loss)
            else:
                scheduler.step()

            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{it}.pt")
            torch.save(
                {
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": it,
                },
                ckpt_path,
            )
            wandb.save(ckpt_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Terminating...")
    finally:
        wandb.finish()
