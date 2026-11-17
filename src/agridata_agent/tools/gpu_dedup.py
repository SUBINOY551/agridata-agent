from pathlib import Path
from typing import List
import torch
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from agridata_agent.schemas import DuplicateCluster
from agridata_agent.tools.parsers import IMG_EXTS


class _ImageFolderDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), str(self.paths[idx])


@torch.inference_mode()
def find_near_duplicates_gpu(
    dataset_dir: str,
    similarity_threshold: float = 0.96,
    batch_size: int = 32,
    model_name: str = "vit_small_patch14_dinov2.lvd142m",
    pretrained: bool = True,
) -> List[DuplicateCluster]:
    """Batch-extracts L2-normalized DINOv2 embeddings on GPU (with CPU fallback) to cluster near-duplicate frames."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted([p for p in Path(dataset_dir).rglob("*") if p.suffix.lower() in IMG_EXTS])
    if len(paths) < 2:
        return []

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0).to(device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    loader = DataLoader(_ImageFolderDataset(paths, transform), batch_size=batch_size, num_workers=0)
    all_embeds, all_paths = [], []

    for tensors, batch_paths in loader:
        feats = model(tensors.to(device))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_embeds.append(feats.cpu())
        all_paths.extend(batch_paths)

    embed_matrix = torch.cat(all_embeds, dim=0)
    sim_matrix = embed_matrix @ embed_matrix.T

    visited = set()
    clusters: List[DuplicateCluster] = []

    for i in range(len(all_paths)):
        if i in visited:
            continue
        matches = []
        for j in range(i + 1, len(all_paths)):
            score = float(sim_matrix[i, j])
            if score >= similarity_threshold:
                visited.add(j)
                matches.append((all_paths[j], round(score, 4)))
        if matches:
            visited.add(i)
            clusters.append(DuplicateCluster(anchor_image=all_paths[i], similar_images=matches))

    return clusters
