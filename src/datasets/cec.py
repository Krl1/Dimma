from pathlib import Path
from typing import Callable, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from src.datasets.meta import PairedImageWithLightnessInput
from src.transforms import load_transforms
from src.utils.image import read_image_cv2

class CEC(Dataset):
    def __init__(
        self, 
        root: Path, 
        transform: Callable, 
        split: str = "train", 
        unsupervised: bool = False,
        limit: Optional[int] = None
    ):
        path = root / split
        self.transform = transform
        self.unsupervised = unsupervised
        
        # W trybie unsupervised interesują nas tylko obrazy 'clean'
        self.target_names = sorted((path / "clean/").glob("*.png"), key=lambda x: int(x.stem))
        
        if not self.unsupervised:
            self.image_names = sorted((path / "corrupted/").glob("*.png"), key=lambda x: int(x.stem))

        if limit is not None:
            self.target_names = self.target_names[:limit]
            if not self.unsupervised:
                self.image_names = self.image_names[:limit]

    def __len__(self):
        return len(self.target_names)

    def __getitem__(self, index: int) -> PairedImageWithLightnessInput:
        if self.unsupervised:
            # Etap 1: Bierzemy jasny obraz i generujemy ciemny transformacją MDN
            target = read_image_cv2(self.target_names[index])
            transformed = self.transform(light=target)
        else:
            # Etap 2: Bierzemy rzeczywistą parę
            image = read_image_cv2(self.image_names[index])
            target = read_image_cv2(self.target_names[index])
            transformed = self.transform(image=image, target=target)
            
        return PairedImageWithLightnessInput(
            image=transformed["image"],
            target=transformed["target"],
            source_lightness=transformed["source_lightness"],
            target_lightness=transformed["target_lightness"],
        )

class CECDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.root = Path(config.path)
        self.config = config
        self.train_transform, self.test_transform = load_transforms(config.transform)

    def setup(self, stage: Optional[str] = None):
        unsupervised = self.config.get("unsupervised", False)
        self.train_ds = CEC(
            self.root, 
            split='train', 
            transform=self.train_transform, 
            unsupervised=unsupervised,
            limit=self.config.get('limit')
        )
        self.val_ds = CEC(
            self.root, 
            split='test', 
            transform=self.test_transform,
            unsupervised=False # Walidacja zawsze na rzeczywistych spinach
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.config.num_workers)
