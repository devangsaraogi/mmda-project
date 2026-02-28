from abc import ABC, abstractmethod

from PIL import Image
from torch.utils.data import Dataset


class BaseClaimDataset(Dataset, ABC):
    """Abstract base class for claim-evidence datasets.

    Each item is a dict with:
        claim_id: str          - unique claim identifier
        claim_text: str        - the textual claim
        gold_image_ids: list   - IDs of gold-standard evidence images
        label: int             - verdict (0=True, 1=False, 2=Unverifiable)
        misinfo_type: str      - misinformation type category
    """

    LABEL_NAMES = {0: "True", 1: "False", 2: "Unverifiable"}
    MISINFO_TYPES = [
        "true",
        "text_contradiction",
        "visual_contradiction",
        "temporal_mismatch",
        "ambiguous",
    ]

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        ...

    @abstractmethod
    def get_image(self, image_id: str) -> Image.Image:
        """Return a PIL Image for the given image ID."""
        ...

    @abstractmethod
    def get_all_image_ids(self) -> list[str]:
        """Return all image IDs in the pool."""
        ...

    @abstractmethod
    def get_split(self, split: str) -> "BaseClaimDataset":
        """Return a subset for the given split ('train', 'val', 'test')."""
        ...

    def label_name(self, label: int) -> str:
        return self.LABEL_NAMES[label]
