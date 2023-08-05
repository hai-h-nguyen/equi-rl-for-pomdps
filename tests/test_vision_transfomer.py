from torchkit.vit import ViT
import torch


v = ViT(
    image_size=128,
    patch_size=8,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    channels=2
)

img = torch.randn(1, 2, 128, 128)

preds = v(img)
assert preds.shape == (1, 1000), 'correct logits outputted'