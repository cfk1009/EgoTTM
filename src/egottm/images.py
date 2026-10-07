"""Load face crops already produced by the upstream preprocessing pipeline."""

import os

import torch
from PIL import Image


def load_face_window(face_crop_root, uid, segment_id, start_frame, end_frame, device, radius=5):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_dir = os.path.join(str(face_crop_root), str(uid), str(segment_id))
    middle = int((int(start_frame) + int(end_frame) + 1) / 2)
    frames = range(middle - radius, middle + radius + 1)
    images, mask = [], []
    for frame_id in frames:
        candidates = []
        if os.path.isdir(image_dir):
            candidates = [name for name in os.listdir(image_dir) if name.startswith(str(frame_id) + "_")]
        if not candidates:
            images.append(torch.zeros(3, 224, 224))
            mask.append(False)
            continue
        image = Image.open(os.path.join(image_dir, sorted(candidates)[0])).convert("RGB")
        images.append(transform(image))
        mask.append(True)
    return torch.stack(images).unsqueeze(0).to(device), torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
