import os, json, numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from datasets.ADNI import ADNI_transform
from models.unet3d import UNet3DClassifier
from torch.multiprocessing import freeze_support

def load_cfg(path):
    with open(path) as f: 
        return json.load(f)

class Cfg:
    def __init__(self, d):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for k, v in d.items(): 
            setattr(self, k, v)

def generate_model(cfg):
    model = UNet3DClassifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.nb_class
    ).to(cfg.device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param  = 2 if getattr(cfg, 'fp16', False) else 4
    print("--------------------model------------------")
    print(f"Total params(M)    : {total_params:,}")
    print(f"Trainable params(M): {trainable_params:,}")
    print(f"Approx. size       : {total_params*bytes_per_param/1024**2:.2f} MB")
    print("model type:", type(model).__name__)

    return model


def infer_single_image(file_path, checkpoint_path=None):
    device = cfg.device
    _, test_tf = ADNI_transform(augment=False)
    ds = Dataset(data=[{'MRI': file_path}], transform=test_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = generate_model(cfg)
    ckpt = checkpoint_path or os.path.join(cfg.checkpoint_dir, 'best_model_fold1.pth')
    try:
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    with torch.no_grad():
        batch = next(iter(loader))
        x = batch['MRI'].to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
    pred_idx = int(np.argmax(probs))
    if getattr(cfg, 'nb_class', 3) == 3:
        classes = ['CN','MCI','AD']
    elif getattr(cfg, 'nb_class', 2) == 2:
        classes = ['CN','AD']
    else:
        classes = [str(i) for i in range(getattr(cfg, 'nb_class', len(probs)))]
    pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
    return {'proba': probs, 'pred': pred_idx, 'pred_label': pred_label, 'classes': classes}

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    config_path = r"C:\Users\dongzj\Desktop\mmad\unet3d_single\config\config.json"
    cfg = Cfg(load_cfg(config_path))

    import sys
    img_path = os.environ.get('WEB_IMAGE_PATH') or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not img_path:
        print(json.dumps({"error": "no image path provided"}, ensure_ascii=False))
        raise SystemExit(1)
    res = infer_single_image(img_path)
    print(json.dumps(res, ensure_ascii=False))
