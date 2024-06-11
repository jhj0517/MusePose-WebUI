import os
import wget
from tqdm import tqdm


def download_models(
    model_dir: str = os.makedirs('pretrained_weights', exist_ok=True)
):
    os.makedirs(model_dir, exist_ok=True)

    urls = ['https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        'https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/denoising_unet.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/motion_module.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/pose_guider.pth',
        'https://huggingface.co/TMElyralab/MusePose/resolve/main/MusePose/reference_unet.pth',
        'https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/unet/diffusion_pytorch_model.bin',
        'https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin',
        'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin'
    ]

    paths = ['dwpose', 'dwpose', 'MusePose', 'MusePose', 'MusePose', 'MusePose', 'sd-image-variations-diffusers/unet', 'image_encoder', 'sd-vae-ft-mse']

    for path in paths:
      dir = os.path.join(model_dir, path)
      os.makedirs(dir, exist_ok=True)

    for url, path in tqdm(zip(urls, paths)):
        filename = os.path.basename(url)
        if filename == "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth":
            filename = "yolox_l_8x8_300e_coco.pth"

        full_file_path = os.path.join(model_dir, path, filename)

        if not os.path.exists(full_file_path):
            print(f"Model '{filename}' does not exists. Downloading to '{full_file_path}'..")
            wget.download(url, full_file_path)

    config_urls = ['https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/unet/config.json',
               'https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json',
               'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json']

    config_paths = ['sd-image-variations-diffusers/unet', 'image_encoder', 'sd-vae-ft-mse']

    # saving config files
    for url, path in tqdm(zip(config_urls, config_paths)):
        filename = os.path.basename(url)
        full_file_path = os.path.join(model_dir, path, filename)
        if not os.path.exists(full_file_path):
            print(f"Model '{filename}' does not exists. Downloading to '{full_file_path}'..")
            wget.download(url, full_file_path)
