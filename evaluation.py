import glob
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pyiqa
from pyiqa import create_metric
from skimage.transform import resize
import torch
from imageio.v2 import imread

import torch
from imageio.v2 import imread
from skimage.transform import resize
device = 'cuda' if torch.cuda.is_available() else 'cpu'


label_path = '/trinity/home/m.romanov/benchmarking/datasets/MIST_HER2/testB/'
image_path = '/trinity/home/m.romanov/benchmarking/PPT/results/test/test_latest/images/'


label_list = sorted([f for f in os.listdir(label_path) if f.endswith('.jpg')])
image_list = [f[:-4]+'_fake_B.png' for f in label_list]
#image_list = sorted([f for f in os.listdir(image_path) if f.endswith('.png')  and not f.startswith('0') and 'heck' not in f])
#-----------------ssim & psnr-----------------#
sum_ssim = 0
sum_psnr = 0

for i in range(len(label_list)):
    #print('Path 1:')
    #print(label_path+ label_list[i])
    #print('Path 2:')
    #print(label_path+ image_list[i])
    label = imread((label_path+ label_list[i]))
    image = imread((image_path+ image_list[i]))
    label = resize(label, image.shape, anti_aliasing=True, preserve_range=True).astype(label.dtype)
    if label.shape == image.shape:
        sum_ssim += ssim(label, image, channel_axis=-1)
        sum_psnr += psnr(label, image)
    else:
        print(f"Skip {label_list[i]} vs {image_list[i]}: shape mismatch {label.shape} vs {image.shape}")
    
    #sum_ssim += ssim(label, image,channel_axis=-1)
    #sum_ssim += ssim(label, image,multichannel=True)
    #sum_psnr += psnr(label, image)


if os.path.isfile(label_path):
    input_paths = [label_path]
    if image_path is not None:
        ref_paths = [image_path]
else:
    input_paths = sorted(glob.glob(os.path.join(label_path, '*')))
    if image_path is not None:
        ref_paths = sorted(glob.glob(os.path.join(image_path, '*')))


# ----------------- lpips ----------------- #
avg_lpips = 0
iqa_model = pyiqa.create_metric('lpips', metric_mode='FR').to(device)  # если есть GPU
metric_mode = iqa_model.metric_mode

for idx, img_path in enumerate(input_paths):
    ref_img_path = ref_paths[idx]

    # читаем обе картинки
    img = imread(img_path)
    ref = imread(ref_img_path)

    # приводим к одинаковому размеру (например, к размерам ref)
    if img.shape != ref.shape:
        img = resize(img, ref.shape, anti_aliasing=True, preserve_range=True).astype(ref.dtype)

    # переводим в torch.Tensor (C, H, W) и нормализуем в [0,1]
    img_t  = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ref_t  = torch.tensor(ref).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # считаем LPIPS
    lpips_score = iqa_model(img_t.to(device), ref_t.to(device)).cpu().item()
    avg_lpips += lpips_score



#-----------------fid-----------------#
fid_metric = pyiqa.create_metric('fid')
fid_score = fid_metric(label_path, image_path)


print("SSIM:",round(sum_ssim/len(label_list),4))
print("PSNR:",round(sum_psnr/len(label_list),4))
print("FID:",round(fid_score,4))
print("LPIPS:", round(avg_lpips / len(ref_paths), 4))

