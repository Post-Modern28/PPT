import glob
import os
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pyiqa
from pyiqa import create_metric
from skimage.transform import resize
import torch
from imageio.v2 import imread

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to generated images folder")
args = parser.parse_args()

label_path = '/trinity/home/m.romanov/benchmarking/datasets/MIST_HER2/testB/'
image_path = args.image_path

label_list = sorted([f for f in os.listdir(label_path) if f.endswith('.jpg')])
image_list = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])

#-----------------ssim & psnr-----------------#
sum_ssim = 0
sum_psnr = 0

err_flag = 0
for i in range(len(label_list)):
    #print('Path 1:', label_path + label_list[i])
    #print('Path 2:', image_path + '/' + image_list[i])
    
    label = imread((label_path+ label_list[i]))
    image = imread((image_path+ '/' + image_list[i]))
    image = resize(image, label.shape, anti_aliasing=True, preserve_range=True).astype(label.dtype)
    # label = resize(label, image.shape, anti_aliasing=True, preserve_range=True).astype(label.dtype)
    if label.shape == image.shape:
        sum_ssim += ssim(label, image, channel_axis=-1)
        sum_psnr += psnr(label, image)
    else:
        err_flag = 1
        print(f"Skip {label_list[i]} vs {image_list[i]}: shape mismatch {label.shape} vs {image.shape}")

if err_flag:
    print('Some images were skipped due to shape mismatch')
    
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
iqa_model = pyiqa.create_metric('lpips', metric_mode='FR').to(device)
metric_mode = iqa_model.metric_mode

for idx, img_path in enumerate(input_paths):
    ref_img_path = ref_paths[idx]
    img = imread(img_path)
    ref = imread(ref_img_path)
    if img.shape != ref.shape:
        img = resize(img, ref.shape, anti_aliasing=True, preserve_range=True).astype(ref.dtype)
    img_t  = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ref_t  = torch.tensor(ref).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips_score = iqa_model(img_t.to(device), ref_t.to(device)).cpu().item()
    avg_lpips += lpips_score

#-----------------fid-----------------#
fid_metric = pyiqa.create_metric('fid')

fid_score = fid_metric(label_path, image_path)
print("SSIM:", round(sum_ssim/len(label_list),4))
print("PSNR:", round(sum_psnr/len(label_list),4))
print("LPIPS:", round(avg_lpips/len(ref_paths),4))
print("FID:", round(fid_score,4))

with open(r'/trinity/home/m.romanov/benchmarking/PPT/logs/eval_mist.txt', 'a') as f:
    print("SSIM:", )
    print("PSNR:", )
    print("LPIPS:", )
    s0 = f'{image_path}'
    s = '| PSNR | SSIM | LPIPS | FID  |\n' 
    s2 = f'|{round(sum_psnr/len(label_list),4)} | {round(sum_ssim/len(label_list),4)} | {round(avg_lpips/len(ref_paths),4)}  | {round(fid_score,4)} |\n'
    f.write(s0)
    f.write(s)
    f.write(s2)
    


