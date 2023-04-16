import os
import time

import cv2
import argparse
import numpy as np
import torch
import skimage.color as sc
import utils
from model import esrt

# Testing settings
parser = argparse.ArgumentParser(description='ESRT')
parser.add_argument("--test_folder", type=str, default='../dataset/',
                    help='the folder containing the datasets to test')
parser.add_argument("--output_folder", type=str, default='results/')
parser.add_argument("--checkpoint", type=str, default='checkpoints/DIV2K_checkpoint_ESRT_x2/epoch_1000.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu')
parser.add_argument("--scale", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--patch_size", type=int, default=48,
                    help='patch size')
parser.add_argument("--batch_size", type=int, default=16,
                    help='batch size for processing patches')
parser.add_argument("--single_dataset", type=str, default='None',
                    help='test on a single dataset')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

# Define the device
cuda = not opt.cpu
device = torch.device('cuda' if cuda else 'cpu')

# Load the model
model = esrt.ESRT(upscale=opt.scale)
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=False)
# model = torch.compile(model, mode="reduce-overhead")

if cuda:
    torch.set_float32_matmul_precision('high')
    model = model.to(device)

# Add date and time of test
now = time.localtime()
s = time.strftime('%Y-%m-%d %H:%M:%S', now) + ' - {}\n'.format(opt.checkpoint)
results = open(os.path.join(opt.output_folder, 'results_X' + str(opt.scale) + '.txt'), 'a')
results.write(s)
results.close()

for dataset in os.listdir(opt.test_folder):

    if dataset == '.idea':
        break

    st = time.time()

    if opt.single_dataset != 'None':
        dataset = opt.single_dataset

    if dataset != 'Manga109':
        ext = '.png'
    else:
        ext = '.jpg'

    hr_list = utils.get_list(os.path.join(opt.test_folder, dataset, 'HR'), ext=ext)
    test_lr_folder = os.path.join(opt.test_folder, dataset, 'LR', 'X' + str(opt.scale))
    output_folder = os.path.join(opt.output_folder, dataset, 'ESRT_X' +
                                 str(opt.scale) +
                                 '_' +
                                 str(opt.checkpoint.split('/')[-1].split('.')[0]))
    print(output_folder)

    dataset_psnr = []
    dataset_ssim = []

    for imname in hr_list:

        if dataset == 'Manga109' and hr_list.index(imname) % 10 != 0:
            continue

        print(imname)
        im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        im_gt = utils.modcrop(im_gt, opt.scale)
        im_l = cv2.imread(os.path.join(test_lr_folder, imname.split('/')[-1].split('.')[0] + ext),
                          cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB

        # Extract patches
        gt_patches = utils.extract_patches(im_gt, opt.patch_size * opt.scale)
        lr_patches = utils.extract_patches(im_l, opt.patch_size)

        # Upscale patches and calculate metrics
        psnr_list = []
        ssim_list = []
        num_patches = len(lr_patches)
        batch_size = opt.batch_size

        for idx in range(0, num_patches, batch_size):
            batch_lr_patches = lr_patches[idx:idx + batch_size]
            batch_im_input = np.stack([patch / 255.0 for patch in batch_lr_patches], axis=0)
            batch_im_input = np.transpose(batch_im_input, (0, 3, 1, 2))
            batch_im_input = torch.from_numpy(batch_im_input).float().to(device)

            # Upscale the patches
            with torch.cuda.amp.autocast(), torch.no_grad():
                batch_im_output = model(batch_im_input)

            batch_im_output = batch_im_output.data.cpu().numpy()
            batch_im_output = np.transpose(batch_im_output, (0, 2, 3, 1))
            batch_im_output = np.clip(batch_im_output * 255.0, 0, 255)
            batch_im_output = batch_im_output.astype(np.uint8)

            # Calculate PSNR and SSIM for each patch in the batch
            for i, (im_output, lr_patch) in enumerate(zip(batch_im_output, batch_lr_patches)):
                gt_patch = gt_patches[idx + i]

                if opt.is_y:
                    im_output_y = utils.quantize(sc.rgb2ycbcr(im_output)[:, :, 0])
                    gt_patch_y = utils.quantize(sc.rgb2ycbcr(gt_patch)[:, :, 0])
                    psnr_list.append(utils.compute_psnr(im_output_y, gt_patch_y))
                    ssim_list.append(utils.compute_ssim(im_output_y, gt_patch_y))
                else:
                    psnr_list.append(utils.compute_psnr(im_output, gt_patch))
                    ssim_list.append(utils.compute_ssim(im_output, gt_patch))

                # Save the first two patches for each image
                if idx + i < 2:
                    output_path = os.path.join(output_folder, imname.split('/')[-1].split('.')[0])
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    cv2.imwrite(os.path.join(output_path, f'patch_lr_{i + 1}.png'), lr_patch[:, :, [2, 1, 0]])
                    cv2.imwrite(os.path.join(output_path, f'patch_output_{i + 1}.png'), im_output[:, :, [2, 1, 0]])
                    cv2.imwrite(os.path.join(output_path, f'patch_gt_{i + 1}.png'), gt_patch[:, :, [2, 1, 0]])

        print(f"PSNR: {np.mean(psnr_list)}, SSIM: {np.mean(ssim_list)}")

        dataset_psnr.append(np.mean(psnr_list))
        dataset_ssim.append(np.mean(ssim_list))

    print(f"Dataset PSNR: {np.mean(dataset_psnr)}, Dataset SSIM: {np.mean(dataset_ssim)}")

    results = open(os.path.join(opt.output_folder, 'results_X' + str(opt.scale) + '.txt'), 'a')
    results.write("{} - Mean PSNR: {}, SSIM: {}, TIME: {}s\n".format(dataset,
                                                                     np.mean(dataset_psnr),
                                                                     np.mean(dataset_ssim),
                                                                     (time.time() - st)))
    results.close()

    if opt.single_dataset != 'None':
        break
