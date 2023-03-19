import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
# from model import architecture, esrt
from model import esrt
from data import DIV2K, Set5_val, Common_val, Flickr8k
import utils
import skimage.color as sc
import random
from collections import OrderedDict
import datetime
from importlib import import_module

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="ESRT")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="../dataset",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model", type=str, default='ESRT')

# New
parser.add_argument("--training_set", type=str, default='DIV2K')
parser.add_argument("--validation_set", type=str, default='Set5')

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ESRT",

    # track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "architecture": "ESRT",
        "dataset": "DIV2K",
        "epochs": args.nEpochs,
        "batch_size": args.batch_size,
        "scale": args.scale,
        "patch_size": args.patch_size,
        "optimizer": "Adam",
    },

    resume=False
)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

if args.training_set == 'DIV2K':
    trainset = DIV2K.div2k(args)
elif args.training_set == 'Flickr8k':
    trainset = Flickr8k.Flickr8k(args)
else:
    raise ValueError('Training set not supported')

testset_path = os.path.join(args.root, "{}/".format(args.validation_set))
testset = Common_val.DatasetFromFolderVal(testset_path + "HR/",
                                          testset_path + "LR/X{}/".format(args.scale),
                                          args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True

model = esrt.ESRT(upscale=args.scale)  # architecture.IMDN(upscale=args.scale)

l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scaler = GradScaler()


def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)

    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])

    running_loss = 0.0
    epoch_loss = 0.0
    print_step = len(training_data_loader) // 10
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            sr_tensor = model(lr_tensor)
            loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        running_loss += loss_l1.item()
        epoch_loss += loss_l1.item()

        scaler.scale(loss_sr).backward()
        scaler.step(optimizer)
        scaler.update()
        if iteration % print_step == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch,
                                                                  iteration,
                                                                  len(training_data_loader),
                                                                  loss_l1.item(),
                                                                  running_loss / print_step))
            running_loss = 0.0

    return epoch_loss / len(training_data_loader)


def forward_chop(model, x, scale, shave=10, min_size=60000):
    # scale = scale#self.scale[self.idx_scale]
    n_GPUs = 1  # min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def valid(scale):
    model.eval()

    avg_psnr, avg_ssim, loss_valid = 0.0, 0.0, 0.0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = forward_chop(model, lr_tensor, scale)  # model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        loss_valid = l1_criterion(pre, hr_tensor).item()
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        # print(im_pre.shape)
        # print(im_label.shape)
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    # print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader),
    #                                                       avg_ssim / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader), loss_valid


def save_checkpoint(epoch):
    model_folder = "experiment/{}_checkpoint_ESRT_x{}/".format(args.training_set, args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
code_start = datetime.datetime.now()
timer = utils.Timer()
for epoch in range(args.start_epoch, args.nEpochs + 1):
    t_epoch_start = timer.t()
    epoch_start = datetime.datetime.now()
    loss = train(epoch)
    psnr, ssim, valid_loss = valid(args.scale)

    # log with wandb
    wandb.log({'training_loss': loss,
               # 'validation_loss': valid_loss_div2k,
               'Set5_loss': valid_loss,
               'psnr': psnr,
               'ssim': ssim
               })

    if epoch % 10 == 0:
        save_checkpoint(epoch)
    epoch_end = datetime.datetime.now()
    print('Epoch cost times: %s' % str(epoch_end - epoch_start))
    t = timer.t()
    prog = (epoch - args.start_epoch + 1) / (args.nEpochs + 1 - args.start_epoch + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
code_end = datetime.datetime.now()
print('Code cost times: %s' % str(code_end - code_start))
