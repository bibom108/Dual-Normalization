import logging
import os
import torch
import torch.optim as optim

import tent
import argparse
from model.unetdsbn import Unet2D
from datasets.dataset import Dataset, ToTensor, CreateOnehotLabel
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import medpy.metric.binary as mmb
import time
import matplotlib.pyplot as plt
from PIL import Image
from utils.palette import color_map

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/brats/npz_data')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--test_domain_list', nargs='+', type=str)
parser.add_argument('--model_dir', type=str,  default='./results/unet_dn/model', help='model_dir')
parser.add_argument('--batch_size', type=int,  default=32)
parser.add_argument('--steps', type=int,  default=1)
parser.add_argument('--save_label', dest='save_label', action='store_true')
parser.add_argument('--label_dir', type=str,  default='./results/unet_dn', help='model_dir')
parser.add_argument('--gpu_ids', type=str,  default='0', help='GPU to use')
parser.add_argument('--opti', type=str,  default='Adam', help='Adam or SGD')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--episodic', type=bool, default=False)
FLAGS = parser.parse_args()
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("tent.log"),
                        logging.StreamHandler()
                    ])


def evaluate(description):
    base_model = Unet2D(num_classes=FLAGS.n_classes, num_domains=2, norm='dsbn')
    base_model.load_state_dict(torch.load(os.path.join(FLAGS.model_dir, 'final_model.pth')))
    base_model = base_model.cuda()

    logging.info("test-time adaptation: TENT")
    model = setup_tent(base_model)

    ori_model = Unet2D(num_classes=FLAGS.n_classes, num_domains=2, norm='dsbn')
    ori_model.load_state_dict(torch.load(os.path.join(FLAGS.model_dir, 'final_model.pth')))
    ori_model = ori_model.cuda()
    
    test_domain_list = FLAGS.test_domain_list
    num_domain = len(test_domain_list)

    logging.info(f"{FLAGS.steps} - {FLAGS.episodic}")
    for test_idx in range(num_domain):
        # reset adaptation for each combination of corruption x severity
        # note: for evaluation protocol, but not necessarily needed
        try:
            model.reset()
            logging.info("resetting model")
        except:
            logging.warning("not resetting model")

        dataset = Dataset(base_dir=FLAGS.data_dir, split='test', domain_list=test_domain_list[test_idx],
                        transforms=tfs.Compose([
                            CreateOnehotLabel(num_classes=FLAGS.n_classes),
                            ToTensor()
                        ]))
        dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        tbar = tqdm(dataloader, ncols=150)

        total_dice = 0
        total_hd = 0
        total_asd = 0
        means_list = []
        vars_list = []
        for i in range(2):
            means, vars = get_bn_statis(model, i)
            means_list.append(means)
            vars_list.append(vars)
        model.means_list = means_list
        model.vars_list = vars_list
        model.means_list0 = means_list
        model.vars_list0 = vars_list
        cmap = color_map(n_color=256, normalized=False).reshape(-1)

        total_time = 0
        for idx, (batch, id) in enumerate(tbar):
            start_time = time.time()
            sample_data = batch['image'].cuda()
            onehot_mask = batch['onehot_label']#.detach().numpy()
            mask = batch['label'].detach().numpy()
            model.dis = 99999999
            model.best_out = None
            for domain_id in range(2):
                output = model(sample_data, domain_id=domain_id)
                pred_y = output.cpu().detach().numpy()
                pred_y = np.argmax(pred_y, axis=1)
            
            ori_output = ori_model(sample_data, domain_label=0*torch.ones(sample_data.shape[0], dtype=torch.long))
            ori_pred_y = ori_output.cpu().detach().numpy()
            ori_pred_y = np.argmax(ori_pred_y, axis=1)

            if pred_y.sum() == 0 or mask.sum() == 0:
                # logging.info("==Line 78==")
                total_dice += 0
                total_hd += 100
                total_asd += 100
            else:
                total_dice += mmb.dc(pred_y, mask)
                total_hd += mmb.hd95(pred_y, mask)
                total_asd += mmb.asd(pred_y, mask)
            # end_time = time.time()
            # total_time += end_time - start_time
            # tbar.set_description(f"{total_time / (idx+1)}")
            # tbar.update(1)
            if FLAGS.save_label and idx == 50:
                if not os.path.exists(os.path.join(FLAGS.label_dir, test_domain_list[test_idx])):
                    os.mkdir(os.path.join(FLAGS.label_dir, test_domain_list[test_idx]))
                # for i, pred_mask in enumerate(pred_y):
                #     pred_mask = Image.fromarray(np.uint8(pred_mask.T))
                #     pred_mask = pred_mask.convert('P')
                #     pred_mask.putpalette(cmap)
                #     pred_mask.save(os.path.join(FLAGS.label_dir, test_domain_list[test_idx], id[i] + '.png'))
                for i, pred_mask in enumerate(pred_y):
                    # Convert pred_mask to an image.
                    pred_mask_img = Image.fromarray(np.uint8(pred_mask))
                    pred_mask_img = pred_mask_img.convert('P')
                    pred_mask_img.putpalette(cmap)

                    ori_pred_mask_img = Image.fromarray(np.uint8(ori_pred_y[i]))
                    ori_pred_mask_img = ori_pred_mask_img.convert('P')
                    ori_pred_mask_img.putpalette(cmap)
                    
                    # Convert mask to an image.
                    mask_img = Image.fromarray(np.uint8(mask[i]))
                    mask_img = mask_img.convert('P')
                    mask_img.putpalette(cmap)
                    
                    image_dir = batch['dir'][i]
                    _, image_name = os.path.split(image_dir)
                    sample_data_img = np.load(image_dir)['image'].astype(np.float32)
                    # mask_img = np.load(image_dir)['label'].astype(np.int64)
                    
                    # Plot sample_data, pred_mask, and mask side by side.
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    axes[0].imshow(sample_data_img)
                    axes[0].set_title("Sample Data", fontsize=25)
                    axes[0].axis('off')

                    axes[1].imshow(ori_pred_mask_img)
                    axes[1].set_title("W/o TENT", fontsize=25)
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_mask_img)
                    axes[2].set_title("W/ TENT", fontsize=25)
                    axes[2].axis('off')
                    
                    axes[3].imshow(mask_img)
                    axes[3].set_title("Ground Truth", fontsize=25)
                    axes[3].axis('off')
                    
                    # Save or show the figure as needed.
                    plt.tight_layout()
                    plt.savefig(os.path.join(FLAGS.label_dir, test_domain_list[test_idx], id[i] + '.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
                exit()
        
        # logging.info(f"[{test_domain_list[test_idx]}] Dice: {100*total_dice/len(tbar):.4}, HD: {total_hd/len(tbar):.4}, AHD: {total_asd/len(tbar):.4}")


def get_bn_statis(model, domain_id):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'bns.{}.running_mean'.format(domain_id) in name:
            means.append(param.clone())
        elif 'bns.{}.running_var'.format(domain_id) in name:
            vars.append(param.clone())
    return means, vars


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=FLAGS.steps,
                           episodic=FLAGS.episodic)
    # logging.info(f"model for adaptation: %s", model)
    # logging.info(f"params for adaptation: %s", param_names)
    # logging.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if FLAGS.opti == 'Adam':
        return optim.Adam(params,
                    lr=FLAGS.lr,
                    betas=(0.9, 0.999),
                    weight_decay=0.)
    elif FLAGS.opti == 'SGD':
        return optim.SGD(params,
                   lr=FLAGS.lr,
                   momentum=0.9,
                   dampening=0.0,
                   weight_decay=0.0,
                   nesterov=True)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"BRAIN dataset evaluation.')