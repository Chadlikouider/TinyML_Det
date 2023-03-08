"""
    This file is for training the ssd_models including(Squezzenet-SSD, EtinyNet-SSD , and mobileNetV2-SSD)

    @author: CHADLI KOUIDER
"""

import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from dataset.pascal_voc import PascalVOCDataset
from models.nn.multibox_loss import MultiboxLoss
from models.ssd.ssd import MatchPrior
from models.ssd.etinynet_ssd import create_etinynet_ssd_lite
from models.ssd.Squeezenet_ssd import create_squeezenet_ssd_lite
from config import squeezenet_ssd_config
from config import etinynet_ssd_config



parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')

parser.add_argument('--net', default="etinynet-ssd",
                    help="The network architecture, it can be squeezenet-ssd, etinynet-ssd, or mb2-ssd.")


# Train params
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

def main():
    timer = Timer()

    if args.net == 'etinynet-ssd':
        create_net = create_etinynet_ssd_lite
        config = etinynet_ssd_config
    elif args.net == 'squeezenet-ssd':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    if config.type == 'float64':
        dtype = torch.float64
    elif config.type == 'float32':
        dtype = torch.float32
    elif config.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8
    
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, config.ssd_iou_thresh)
    logging.info("Prepare training datasets.")
    datasets = []
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean = config.MEAN, 
                                                                std = config.STD)])
    
    train_dataset = PascalVOCDataset(root_dir = config.train_path, transform = train_transforms)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=False, collate_fn=train_dataset.collate)


    logging.info("Prepare Validation datasets.")
    val_dataset = PascalVOCDataset(root_dir = config.val_path, transform = train_transforms)
    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle = False, collate_fn=val_dataset.collate)
    

    logging.info("Build network.")
    net = create_net(config.num_classes)
    min_loss = -10000.0
    last_epoch = -1
    

    if config.start_from_scratch == False:
        # resume checkpoint or import a pre-trained model
        timer.start("Load Model") 
        if config.resume:
            if not os.path.isfile(config.resume):
                raise ValueError(f"Checkpoint file {config.resume} not found.")
            logging.info(f"Resume from the model {config.resume}")
            net.load(config.resume)
        elif config.base_net:
            if not os.path.isfile(config.base_net):
                raise ValueError(f"Base network file {config.base_net} not found.")
            logging.info(f"Init from base net {config.base_net}")
            net.init_from_base_net(config.base_net)
        elif config.pretrained_ssd:
            if not os.path.isfile(config.pretrained_ssd):
                raise ValueError(f"Pretrained SSD file {config.pretrained_ssd} not found.")
            logging.info(f"Init from pretrained ssd {config.pretrained_ssd}")
            net.init_from_pretrained_ssd(config.pretrained_ssd)
        else:
            raise ValueError("One path of `resume`, `base_net`, or `pretrained_ssd` must be provided.")
        logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    
    else:
        logging.info("start from scratch: initialize the entirety of the model")
        net.init()

    
    # Freezing layers
    if config.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': config.learning_rate},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif config.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': config.learning_rate},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': config.learning_rate},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
        logging.info("There are no layers that are frozen.")
    
    net.to(DEVICE)

    # define loss function (criterion) and optimizer
    criterion = MultiboxLoss(config.priors, iou_threshold=config.ssd_iou_thresh, neg_pos_ratio=3,
                             center_variance=config.center_variance, size_variance=config.size_variance,
                             device=DEVICE)
    
    optimizer = torch.optim.SGD(params, lr=config.learning_rate, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    # Select the type of scheduler
    if config.scheduler_type == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in config.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=config.gamma, last_epoch=last_epoch)
    elif config.scheduler_type == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        # 120 is (T_max) value for Cosine Annealing Scheduler
        scheduler = CosineAnnealingLR(optimizer, 120, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {config.scheduler_type}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # ckeck of the checkpoint directory exists, if not then make one
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # create a SummaryWriter object to write to TensorBoard logs
    log_dir = config.log_dir  # specify the directory where you want to store the logs
    writer = SummaryWriter(log_dir=log_dir)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, config.num_epochs):

        print('\nEpoch: [%d | %d]' % (epoch + 1, config.num_epochs))

        scheduler.step()

        train_loss, train_regression_loss, train_classification_loss = train(train_loader,
                                                                             net, criterion, 
                                                                             optimizer,
                                                                             device=DEVICE)
        
        val_loss, val_regression_loss, val_classification_loss = validate(val_loader, net, criterion, DEVICE)


        # log training and validation losses to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train_regression", train_regression_loss, epoch)
        writer.add_scalar("Loss/train_classification", train_classification_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/val_regression", val_regression_loss, epoch)
        writer.add_scalar("Loss/val_classification", val_classification_loss, epoch)



        logging.info(
                f"Epoch: {epoch}, " +
                f"Training Loss: {train_loss:.4f}, " +
                f"Training localization Loss {train_regression_loss:.4f}, " +
                f"Training Classification Loss: {train_classification_loss:.4f}"+
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
        

        # Save check point
        if epoch % config.save_interval == 0 or epoch == config.num_epochs - 1:

            now = datetime.now()
            date_time = now.strftime("%Y%m%dT%H%M")

            model_path = os.path.join(config.checkpoint_dir, f"{args.net}_Epoch{epoch}_Loss{val_loss}_{date_time}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")


    # close the SummaryWriter
    writer.close()   # to visualize the results, run this in terminal: tensorboard --logdir=path/to/log/directory


# Train function
def train(loader, net, criterion, optimizer, device):

    #switch to train mode
    net.train()

    # initialize the return variables
    total_loss = 0
    loc_loss = 0
    class_loss = 0


    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)


        
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss

        # accumelate losses
        total_loss += loss.item()
        loc_loss += regression_loss.item()
        class_loss += classification_loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Average the results
    total_loss = total_loss/len(loader)
    loc_loss = loc_loss/len(loader)
    class_loss = class_loss/len(loader)

    return total_loss, loc_loss, class_loss




# Validation function
# Use torch.no_grad() to disable gradient calculation during validation
@torch.no_grad()
def validate(loader, net, criterion, device):

    # switch to evaluate mode
    net.eval()

    total_loss = 0
    loc_loss = 0
    class_loss = 0

    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss


        total_loss += loss.item()
        loc_loss += regression_loss.item()
        class_loss += classification_loss.item()

    
    return total_loss/len(loader), loc_loss/len(loader), class_loss/len(loader)
        

if __name__ == '__main__':
    main()
