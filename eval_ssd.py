import torch
import argparse
import pathlib
import numpy as np
import logging
import sys


from utils import box_utils, common_tools
from utils.misc import str2bool, Timer
from config import etinynet_ssd_config, squeezenet_ssd_config
from models.ssd.etinynet_ssd import create_etinynet_ssd_lite, create_etinynet_ssd_lite_predictor
from models.ssd.Squeezenet_ssd import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="etinynet-ssd",
                    help="The network architecture, it should be of squeezenet-ssd,etinynet-ssd, or mobilenet-ssd..")

args = parser.parse_args()
use_cuda = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")



def main():
    timer = Timer()
    if args.net == 'etinynet-ssd':
        config = etinynet_ssd_config
        net = create_etinynet_ssd_lite(config.num_classes,is_test=True)
    elif args.net == 'squeezenet-ssd':
        config = squeezenet_ssd_config
        net = create_squeezenet_ssd_lite(config.num_classes, is_test=True)
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)


    eval_path = pathlib.Path(config.eval_dir)
    eval_path.mkdir(exist_ok=True)

    class_names = config.class_names

    timer.start("Load model")
    net.load(config.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    if args.net == 'etinynet-ssd':
        predictor = create_etinynet_ssd_lite_predictor(net, nms_method=config.nms_method, device=DEVICE)
    elif args.net == 'squeezenet-ssd':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of etinynet-ssd, squeezenet-ssd and mb2-ssd.")
        parser.print_help(sys.stderr)
        sys.exit(1)


    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    
if __name__ == '__main__':
    main()