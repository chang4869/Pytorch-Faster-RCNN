from engine import train_one_epoch, evaluate
from dataset.Dataset import ForkLiftDataset
from dataset.transform_data import get_transform
import utils
import torch
import os
import torchvision


def main(args):
    print(args)
    device = torch.device(args.device)

# 2 classes, forklift，background
    num_classes = args.num_class
# use our dataset and defined transformations
    print("loading data")
    dataset_train = ForkLiftDataset(args.data_path, get_transform(train=True))
    dataset_test = ForkLiftDataset(args.data_path, get_transform(train=False))

# split the dataset in train and test set
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:525])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[525:])
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    print("creating model ++++++++++++++++++++++++++++++++++++++")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                                num_classes=num_classes, pretrained_backbone=args.pretrained)  # 或get_object_detection_model(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=args.lr,
                            momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
    num_epochs = args.epochs
    output_dir = args.output_dir
    print("start training")
    for epoch in range(num_epochs):

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

    # update the learning rate
        lr_scheduler.step()
        if (epoch + 1) % 25 == 0:
            print("save the {} epoch model parameters++++++++++++++++++++++++".format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(output_dir, "model_{}.pth".format(epoch + 1)))
    # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        print('')
        print('==================================================')
        print('')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="start train faster rcnn")
    parser.add_argument('--data_path', default='/home/appuser/detectron2_repo/'
                                               'pytorch_faster_rcnn/VOC2007_total_tea', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num_class', default='3', type=int)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training ')
    parser.add_argument( "--pretrained", default=True, type=bool,
                          help="Use pre-trained models from the modelzoo",
    )
    parser.add_argument('--output_dir', default='/home/appuser/detectron2_repo/pytorch_faster_rcnn'
                                                '/output', help='path where to save')
    args = parser.parse_args()
    main(args)
