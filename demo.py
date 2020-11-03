import torch
import cv2
import numpy as np
import torchvision
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
save_dir = "/home/appuser/detectron2_repo/pytorch_faster_rcnn/demo.jpg"
model_path = "/home/appuser/detectron2_repo/pytorch_faster_rcnn/output/model_100.pth"

transform = transforms.Compose([transforms.ToTensor()])
class_names = ["bg", "tea_box", "tea_stack"]


def get_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2,
                                                             pretrained_backbone=True)  # 或get_object_detection_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def inference(model, image):
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        input_x = transform(image)
        output = model([input_x.to(device)])[0]
    print(output)
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    index = 0
    for x1, y1, x2, y2 in boxes:
        if scores[index] > 0.5:
            print("boxes info", x1, y1, x2, y2)
            cv2.rectangle(image, (np.int32(x1), np.int32(y1)),
                         (np.int32(x2), np.int32(y2)), (0, 255, 255), 1, 8, 0)
            label_id = labels[index]
            label_txt = class_names[label_id]
            cv2.putText(image, label_txt, (np.int32(x1), np.int32(y1)), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        index += 1
    print("over")
    return image


image_path = "/home/appuser/detectron2_repo/frame_800.jpg"
image = cv2.imread(image_path)
model = get_model(model_path)
img = inference(model, image)
cv2.imwrite(save_dir, img)
