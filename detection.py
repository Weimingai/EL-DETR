"""
@Time : 2024/11/9 11:49
@Auth ： Weiming
@github : https://github.com/Weimingai
@Blog : https://www.cnblogs.com/weimingai/
@File ：detection.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""


import argparse
import warnings
from typing import Dict
from models import build_model
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

torch.set_grad_enabled(False)


def box_cxcywh_to_xyxy(x):
    print('X: ',type(x),x,x.shape)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device) # wxw
    return b


# CLASSES = ['adenomatous','hyperplastic']
CLASSES = ['adenoma','hyperplasic','serrated']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------------------------1. Load model and param ---------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',  # DC 5 need more GPU
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default=r'your_coco_dataset_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='resume_path', # or pretrained path
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # hybrid and EC-loss
    parser.add_argument('--lambda_one2many', default=6, type=int, help='k_one2many')
    parser.add_argument('--alphas_one2one', default=1, type=int, help='lambda_one2one')
    parser.add_argument('--alphas_one2many', default=0.5, type=int, help='lambda_one2many')
    parser.add_argument('--alphas_aux', default=0.5, type=float, help='aux_coff')
    parser.add_argument('--num_queries_one2many', default=500, type=int, help='num_queries_one2many')

    return parser

parser = argparse.ArgumentParser('Load El-DETR', parents=[get_args_parser()])
args = parser.parse_args()
device = torch.device(args.device)

model, criterion, postprocessors = build_model(args)
model.to(device)

# Load state
def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
    missed_list = []
    unmatched_list = []
    matched_state = {}
    for k, v in state.items():
        if k in params:
            if v.shape == params[k].shape:
                matched_state[k] = params[k]
            else:
                unmatched_list.append(k)
        else:
            missed_list.append(k)

    return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

model_without_ddp = model


if args.resume!='':
    state = torch.load(args.resume, map_location='cpu')

    # TODO hard code
    if 'ema' in state:
        stat, infos = _matched_state(model.state_dict(), state['ema']['module'])
    else:
        stat, infos = _matched_state(model.state_dict(), state['model'])

    model_without_ddp.load_state_dict(stat, strict=False)
    print(f'Load model.state_dict, {infos}')
else:
    print('No state_dict existed!')

model.eval()

for name, parameters in model.named_parameters():
    if name == 'query_embed.weight':
        pq = parameters
    if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_weight':
        in_proj_weight = parameters
    if name == 'transformer.decoder.layers.5.multihead_attn.in_proj_bias':
        in_proj_bias = parameters

# --------------------------------------------2.load image and process FF prograss--------------------------------------------------

img_path = 'image.jpg'
im = Image.open(img_path)
img = transform(im).unsqueeze(0).to(device)

# propagate through the model
outputs = model(img)

gt={"category_id":0, "bbox": [318.0, 230.0, 240.0, 236.0]} # ground-truth

result_list=[]

class BBox:
    def __init__(self,bbox, size):
        img_w, img_h = size
        self.x = bbox[0] * img_w
        self.y = bbox[1] * img_h
        self.w = bbox[2] * img_w
        self.h = bbox[3] * img_h
    def print_info(self):
        print(self.x, self.y, self.w, self.h)

    def xyxy(self):
        self.xmin = self.x - self.w * 0.5
        self.ymin = self.y - self.h * 0.5
        self.xmax = self.x + self.w * 0.5
        self.ymax = self.y + self.h * 0.5
        return (self.xmin, self.ymin, self.xmax, self.ymax)

def iou(a,b):
    assert isinstance(a,BBox)
    assert isinstance(b,BBox)
    area_a = a.w * a.h
    area_b = b.w * b.h
    w = min(b.x+b.w,a.x+a.w) - max(a.x,b.x)
    h = min(b.y+b.h,a.y+a.h) - max(a.y,b.y)
    if w <= 0 or h <= 0:
        return 0
    area_c = w * h
    return area_c / (area_a + area_b - area_c)

object_bbox = BBox(gt["bbox"], (1,1))
object_dict = {'idx':-1,
              'logit': -1,
              'class_index': gt["category_id"],
              'bbox':  gt["bbox"],
              'iou': iou( object_bbox, object_bbox),
              'xyxy': object_bbox.xyxy()
              }
for ele in range(num_queries):
    pre_bbox = BBox(outputs['pred_boxes'][0][ele], im.size)
    s_dict = {'idx':ele,
              'logit': outputs['pred_logits'][0][ele],
              'class_index': torch.argmax(outputs['pred_logits'][0][ele]),
              'bbox': outputs['pred_boxes'][0][ele],
              'iou': iou( pre_bbox, object_bbox),
              'xyxy': pre_bbox.xyxy()
              }
    result_list.append(s_dict)

sorted_results = sorted(result_list, key=lambda e_reslut: e_reslut['iou'], reverse=True)


# ------------------------------------------------3. att value---------------------------------------------------
# use lists to store the outputs via up-values
conv_features, dec_attn_weights = [], []
cq = []
pk = []


# hook
hooks = [
    model.backbone[-2].register_forward_hook(
        lambda self, input, output: conv_features.append(output)
    ),
    model.transformer.decoder.layers[-6].cross_attn.register_forward_hook( # wxw cross_attn  multihead_attn
        lambda self, input, output: dec_attn_weights.append(output[1])
    ),
    model.backbone[-1].register_forward_hook(
        lambda self, input, output: pk.append(output)
    ),
]

# propagate through the model
outputs = model(img)

for hook in hooks:
    hook.remove()

# don't need the list anymore
conv_features = conv_features[0]
dec_attn_weights = dec_attn_weights[0]

# ----------------------------------------------------------4. plot---------------------------------------------------------
h, w = conv_features['0'].tensors.shape[-2:]

plot_result = sorted_results[:3] # top three also set threshold of IOU etc.
print('object: ',object_dict,'img size: ',im.size)

fig, axs = plt.subplots(ncols=len(plot_result), nrows=2,figsize=(10, 6))
colors = COLORS * 100

for ax_i, s_result in zip(axs.T, plot_result):
    print('result: ',s_result)
    idx = s_result['idx']
    (xmin, ymin, xmax, ymax) = s_result['xyxy']

    # 可视化decoder的注意力权重
    ax = ax_i[0]
    ax.imshow(dec_attn_weights[0, idx].cpu().view(h, w))

    ax.axis('off')
    ax.set_title(f'query id: {idx}', fontsize=15)

    if idx == -1:
        s_color='red'
    else:
        s_color='blue'

    ax = ax_i[1]
    ax.imshow(im)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color=s_color, linewidth=3))
    ax.axis('off')
    ax.set_title(CLASSES[s_result['class_index']], fontsize=15)
fig.tight_layout()
plt.show()

