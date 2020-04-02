import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, model_urls
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torch.jit.annotations import Tuple, List, Dict, Optional
from torchvision.models.detection import _utils as det_utils

from torchvision.ops import boxes as box_ops
from torchvision.ops import misc as misc_nn_ops

import warnings


def ssd_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (list[Tensor])

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    #
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    #sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    #labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    #box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression,
        regression_targets,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss



class SSDResNet(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone_name == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone_name == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone_name == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]


        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
class SSDMultilevelFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.feature_extractor = backbone
        self.out_channels = backbone.out_channels
        self._build_additional_features(self.feature_extractor.out_channels)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        num_internal_channels = [256, 256, 128, 128, 128]
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], num_internal_channels)):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        for layer in self.additional_blocks:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.feature_extractor(x)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        return detection_feed

COCO_NUM_CLASSES = 81 
NUM_BOX_COORDINATES = 4
class SSDPredictor(nn.Module): 
    def __init__(self, channels_per_featuremap, label_num=COCO_NUM_CLASSES, num_boxes_per_loc=6):
        super().__init__()
        #self.num_boxes_per_layer = [4, 6, 6, 6, 4, 4]
        self.locs = []
        self.confs = []
        self.label_num = label_num
        self.num_boxes_per_layer = num_boxes_per_loc

        for feature_channels in channels_per_featuremap:
            self.locs.append(nn.Conv2d(feature_channels, self.num_boxes_per_layer * NUM_BOX_COORDINATES, kernel_size=3, padding=1))
            self.confs.append(nn.Conv2d(feature_channels, self.num_boxes_per_layer * self.label_num, kernel_size=3, padding=1))
        self.locs = nn.ModuleList(self.locs)
        self.confs = nn.ModuleList(self.confs)
        self._init_weights()
    def _init_weights(self):
        layers = [*self.locs, *self.confs]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def forward(self, feature_list):
        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        ret = []

        # feature_list is a list of feature maps. Each map have shape (n_batches, n_channels, H, W)
        first_map = 0
        n_batches = feature_list[first_map].shape[0]
        plain_height_width_box_types = -1

        for feature_map, regression_layer, clf_layer in zip(feature_list, self.locs, self.confs):
            box_regression = regression_layer(feature_map).view(
                    n_batches, plain_height_width_box_types, NUM_BOX_COORDINATES)
            clf_logits = clf_layer(feature_map).view(
                    n_batches, plain_height_width_box_types, self.label_num)
            ret.append((box_regression, clf_logits))

        locs, confs = list(zip(*ret))
        along_plain_height_width_box_type_dim = 1
        locs = torch.cat(locs, along_plain_height_width_box_type_dim)
        confs = torch.cat(confs, along_plain_height_width_box_type_dim)

        # convert batch dimension to list
        locs = [batch_elem for batch_elem in locs]
        confs = [batch_elem for batch_elem in confs]

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nbox_coord} results
        return locs, confs



class SSDHead(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 ):
        super(SSDHead, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor])
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = torch.tensor(0)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = torch.tensor(-1)  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor])
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor])
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def DELTEME_all(self, the_list):
        # type: (List[bool])
        for i in the_list:
            if not i:
                return False
        return True

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]])
        assert targets is not None
        assert self.DELTEME_all(["boxes" in t for t in targets])
        assert self.DELTEME_all(["labels" in t for t in targets])
        

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        #$

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        if len(boxes_per_image) == 1:
            # TODO : remove this when ONNX support dynamic split sizes
            # and just assign to pred_boxes instead of pred_boxes_list
            pred_boxes_list = [pred_boxes]
            pred_scores_list = [pred_scores]
        else:
            pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
            pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            
            row_to, collumn = 1, -1
            labels = labels.view(row_to, collumn).expand_as(scores)

            along_class_prediction = 1
            pred_class_labels = torch.argmax(scores, dim=along_class_prediction)

            prediction_num = torch.arange(boxes.shape[0], device=device)
            flat_pred_class_idxs = prediction_num * num_classes + pred_class_labels
            top1_scores = torch.take(scores, flat_pred_class_idxs)
            labels = pred_class_labels
            


            # remove predictions with the background label
            #boxes = boxes[:, 1:]
            #scores = scores[:, 1:]
            #labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            #boxes = boxes.reshape(-1, 4)
            #scores = scores.reshape(-1)
            #labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(top1_scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds].squeeze(1), top1_scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                

        #
    #import pdb; pdb.set_trace()

        box_regression, class_logits = self.box_predictor(features)
        
        
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            class_logits = [img_logits[matched_idx,:] for img_logits, matched_idx in zip(class_logits, matched_idxs)]
            box_regression = [img_regression[matched_idx, :] for img_regression, matched_idx in zip(box_regression, matched_idxs)]
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        
        class_logits, box_regression = torch.cat(class_logits), torch.cat(box_regression)
        #labels, regression_targets = torch.cat(labels), torch.cat(regression_targets)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = ssd_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result, losses


class GeneralizedSSD300(nn.Module):
    def __init__(self, transform, backbone, anchor_generator, ssd_head, num_classes):

        super().__init__()
        self.transform = transform
        self.feature_extractor = backbone

        self.anchor_sizes = [128, 256, 512]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        
        self.label_num = num_classes  # number of COCO classes
        self.ssd_head = ssd_head
        self._has_warned = False

        self.anchor_generator = anchor_generator

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        #import pdb; pdb.set_trace()
        images, targets = self.transform(images, targets)

        features = self.feature_extractor(images.tensors)
        anchors = self.anchor_generator(images, features)

        detections, detector_losses = self.ssd_head(features, anchors, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)

        vec_gd = self._loc_vec(gloc)

        # sum on four coordinates, and mask
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float()*sl1).sum(dim=1)

        # hard negative mining
        con = self.con_loss(plabel, glabel)

        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*(mask.float() + neg_mask.float())).sum(dim=1)

        # avoid no object detected
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        return ret
