import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import torchvision

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class RetinaNet(nn.Module):

    # MODIFIED: Added more freezing options to the constructor
    def __init__(self, num_classes, backbone='resnet50', pretrained=False, 
                 freeze_backbone=False, freeze_fpn=False, 
                 freeze_regression_head=False, freeze_classification_head=False):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        if 'resnet' in backbone:
            # ResNet specific initialization
            if backbone == 'resnet18':
                layers = [2, 2, 2, 2]
                block = BasicBlock
            elif backbone == 'resnet34':
                layers = [3, 4, 6, 3]
                block = BasicBlock
            elif backbone == 'resnet50':
                layers = [3, 4, 6, 3]
                block = Bottleneck
            elif backbone == 'resnet101':
                layers = [3, 4, 23, 3]
                block = Bottleneck
            elif backbone == 'resnet152':
                layers = [3, 8, 36, 3]
                block = Bottleneck
            else:
                raise ValueError('Unsupported ResNet backbone version.')

            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            if block == BasicBlock:
                fpn_sizes = [self.layer2[-1].conv2.out_channels, self.layer3[-1].conv2.out_channels,
                             self.layer4[-1].conv2.out_channels]
            elif block == Bottleneck:
                fpn_sizes = [self.layer2[-1].conv3.out_channels, self.layer3[-1].conv3.out_channels,
                             self.layer4[-1].conv3.out_channels]
            else:
                raise ValueError(f"Block type {block} not understood")
            
            if pretrained:
                print(f"Loading pretrained weights for {backbone}")
                state_dict = model_zoo.load_url(model_urls[backbone])
                self.load_state_dict(state_dict, strict=False)


        elif backbone == 'efficientnet-b0':
            effnet = torchvision.models.efficientnet_b0(pretrained=pretrained)
            self.backbone_layers = effnet.features
            fpn_sizes = [40, 112, 320]

        else:
            raise ValueError(f"Backbone '{backbone}' not supported.")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256, num_anchors=9, feature_size=256)
        self.classificationModel = ClassificationModel(256, num_anchors=9, num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        for m in self.fpn.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.regressionModel.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.classificationModel.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        # NEW: Granular freezing logic
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_fpn:
            self._freeze_fpn()
        if freeze_regression_head:
            self._freeze_regression_head()
        if freeze_classification_head:
            self._freeze_classification_head()
        
        self.freeze_bn()

    def _freeze_backbone(self):
        print("Freezing backbone layers...")
        if 'resnet' in self.backbone_name:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False
        elif 'efficientnet' in self.backbone_name:
            for param in self.backbone_layers.parameters():
                param.requires_grad = False

    # NEW: Method to freeze FPN layers
    def _freeze_fpn(self):
        print("Freezing FPN layers...")
        for param in self.fpn.parameters():
            param.requires_grad = False

    # NEW: Method to freeze regression head
    def _freeze_regression_head(self):
        print("Freezing regression head...")
        for param in self.regressionModel.parameters():
            param.requires_grad = False

    # NEW: Method to freeze classification head
    def _freeze_classification_head(self):
        print("Freezing classification head...")
        for param in self.classificationModel.parameters():
            param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers by setting them to eval mode.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        if 'resnet' in self.backbone_name:
            x = self.conv1(img_batch)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x1 = self.layer1(x)
            c3 = self.layer2(x1)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            features = self.fpn([c3, c4, c5])
        
        elif self.backbone_name == 'efficientnet-b0':
            x = self.backbone_layers[0](img_batch)
            x = self.backbone_layers[1](x)
            x = self.backbone_layers[2](x)
            c3 = self.backbone_layers[3](x)
            x = self.backbone_layers[4](c3)
            c4 = self.backbone_layers[5](x)
            x = self.backbone_layers[6](c4)
            c5 = self.backbone_layers[7](x)
            features = self.fpn([c3, c4, c5])
            
        else:
            raise ValueError(f"Backbone '{self.backbone_name}' not supported in forward pass.")


        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            finalResult = [[], [], []]
            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])
            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            if classification.shape[1] > 0:
                for i in range(classification.shape[2]):
                    scores = torch.squeeze(classification[:, :, i])
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:
                        continue
                    scores = scores[scores_over_thresh]
                    anchorBoxes = torch.squeeze(transformed_anchors)
                    anchorBoxes = anchorBoxes[scores_over_thresh]
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                    finalResult[0].extend(scores[anchors_nms_idx])
                    finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                    finalResult[2].extend(anchorBoxes[anchors_nms_idx])
                    finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                    finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                    if torch.cuda.is_available():
                        finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
                    finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                    finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


# MODIFIED: All factory functions now accept **kwargs to pass freezing options
def resnet18(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, backbone='resnet18', pretrained=pretrained, **kwargs)
    return model

def resnet34(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, backbone='resnet34', pretrained=pretrained, **kwargs)
    return model

def resnet50(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, backbone='resnet50', pretrained=pretrained, **kwargs)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, backbone='resnet101', pretrained=pretrained, **kwargs)
    return model

def resnet152(num_classes, pretrained=False, **kwargs):
    model = RetinaNet(num_classes, backbone='resnet152', pretrained=pretrained, **kwargs)
    return model

def efficientnet_b0_retinanet(num_classes, pretrained=False, **kwargs):
    """
    Constructs a RetinaNet model with an EfficientNet-B0 backbone.
    **kwargs can be used to pass freezing options like `freeze_backbone`, `freeze_fpn`, etc.
    """
    model = RetinaNet(num_classes, backbone='efficientnet-b0', pretrained=pretrained, **kwargs)
    return model