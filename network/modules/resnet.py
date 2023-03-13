import torchvision

def get_resnet(name, pretrained=False):
    
    #Here, pretrained parameter is sent as str. Click Sucks
    if pretrained == 'True':
        resnets = {
        "resnet18": torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT'), # or weights= 'DEFAULT'
        "resnet50": torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT'), # or weights= 'DEFAULT'
        "wideresnet": torchvision.models.wide_resnet101_2(weights='Wide_ResNet101_2_Weights.DEFAULT'),
        "alexnet": torchvision.models.alexnet(weights='AlexNet_Weights.DEFAULT'),
        'vit': torchvision.models.vit_l_32(weights= 'ViT_L_32_Weights.DEFAULT'),
        'regnet': torchvision.models.regnet_y_16gf(weights= 'RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1'),
        'regnet_large': torchvision.models.regnet_y_128gf(weights='IMAGENET1K_SWAG_E2E_V1')
        }
    elif pretrained == 'False':
        resnets = {
        "resnet18": torchvision.models.resnet18(weights= None),
        "resnet50": torchvision.models.resnet50(weights= None),
        "wideresnet": torchvision.models.wide_resnet101_2(weights=None),
        "alexnet": torchvision.models.alexnet(weights=None),
        'vit': torchvision.models.vit_l_32(weights= None),
        'regnet': torchvision.models.regnet_y_16gf(weights= None),
        'regnet_large': torchvision.models.regnet_y_128gf(weights=None)
        }
    
    #[TODO] Solve this - UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
