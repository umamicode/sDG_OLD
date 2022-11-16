import torchvision


def get_resnet(name, pretrained=False):
    '''
    resnets = {
        "resnet18": torchvision.models.resnet18(weights=pretrained),
        "resnet50": torchvision.models.resnet50(weights=pretrained),
    }
    '''
    if pretrained == True:
        resnets = {
        "resnet18": torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT),
        "resnet50": torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT),
        }
    elif pretrained == False:
        resnets = {
        "resnet18": torchvision.models.resnet18(weights=None),
        "resnet50": torchvision.models.resnet50(weights=None),
        }
    #[TODO] Solve this - UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
