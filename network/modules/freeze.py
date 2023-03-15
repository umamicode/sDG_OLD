def freeze_(model):
    """Freeze model
    Note that this function does not control BN
    """
    for p in model.parameters():
        p.requires_grad_(False)
        
def unfreeze_(model):
    """Unfreeze model
    Note that this function does not control BN
    """
    for p in model.parameters():
        p.requires_grad_(True)


def freeze(freeze, model):
    '''
    Freeze a Model
    --freeze (Which block to freeze --encoder/heads/all [str])
    --model (which model to freeze [str])
    '''
    if freeze is not None:
        if freeze == "all":
            freeze_(model)
        elif freeze == 'encoder':
            freeze_(model.encoder)
        elif freeze == 'heads':
            freeze_(model.cls_head_src)
            freeze_(model.cls_head_tgt)
            freeze_(model.pro_head)
        else:
            raise ValueError("Please Freeze Either all/encoder/heads")
def unfreeze(unfreeze, model):
    '''
    Unfreeze a Model
    --unfreeze (Which block to unfreeze --encoder/heads/all [str])
    --model (which model to freeze [str])
    '''
    if unfreeze is not None:
        if unfreeze == "all":
            unfreeze_(model)
        elif unfreeze == 'encoder':
            unfreeze_(model.encoder)
        elif unfreeze == 'heads':
            unfreeze_(model.cls_head_src)
            unfreeze_(model.cls_head_tgt)
            unfreeze_(model.pro_head)
        else:
            raise ValueError("Please Unfreeze Either all/encoder/heads")/home/dongkyu/sDG/saved-model/improvegen