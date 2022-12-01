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