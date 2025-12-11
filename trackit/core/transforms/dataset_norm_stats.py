import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_STD, OPENAI_CLIP_MEAN


def get_dataset_norm_stats_transform(dataset: str, inplace: bool):
    """
    Returns a torchvision transform object performing normalization
    for the specified dataset.

    Args:
        dataset (str): The dataset for which the normalization
        is required. Options are 'imagenet' or 'openai_clip'.
        inplace (bool): Enable inplace normalization.

    Returns:
        torchvision.transforms.Normalize: A torchvision normalization transform.
    """
    if dataset == 'imagenet':
        return transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=inplace)
    elif dataset == 'openai_clip':
        return transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, inplace=inplace)

    # Zekai Shao: Add RGB-T dataset normalization
    elif dataset == 'mm':
        return transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.449, 0.449, 0.449],
                                    std=[0.229, 0.224, 0.225, 0.226, 0.226, 0.226], inplace=inplace)
    else:
        raise NotImplementedError()


def get_dataset_norm_stats_transform_reversed(dataset: str, inplace: bool):
    """
    Returns a torchvision transform object performing denormalization
    for the specified dataset. This is the inverse of the normalization

    Args:
        dataset (str): The dataset for which the normalization
        is required. Options are 'imagenet' or 'openai_clip'.
        inplace (bool): Enable inplace normalization.

    Returns:
        torchvision.transforms.Normalize: A torchvision normalization transform.
    """
    if dataset == 'imagenet':
        return transforms.Normalize(mean=[-m / s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
                                    std=[1 / s for s in IMAGENET_DEFAULT_STD], inplace=inplace)
    elif dataset == 'openai_clip':
        return transforms.Normalize(mean=[-m / s for m, s in zip(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)],
                                    std=[1 / s for s in OPENAI_CLIP_STD], inplace=inplace)

    # Zekai Shao: Add RGB-T dataset reverse normalization
    elif dataset == 'mm':
        return transforms.Normalize(mean=[-m / s for m, s in zip([0.485, 0.456, 0.406, 0.449, 0.449, 0.449],
                                                                 [0.229, 0.224, 0.225, 0.226, 0.226, 0.226])],
                                    std=[1 / s for s in [0.229, 0.224, 0.225, 0.226, 0.226, 0.226]], inplace=inplace)
    else:
        raise NotImplementedError()
