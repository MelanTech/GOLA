from .pipeline import AugmentationPipeline, AugmentationConfig


def build_augmentation_pipeline(augmentation_pipline_config: list):
    pipelines = []
    for augmentation_config in augmentation_pipline_config:
        if augmentation_config['type'] == 'gray_scale':
            from .gray_scale import GrayScaleAugmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   GrayScaleAugmentation(augmentation_config['probability']),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'horizontal_flip':
            from .horizontal_flip import HorizontalFlipAugmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   HorizontalFlipAugmentation(augmentation_config['probability']),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'color_jitter':
            from .color_jitter import ColorJitter

            brightness_factor = augmentation_config['brightness']
            contrast_factor = augmentation_config['contrast']
            saturation_factor = augmentation_config['saturation']

            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   ColorJitter(brightness_factor, contrast_factor, saturation_factor),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'DeiT_3_aug':
            from .deit_3_augmentation import DeiT3Augmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   DeiT3Augmentation(),
                                   augmentation_config.get('joint', True)))

        # Zekai Shao: Add RGB-T data augmentation builder
        elif augmentation_config['type'] == 'color_jitter_mm':
            from .color_jitter import MMColorJitter

            brightness_factor = augmentation_config['brightness']
            contrast_factor = augmentation_config['contrast']
            saturation_factor = augmentation_config['saturation']

            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   MMColorJitter(brightness_factor, contrast_factor, saturation_factor),
                                   augmentation_config.get('joint', True)))
        elif augmentation_config['type'] == 'DeiT_3_aug_mm':
            from .deit_3_augmentation import MMDeiT3Augmentation
            pipelines.append(
                AugmentationConfig(augmentation_config['target'],
                                   MMDeiT3Augmentation(),
                                   augmentation_config.get('joint', True)))
        else:
            raise ValueError(f"Unknown augmentation type: {augmentation_config['type']}")
    return AugmentationPipeline(pipelines)
