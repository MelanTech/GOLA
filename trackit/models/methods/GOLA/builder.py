from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
from .sample_data_generator import build_sample_input_data_generator


def get_GOLA_build_context(config: dict):
    print('GOLA model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_GOLA_model(config, impl_advice),
                                lambda impl_advice: get_GOLA_build_string(config['model']['type'], impl_advice),
                                build_sample_input_data_generator(config))


def build_GOLA_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    if model_type == 'dinov2':
        if model_impl_suggestions.optimize_for_inference:
            from .gola_full_finetune import GOLABaseline_DINOv2
            model = GOLABaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])

            # Zekai Shao: Use the weight loading function of the inference model
            # to load the weights after merging the LoRA parameters
            if config['model']['eval']:
                for path in config['model']['weight_path']:
                    model.load_state_dict_from_file(path)
        else:
            from .gola import GOLA_DINOv2
            model = GOLA_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'],
                                 model_config['lora']['r'], model_config['lora']['alpha'],
                                 model_config['lora']['dropout'], model_config['lora']['use_rslora'])
    elif model_type == 'dinov2_full_finetune':
        from .gola_full_finetune import GOLABaseline_DINOv2
        model = GOLABaseline_DINOv2(backbone, common_config['template_feat_size'], common_config['search_region_feat_size'])
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    return model


def get_GOLA_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'GOLA'
    if 'full_finetune' in model_type:
        build_string += '_full_finetune'
    else:
        if model_impl_suggestions.optimize_for_inference:
            build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string
