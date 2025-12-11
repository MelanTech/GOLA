# Added by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Add support for model profiling

import argparse
import os

import torch

from trackit.core.boot.funcs.main.load_config import load_config
from trackit.models import ModelManager, ModelImplSuggestions
from trackit.models.methods.builder import get_model_build_context
from trackit.models.utils.efficiency_assessment.latency import _test_model_latency_cuda
from trackit.core.runtime.application.default.model_efficiency_assessment import _run_model_flop_count_assessment

def setup_arg_parser():
    parser = argparse.ArgumentParser('Set runtime parameters', add_help=False)
    parser.add_argument('method_name', type=str, help='Method name')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--mixin_config', type=str, action='append')

    return parser


if __name__ == '__main__':
    parser = setup_arg_parser()
    args = parser.parse_args()
    args.root_path = os.path.dirname(os.path.abspath(__file__))
    args.config_path = os.path.join(args.root_path, 'config')

    config = load_config(args)
    device = torch.device(args.device)
    suggestions = ModelImplSuggestions(False, True)

    model_manager = ModelManager(get_model_build_context(config))
    model_instance = model_manager.create(device, suggestions)
    test_data = model_manager.sample_input_data_generator.get(batch_size=1, device=device)

    _run_model_flop_count_assessment(model_manager, device)

    consumed_time = _test_model_latency_cuda(model_instance.model,
                                             test_data,
                                             loops=1000,
                                             warmup_loops=500,
                                             auto_mixed_precision=
                                             config['run']['efficiency_assessment']['latency']['auto_mixed_precision'][
                                                 'enabled'])

    print(f'\nFPS: {1000 / consumed_time}')
