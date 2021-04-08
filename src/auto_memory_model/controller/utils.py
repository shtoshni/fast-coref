from auto_memory_model.controller import *


def pick_controller(mem_type='unbounded', dataset='litbank', device='cuda', **kwargs):
    if mem_type in ['learned', 'unbounded', 'lru']:
        model = ControllerPredInvalid(mem_type=mem_type, dataset=dataset, device=device, **kwargs)
    else:
        raise NotImplementedError(mem_type)

    return model


