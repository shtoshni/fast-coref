import torch


def print_model_info(model):
    """Prints model parameters and their total count"""
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            dims = list(param.data.size())
            local_params = 1
            for dim in dims:
                local_params *= dim
            total_params += local_params
            if not ("lm_encoder." in name):
                print(name, param.data.size())
    print("\nTotal Params:{:.2f} (in millions)".format(total_params / 10**6))


def enough_memory():
    if torch.cuda.is_available():
        memory_in_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        if memory_in_gb > 40:
            return True

    return False


def get_sequence_mask(sequence_len):
    """Returns Sequence Mask.
    sequence_len: Tensor of size (B,) with entries indicating length of seq.
    """
    batch_size = sequence_len.size()[0]
    max_len = torch.max(sequence_len)
    tmp = torch.arange(max_len, device=sequence_len.device).expand(batch_size, max_len)
    return tmp < sequence_len.unsqueeze(1)


def get_span_mask(start_ids, end_ids, max_len):
    tmp = (
        torch.arange(max_len, device=start_ids.device)
        .unsqueeze(0)
        .expand(start_ids.shape[0], -1)
    )
    batch_start_ids = start_ids.unsqueeze(1).expand_as(tmp)
    batch_end_ids = end_ids.unsqueeze(1).expand_as(tmp)
    mask = (tmp >= batch_start_ids).float() * (tmp <= batch_end_ids).float()
    return mask


def check_nan_grad(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        else:
            num_nan = torch.sum(torch.isnan(param.grad.data))
            if num_nan:
                print(name)


def get_l2_norm(model, debug=False):
    total_l2_norm = {"param": 0, "grad": 0}
    param_norm_list = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.data, p=2)
            if torch.isnan(param_norm):
                print("NaN parameter:", name)
            param_norm_list.append((name, param_norm.item()))
            total_l2_norm["param"] += torch.norm(param.data, p=2).item()
            total_l2_norm["grad"] += torch.norm(param.grad.data, p=2).item()
    if debug:
        print("Summation of L2 norm: %.3f" % total_l2_norm["param"])
        # Sort param list by L2 norm
        sorted_param_list = sorted(param_norm_list, key=lambda x: x[1], reverse=True)
        topk_list = sorted_param_list[:5]
        for name, param_norm in topk_list:
            print(
                "Name: %s\tNorm (%%): %.3f\tNorm: %.3f"
                % (name, param_norm * 100 / total_l2_norm["param"], param_norm)
            )

    return total_l2_norm
