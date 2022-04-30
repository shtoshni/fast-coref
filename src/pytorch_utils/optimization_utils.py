from torch.optim.lr_scheduler import LambdaLR


def get_inverse_square_root_decay(optimizer, num_warmup_steps=0, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            if num_warmup_steps > 0:
                return (num_warmup_steps / current_step) ** 0.5
            else:
                return (1 / (current_step + 1)) ** 0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)
