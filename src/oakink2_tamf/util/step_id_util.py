def handle_step_epoch(epoch_id, freq):
    if epoch_id == 0:
        return 0
    else:
        return epoch_id // freq + 1
