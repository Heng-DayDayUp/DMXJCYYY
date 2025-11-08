# utils.py
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch
    }
    if extra:
        state.update(extra)
    torch.save(state, str(path))


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None and state.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler is not None and state.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state


def plot_train_curve(losses, out_path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('steps')
    plt.ylabel('train_loss')
    plt.title('Training Loss')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path))
    plt.close()


def save_vocab(chars, path):
    # chars: list of characters, index -> char mapping order assumed
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chars, f, ensure_ascii=False, indent=2)


def load_vocab(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
