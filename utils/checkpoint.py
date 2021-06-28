import json
import logging
import os
import signal
from pathlib import Path
import glob

import torch
from torch.nn.parameter import Parameter

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

#print (net)
#size mismatch for middle_feature_extractor.middle_conv.0.weight: copying a param with shape torch.Size([3456, 64]) from checkpoint, the shape in current model is torch.Size([27, 128, 64]).                                                                                   size mismatch for middle_feature_extractor.middle_conv.2.weight: copying a param with shape torch.Size([192, 64]) from checkpoint, the shape in current model is torch.Size([3, 64, 64]).
#size mismatch for middle_feature_extractor.middle_conv.4.weight: copying a param with shape torch.Size([1728, 64]) from checkpoint, the shape in current model is torch.Size([27, 64, 64]).
#size mismatch for middle_feature_extractor.middle_conv.6.weight: copying a param with shape torch.Size([1728, 64]) from checkpoint, the shape in current model is torch.Size([27, 64, 64]).
#size mismatch for middle_feature_extractor.middle_conv.8.weight: copying a param with shape torch.Size([192, 64]) from checkpoint, the shape in current model is torch.Size([3, 64, 64]).

def load_state_dict(net, state_dict, strict=True):
    own_state = net.state_dict()

    print("num. para. in model:",len(own_state))
    #print(own_state)
    print("num. para. in chkpt:",len(state_dict))
    #print(state_dict)
    for name, param in state_dict.items():
        name = name.replace("runningMean","running_mean")
        name = name.replace("runningVar","running_var")
        #print(name)
        if name in own_state:
            if isinstance(param, Parameter):
                param = param.data
            try:
                if name == "middle_feature_extractor.middle_conv.0.weight":
                    param = param.view(27, 128, 64)
                if name == "middle_feature_extractor.middle_conv.2.weight":
                    param = param.view(3, 64, 64)
                if name == "middle_feature_extractor.middle_conv.4.weight":
                    param = param.view(27, 64, 64)
                if name == "middle_feature_extractor.middle_conv.6.weight":
                    param = param.view(27, 64, 64)
                if name == "middle_feature_extractor.middle_conv.8.weight":
                    param = param.view(3, 64, 64)
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, '
                                   'whose dimensions in the model are {} and '
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def best_checkpoint(model_dir, model_name):
    """return path of best checkpoint in a model_dir
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model_name: name of your model. we find ckpts by name
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """
    filenames = glob.glob(os.path.join(model_dir,model_name+".*"))
    best = 0
    best_ckpt = ""
    for fname in filenames:
        tmp = float(fname.replace(os.path.join(model_dir,model_name+"."),""))
        if best < tmp:
            best = tmp
            best_ckpt = fname
    return best_ckpt

def latest_checkpoint(model_dir, model_name):
    """return path of latest checkpoint in a model_dir
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model_name: name of your model. we find ckpts by name
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """
    ckpt_info_path = Path(model_dir) / "checkpoints.json"
    if not ckpt_info_path.is_file():
        return None
    with open(ckpt_info_path, 'r') as f:
        ckpt_dict = json.loads(f.read())
    if model_name not in ckpt_dict['latest_ckpt']:
        return None
    latest_ckpt = ckpt_dict['latest_ckpt'][model_name]
    ckpt_file_name = Path(model_dir) / latest_ckpt
    if not ckpt_file_name.is_file():
        return None

    return str(ckpt_file_name)

def _ordered_unique(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def save(model_dir,
         model,
         model_name,
         global_step,
         max_to_keep=8,
         keep_latest=True):
    """save a model into model_dir.
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model: torch.nn.Module instance.
        model_name: name of your model. we find ckpts by name
        global_step: int, indicate current global step.
        max_to_keep: int, maximum checkpoints to keep.
        keep_latest: bool, if True and there are too much ckpts,
            will delete oldest ckpt. else will delete ckpt which has
            smallest global step.
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """

    # prevent save incomplete checkpoint due to key interrupt
    with DelayedKeyboardInterrupt():
        ckpt_info_path = Path(model_dir) / "checkpoints.json"
        ckpt_filename = "{}-{}.tckpt".format(model_name, global_step)
        ckpt_path = Path(model_dir) / ckpt_filename
        if not ckpt_info_path.is_file():
            ckpt_info_dict = {'latest_ckpt': {}, 'all_ckpts': {}}
        else:
            with open(ckpt_info_path, 'r') as f:
                ckpt_info_dict = json.loads(f.read())
        ckpt_info_dict['latest_ckpt'][model_name] = ckpt_filename
        if model_name in ckpt_info_dict['all_ckpts']:
            ckpt_info_dict['all_ckpts'][model_name].append(ckpt_filename)
        else:
            ckpt_info_dict['all_ckpts'][model_name] = [ckpt_filename]
        all_ckpts = ckpt_info_dict['all_ckpts'][model_name]

        torch.save(model.state_dict(), ckpt_path)
        # check ckpt in all_ckpts is exist, if not, delete it from all_ckpts
        all_ckpts_checked = []
        for ckpt in all_ckpts:
            ckpt_path_uncheck = Path(model_dir) / ckpt
            if ckpt_path_uncheck.is_file():
                all_ckpts_checked.append(str(ckpt_path_uncheck))
        all_ckpts = all_ckpts_checked
        if len(all_ckpts) > max_to_keep:
            if keep_latest:
                ckpt_to_delete = all_ckpts.pop(0)
            else:
                # delete smallest step
                get_step = lambda name: int(name.split('.')[0].split('-')[1])
                min_step = min([get_step(name) for name in all_ckpts])
                ckpt_to_delete = "{}-{}.tckpt".format(model_name, min_step)
                all_ckpts.remove(ckpt_to_delete)
            #os.remove(str(Path(model_dir) / ckpt_to_delete))
            try:
                os.remove(ckpt_to_delete)
            except FileNotFoundError:
                print(ckpt_to_delete)

        all_ckpts_filename = _ordered_unique([Path(f).name for f in all_ckpts])
        ckpt_info_dict['all_ckpts'][model_name] = all_ckpts_filename
        with open(ckpt_info_path, 'w') as f:
            f.write(json.dumps(ckpt_info_dict, indent=2))


def restore(ckpt_path, model, eval=False):
    print("Try to Restore parameters from {}".format(ckpt_path))
    if not Path(ckpt_path).is_file():
        raise ValueError("checkpoint {} not exist.".format(ckpt_path))
    #model.load_state_dict(torch.load(ckpt_path))
    if eval:
        load_state_dict(model, torch.load(ckpt_path), strict=False)
    else:
        model.load_state_dict(torch.load(ckpt_path))
    print("Restoring parameters from {}".format(ckpt_path))


def _check_model_names(models):
    model_names = []
    for model in models:
        if not hasattr(model, "name"):
            raise ValueError("models must have name attr")
        model_names.append(model.name)
    if len(model_names) != len(set(model_names)):
        raise ValueError("models must have unique name: {}".format(
            ", ".join(model_names)))


def _get_name_to_model_map(models):
    if isinstance(models, dict):
        name_to_model = {name: m for name, m in models.items()}
    else:
        _check_model_names(models)
        name_to_model = {m.name: m for m in models}
    return name_to_model


def try_restore_latest_checkpoints(model_dir, models, eval=False):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_ckpt = latest_checkpoint(model_dir, name)
        if latest_ckpt is not None:
            restore(latest_ckpt, model, eval)

def restore_latest_checkpoints(model_dir, models):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        latest_ckpt = latest_checkpoint(model_dir, name)
        if latest_ckpt is not None:
            restore(latest_ckpt, model)
        else:
            raise ValueError("model {}\'s ckpt isn't exist".format(name))

def restore_models(model_dir, models, global_step):
    name_to_model = _get_name_to_model_map(models)
    for name, model in name_to_model.items():
        ckpt_filename = "{}-{}.tckpt".format(name, global_step)
        ckpt_path = model_dir + "/" + ckpt_filename
        restore(ckpt_path, model)


def save_models(model_dir,
                models,
                global_step,
                max_to_keep=15,
                keep_latest=True):
    with DelayedKeyboardInterrupt():
        name_to_model = _get_name_to_model_map(models)
        for name, model in name_to_model.items():
            save(model_dir, model, name, global_step, max_to_keep, keep_latest)

def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir
