from safetensors.torch import save_file
import collections
import re

# Check if an object has a state_dict() method
def has_state_dict(obj):
    return hasattr(obj, 'state_dict') and callable(obj.state_dict)

def remove_duplicate(model):
    state_dict = model.state_dict()
    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        ptrs[tensor.data_ptr()].append(name)

    # These are all the pointers of shared tensors.
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    for names in shared_ptrs.values():
        # Removing the keys which are declared as known duplicates on
        # load. This allows to make sure the name which is kept is consistent.
        if model._keys_to_ignore_on_load_missing is not None:
            for name in names:
                matches_pattern = any(re.search(pat, name) for pat in model._keys_to_ignore_on_load_missing)
                if matches_pattern and name in state_dict:
                    del state_dict[name]

        # When not all duplicates have been cleaned, still remove those keys, but put a clear warning. If there is a mismatch transformers will show a warning on loading the model. Applicable when users make connections between tensors at runtime before saving the model.
        found = 0
        for name in names:
            if name in state_dict:
                found += 1
                if found > 1:
                    del state_dict[name]
                    warn_names.add(name)
    if len(warn_names) > 0:
        print(f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading")
    return state_dict


def custom_save(model):
    if has_state_dict(model):
        model.cpu()
        # Safetensors does not allow tensor aliasing (tensors that share memory) so we need to remove duplicated before saving. This is fine on most models and happens mainly on HF hub models.
        state_dict = remove_duplicate(model)
        save_file(state_dict, "model.safetensors")
    else:
        raise "Model object doesn't contain state dict"
