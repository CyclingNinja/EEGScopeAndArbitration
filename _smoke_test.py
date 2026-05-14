import torch
from eeg_win_stack.models import ModelFactory

N_CHANNELS = 21
N_CLASSES = 2
WINDOW = 6000
BATCH = 2

def check(name, model, x):
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"  {name}: input {tuple(x.shape)} -> output {tuple(out.shape)}  OK")

print("Registry:", ModelFactory.available())
assert ModelFactory.available() == [
    'deep4', 'eegnetv1', 'eegnetv4', 'eegresnet', 'hybridnet', 'hybridnet_1',
    'shallow_smac', 'sleep2018', 'sleep2020', 'tcn', 'tcn_1', 'tidnet', 'usleep', 'vit',
], "unexpected registry contents"

x = torch.zeros(BATCH, N_CHANNELS, WINDOW)

print("\ntcn_1")
tcn = ModelFactory.create(
    'tcn_1', n_channels=N_CHANNELS, n_classes=N_CLASSES,
    input_window_samples=WINDOW,
    n_blocks=5, n_filters=55, kernel_size=12,
    drop_prob=0.1, add_log_softmax=False, last_layer_type='conv',
)
check('tcn_1', tcn, x)

print("\nvit")
vit = ModelFactory.create(
    'vit', n_channels=N_CHANNELS, n_classes=N_CLASSES,
    input_window_samples=WINDOW,
    patch_size=100, dim=64, depth=2, heads=4, mlp_dim=128,
    dropout=0.1, emb_dropout=0.1,
)
check('vit', vit, x)

print("\nhybridnet_1")
hybrid = ModelFactory.create(
    'hybridnet_1', n_channels=N_CHANNELS, n_classes=N_CLASSES,
    input_window_samples=WINDOW,
)
check('hybridnet_1', hybrid, x)

print("\nAll tests passed.")
