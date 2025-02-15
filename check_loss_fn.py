from training.utils import stablemax, taylor_softmax, custom_cross_entropy
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def check(logits, targets, cce_fn):
    if cce_fn == "stablemax":
        cce_fn = custom_cross_entropy(softmax_fn=stablemax)
    elif cce_fn == "taylor_softmax":
        cce_fn = custom_cross_entropy(softmax_fn=taylor_softmax)
    elif cce_fn == "true":
        cce_fn = custom_cross_entropy()
    elif cce_fn == "false":
        cce_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Custom cross entropy function {cce_fn} not supported")
    # print(logits.shape, logits.dtype, targets.shape, targets.dtype)
    return cce_fn(logits, targets)

def get_avg(losses, rng, scaler_max):
    temp = [0 for _ in range(scaler_max)]
    for i in range(rng):
        temp[i] = sum(losses[j] for j in range(i, rng*scaler_max, rng))
    return temp


# cce_fn = "true"
# print("Custom cross entropy function:", cce_fn)
# print(check(logits, targets, cce_fn))  # tensor(1.2039)

# cce_fn = "false"
# print("Custom cross entropy function:", cce_fn)
# print(check(logits, targets, cce_fn))  # tensor(1.2039)
# # f = check(logits, targets, cce_fn)

# cce_fn = "stablemax"
# print("Custom cross entropy function:", cce_fn)
# print(check(logits, targets, cce_fn))  # tensor(1.2039)

# cce_fn = "taylor_softmax"
# print("Custom cross entropy function:", cce_fn)
# print(check(logits, targets, cce_fn))  # tensor(1.2039)

true_data = list()
false_data = list()
stablemax_data = list()
taylor_softmax_data = list()
scaler_max = 10
rng = 10


targets = torch.tensor([0, 1])
targets = targets.repeat(1, 1)

for _ in range(rng):
    logits = torch.randn(1, 10, 2)

    for scaler in range(scaler_max):
        scaled_logits = logits * scaler
        false_data.append(check(scaled_logits, targets, "false"))
        true_data.append(check(scaled_logits, targets, "true"))
        stablemax_data.append(check(scaled_logits, targets, "stablemax"))
        taylor_softmax_data.append(check(scaled_logits, targets, "taylor_softmax"))


true_data = get_avg(true_data, rng, scaler_max)
false_data = get_avg(false_data, rng, scaler_max)
stablemax_data = get_avg(stablemax_data, rng, scaler_max)
taylor_softmax_data = get_avg(taylor_softmax_data, rng, scaler_max)

plt.plot(false_data, label="false") # this is the same as true when logits only have 2 dims
plt.plot(true_data, label="true")
plt.plot(stablemax_data, label="stablemax")
plt.plot(taylor_softmax_data, label="taylor_softmax")
plt.legend()
plt.show()
