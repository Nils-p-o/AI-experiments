from training.utils import stablemax, taylor_softmax, custom_cross_entropy
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def check(logits, targets, cce_fn):
    if cce_fn == "stablemax":
        cce_fn = custom_cross_entropy(softmax_fn=stablemax)
    elif cce_fn == "taylor_softmax":
        cce_fn = custom_cross_entropy(softmax_fn=taylor_softmax)
    elif cce_fn == "false":
        cce_fn = nn.CrossEntropyLoss()
    elif cce_fn == "true":
        cce_fn = custom_cross_entropy()
    else:
        raise ValueError(f"Custom cross entropy function {cce_fn} not supported")
    return cce_fn(logits, targets)


logits = torch.tensor([[1, 0.9], [0.09, 0.1]])

targets = torch.tensor([0, 1])

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
scaler_data = list()

for scaler in range(10):
    scaled_logits = logits * scaler
    true_data.append(check(scaled_logits, targets, "true"))
    false_data.append(check(scaled_logits, targets, "false"))
    # stablemax_data.append(check(scaled_logits, targets, "stablemax"))
    # taylor_softmax_data.append(check(scaled_logits, targets, "taylor_softmax"))

plt.plot(true_data, label="true")
plt.plot(false_data, label="false")
# plt.plot(stablemax_data, label="stablemax")
# plt.plot(taylor_softmax_data, label="taylor_softmax")
plt.legend()
plt.show()  # all the same

a = "time"