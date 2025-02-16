import torch


def scaleup_weights(w : torch.Tensor, method="noisy_diagonal"):
    match method:
        case "symmetric":
            return w.repeat(2, 2)/2
        case "diagonal": # TODO check which is faster
            nw = torch.zeros(2*w.shape[0], 2*w.shape[1])
            nw[:w.shape[0], :w.shape[1]] = w
            nw[w.shape[0]:, w.shape[1]:] = w
            return nw
            # nw = torch.cat([torch.cat([w,torch.zeros(w.shape)], dim=1), torch.cat([torch.zeros(w.shape), w], dim=1)], dim=0)

        case "noisy_symmetric": # TODO which dim is which???
            n1 = torch.randn(w.shape[0], w.shape[1]) * 0.1 # TODO is this signal to noise ratio? (should be 10dB)
            n2 = torch.randn(w.shape[0], w.shape[1]) * 0.1
            n12 = torch.cat(torch.cat([n1, -n1], dim=1), torch.cat([n2, -n2], dim=1), dim=0)
            nw = w.repeat(2, 2)/2 + n12
            return nw
        case "noisy_diagonal":
            nw = torch.zeros(2*w.shape[0], 2*w.shape[1])
            nw[:w.shape[0], :w.shape[1]] = w
            nw[w.shape[0]:, w.shape[1]:] = w
            n1 = torch.randn(w.shape[0], w.shape[1]) * 0.1
            n2 = torch.randn(w.shape[0], w.shape[1]) * 0.1
            n12 = torch.cat(torch.cat([n1, -n1], dim=1), torch.cat([n2, -n2], dim=1), dim=0)
            return nw + n12

def scaleup_bias(b : torch.Tensor, method="noisy"):
    match method:
        case "symmetric":
            return b.repeat(2)
        case "noisy":
            n = torch.randn(b.shape) * 0.1
            return b + n



print(torch.randn(2,3))