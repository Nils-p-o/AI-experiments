import torch
import time


def generate_noisy_weights(w : torch.Tensor, snr_dB=10):
    rms_signal = torch.sqrt(torch.mean(w**2))
    snr_linear = 10**(snr_dB/20)
    rms_noise = rms_signal/snr_linear
    noise_tensor = torch.randn(w.shape, device=w.device) * rms_noise
    return noise_tensor

def scaleup_weights(w : torch.Tensor, method="noisy_diagonal"):
    match method:
        case "symmetric":
            return w.repeat(2, 2)/2
        case "diagonal": 
            return torch.cat([torch.cat([w,torch.zeros(w.shape)], dim=1), torch.cat([torch.zeros(w.shape), w], dim=1)], dim=0)

        case "noisy_symmetric": # 0 = height, 1 = width
            n1 = generate_noisy_weights(w)
            n2 = generate_noisy_weights(w)
            n12 = torch.cat((torch.cat([n1, -n1], dim=1), torch.cat([n2, -n2], dim=1)), dim=0)
            nw = w.repeat(2, 2)/2 + n12
            return nw
        case "noisy_diagonal":
            nw = torch.zeros(2*w.shape[0], 2*w.shape[1])
            nw[:w.shape[0], :w.shape[1]] = w
            nw[w.shape[0]:, w.shape[1]:] = w
            n1 = generate_noisy_weights(w)
            n2 = generate_noisy_weights(w)
            n12 = torch.cat((torch.cat([n1, -n1], dim=1), torch.cat([n2, -n2], dim=1)), dim=0)
            return nw + n12

def scaleup_bias(b : torch.Tensor):
    return b.repeat(2)


def scaleup_model(model, method="noisy_diagonal"):
    for layer in model.layers: # TODO refer to paper and make sure everything matches up
        layer.mha.q_linear.weight.data = scaleup_weights(layer.mha.q_linear.weight.data, method=method)
        layer.mha.k_linear.weight.data = scaleup_weights(layer.mha.k_linear.weight.data, method=method)
        layer.mha.v_linear.weight.data = scaleup_weights(layer.mha.v_linear.weight.data, method=method)
        layer.mha.o.weight.data = scaleup_weights(layer.mha.o.weight.data, method=method)
        
        layer.mha.q_linear.bias.data = scaleup_bias(layer.mha.q_linear.bias.data)
        layer.mha.k_linear.bias.data = scaleup_bias(layer.mha.k_linear.bias.data)
        layer.mha.v_linear.bias.data = scaleup_bias(layer.mha.v_linear.bias.data)
        layer.mha.o.bias.data = scaleup_bias(layer.mha.o.bias.data)

        layer.ff.linear_in.weight.data = scaleup_weights(layer.ff.linear_in.weight.data, method=method)
        layer.ff.linear_gate.weight.data = scaleup_weights(layer.ff.linear_gate.weight.data, method=method)
        layer.ff.linear_out.weight.data = scaleup_weights(layer.ff.linear_out.weight.data, method=method)

        layer.ff.linear_in.bias.data = scaleup_bias(layer.ff.linear_in.bias.data)
        layer.ff.linear_gate.bias.data = scaleup_bias(layer.ff.linear_gate.bias.data)
        layer.ff.linear_out.bias.data = scaleup_bias(layer.ff.linear_out.bias.data)

        layer.norm1.weight.data = scaleup_weights(layer.norm1.weight.data, method=method)
        layer.norm2.weight.data = scaleup_weights(layer.norm2.weight.data, method=method)
        layer.norm1.bias.data = scaleup_bias(layer.norm1.bias.data)
        layer.norm2.bias.data = scaleup_bias(layer.norm2.bias.data)

    model.input_embedding.weight.data = model.input_embedding.weight.data.repeat(2) # TODO do this right
    model.input_embedding.bias.data = model.input_embedding.bias.data.repeat(2) # TODO do this right
    model.out.weight.data = torch.cat(model.out.weight.data, model.out.weight.data, dim=1) # TODO do this right

    model.norm.weight.data = scaleup_weights(model.norm.weight.data, method=method)



example = torch.randn(512,512)

start_time = time.time_ns()
for _ in range(100):
    a = torch.randn_like(example)
    b = scaleup_weights(a, method="symmetric")
print(f"time symmetric: {(time.time_ns() - start_time)/1e+9}")

start_time = time.time_ns()
for _ in range(100):
    a = torch.randn_like(example)
    b = scaleup_weights(a, method="diagonal")
print(f"time diagonal: {(time.time_ns() - start_time)/1e+9}")

start_time = time.time_ns()
for _ in range(100):
    a = torch.randn_like(example)
    b = scaleup_weights(a, method="noisy_symmetric")
print(f"time noisy_symmetric: {(time.time_ns() - start_time)/1e+9}")

start_time = time.time_ns()
for _ in range(100):
    a = torch.randn_like(example)
    b = scaleup_weights(a, method="noisy_diagonal")
print(f"time noisy_diagonal: {(time.time_ns() - start_time)/1e+9}")
