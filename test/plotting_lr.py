import math
import matplotlib.pyplot as plt

warmup_steps = 100
t_0 = 1000
t_mult = 1.5
lr_mult = 0.5
learning_rate = 2e-4
current_peak_lr = learning_rate


def lr_lambda(current_step):
    min_lr = 1e-6

    current_cycle_step = current_step
    cycle_nr = 0
    for _ in range(50):
        t_curr = t_0 * (t_mult ** cycle_nr)
        if current_cycle_step > t_curr:
            current_cycle_step -= t_curr
            cycle_nr += 1
        else:
            break
    
    current_peak_lr = learning_rate * (lr_mult ** cycle_nr)
    

    if current_cycle_step < warmup_steps: # Linear warmup
        return (current_peak_lr - min_lr) * float(current_cycle_step) / float(max(1, warmup_steps)) + min_lr

    if current_cycle_step >= warmup_steps and current_cycle_step <= t_curr:
        progress = float(current_cycle_step - warmup_steps) / float(max(1, t_curr - warmup_steps))
        return current_peak_lr * 0.5 * (math.cos(math.pi * progress) + 1) + 1e-6

    # # Calculate current restart cycle
    # cycle = math.floor((1 + math.log((current_step - warmup_steps)/ t_0 * (t_mult -1) + 1, t_mult)))
    # t_curr = t_0 * (t_mult ** (cycle - 1) - 1) / (t_mult - 1)
    # t_i = t_0 * (t_mult ** (cycle -1))
    # progress = float(current_step - warmup_steps - t_curr) / float(max(1, t_i))

    # #Crucial change: Multiply by current_peak_lr
    # return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress)) * (current_peak_lr / learning_rate) # Scale by initial lr
    return 1


scheduler = [lr_lambda(i) for i in range(10000)]
plt.plot(scheduler)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()