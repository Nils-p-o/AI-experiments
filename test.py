import torch
import torch.version

# print(torch.__version__)
# print(torch.version.cuda)

# print(torch.backends.cuda.is_flash_attention_available())
# torch.backends.cuda.enable_flash_sdp(True)

# # print(torch.nn.attention.sdpa_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False, enable_cudnn=False))
# with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH, torch.nn.attention.SDPBackend.CUDNN_ATTENTION], set_priority=True):
#     a = torch.nn.functional.scaled_dot_product_attention(
#         torch.rand(2, 3, 4), torch.rand(2, 3, 4), torch.rand(2, 3, 4))

a = torch.rand(2)
a = a.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
print(a.shape)
a = a.repeat_interleave(2, dim=0)

print(a.shape)
# print(a)
from torch_lr_finder import LRFinder
from transformer_arch.LLaMa import LLaMa
from training.utils import (
    count_parameters,
    ShakespeareDataModule,
    download_and_split_shakespeare,
    custom_cross_entropy,
    stablemax,
    taylor_softmax,
)

if __name__ == "__main__":
    data_module = ShakespeareDataModule(
        train_file="train.txt",
        val_file="val.txt",
        test_file="test.txt",
        seq_len=256,
        batch_size=32,
    )
    data_module.setup()
    vocab_size = data_module.get_vocab_size()

    model = LLaMa(
        d_model=512,
        nhead=8,
        num_layers=6,
        d_ff=None,
        groups=4,
        dropout=0.1,
        vocab_size=vocab_size,
        seq_len=256,
    )

    loss_fns = [                                # batch_size=32, seq_len=256, d_model=512, nhead=8, num_layers=6
        torch.nn.CrossEntropyLoss(),            #~9.11e-4
        custom_cross_entropy(),                 #~1.23e-2
        custom_cross_entropy(softmax_fn=stablemax), #~2.31e-3 didn't diverge
        custom_cross_entropy(softmax_fn=taylor_softmax), #~8.50e-3 didn't diverge
    ]
    for loss_fn in loss_fns:
        criterion = loss_fn
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

        lr_finder = LRFinder(model, optimizer, criterion, device="cpu")

        lr_finder.range_test(data_module.train_dataloader(), end_lr=10, num_iter=100)
        lr_finder.plot()
        lr_finder.reset()
