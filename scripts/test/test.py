import torch,torchaudio
print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"torchaudio",torchaudio.__version__)
print("device",torch.device("cuda" if torch.cuda.is_available() else "cpu"))
a=torch.randn(2048,2048,device="cuda")
print("matmul ok:",(a@a).sum().item())