import torch

def main():
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create a 3 by 4 2d tensor filled from 1 to 12
    t = torch.arange(1, 13, device=device).reshape(3, 4)
    print(t)
    print(t.dim())
    print(t.shape)
    print(t.size())

if __name__ == "__main__":
    main()
