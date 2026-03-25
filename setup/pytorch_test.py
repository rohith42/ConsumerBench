import torch

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    # Set device to CUDA
    device = torch.device("cuda")

    # Create two random matrices on the CUDA device
    a = torch.randn(3, 3, device=device)
    b = torch.randn(3, 3, device=device)

    # Perform matrix multiplication
    c = torch.matmul(a, b)

    print("Matrix A:\n", a)
    print("Matrix B:\n", b)
    print("A x B =\n", c)

if __name__ == "__main__":
    main()
