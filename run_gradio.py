from harmonai_tools.interface.gradio import create_ui
import torch

def main():
    torch.manual_seed(42)
    interface = create_ui()
    interface.launch(share=True)

if __name__ == "__main__":
    main()