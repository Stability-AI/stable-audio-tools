import argparse
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to the zero checkpoint")
    parser.add_argument("--output_path", type=str, help="Path to the output checkpoint", default="lightning_model.pt")
    args = parser.parse_args()

    # lightning deepspeed has saved a directory instead of a file
    save_path = args.save_path
    output_path = args.output_path
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)