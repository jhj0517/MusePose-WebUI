import argparse
from omegaconf import OmegaConf
import os
from musepose_inference import MusePoseInference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/test_stage.yaml")
    parser.add_argument("-W", type=int, default=768, help="Width")
    parser.add_argument("-H", type=int, default=768, help="Height")
    parser.add_argument("-L", type=int, default=300, help="video frame length")
    parser.add_argument("-S", type=int, default=48,  help="video slice frame number")
    parser.add_argument("-O", type=int, default=4,   help="video slice overlap frame number")

    parser.add_argument("--cfg",   type=float, default=3.5, help="Classifier free guidance")
    parser.add_argument("--seed",  type=int,   default=99)
    parser.add_argument("--steps", type=int,   default=20, help="DDIM sampling steps")
    parser.add_argument("--fps",   type=int)
    parser.add_argument("--weight_dtype", type=str, default="fp16")
    parser.add_argument('--model_dir', type=str, default=os.path.join("pretrained_weights"), help='Pretrained models directory for MusePose')
    parser.add_argument('--output_dir', type=str, default=os.path.join("assets", "videos"), help='Output directory for the result')

    parser.add_argument("--skip",  type=int,   default=1, help="frame sample rate = (skip+1)")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    musepose_infer = MusePoseInference(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )

    ref_image_path = list(config["test_cases"].keys())[0]
    pose_video_path = config["test_cases"][ref_image_path][0]

    output_file_path, output_demo_file_path = musepose_infer.infer_musepose(
        ref_image_path=ref_image_path,
        pose_video_path=pose_video_path,
        weight_dtype=args.weight_dtype,
        W=args.W,
        H=args.H,
        L=args.L,
        S=args.S,
        O=args.O,
        cfg=args.cfg,
        seed=args.seed,
        steps=args.steps,
        fps=args.fps,
        skip=args.skip
    )

    print(f"{output_file_path} is saved")


if __name__ == "__main__":
    main()
