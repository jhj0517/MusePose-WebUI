import gradio as gr
import argparse
import os

from musepose_inference import MusePoseInference
from pose_align import PoseAlignmentInference
from downloading_weights import download_models


class App:
    def __init__(self, args):
        self.pose_alignment_infer = PoseAlignmentInference(
            model_dir=args.model_dir,
            output_dir=args.output_dir
        )
        self.musepose_infer = MusePoseInference(
            model_dir=args.model_dir,
            output_dir=args.output_dir
        )
        if not args.disable_model_download_at_start:
            download_models(model_dir=args.model_dir)

    def musepose_demo(self):
        with gr.Blocks() as demo:
            md_header = self.header()
            with gr.Tabs():
                with gr.TabItem('Step1: Pose Alignment'):
                    with gr.Row():
                        with gr.Column(scale=3):
                            img_input = gr.Image(label="Input Image here", type="filepath", scale=5)
                            vid_dance_input = gr.Video(label="Input Dance Video", scale=5)
                        with gr.Column(scale=3):
                            vid_dance_output = gr.Video(label="Aligned pose output will be displayed here", scale=5)
                            vid_dance_output_demo = gr.Video(label="Output demo video will be displayed here", scale=5)
                        with gr.Column(scale=3):
                            with gr.Column():
                                nb_detect_resolution = gr.Number(label="Detect Resolution", value=512, precision=0)
                                nb_image_resolution = gr.Number(label="Image Resolution.", value=720, precision=0)
                                nb_align_frame = gr.Number(label="Align Frame", value=0, precision=0)
                                nb_max_frame = gr.Number(label="Max Frame", value=300, precision=0)

                            with gr.Row():
                                btn_align_pose = gr.Button("ALIGN POSE", variant="primary")
                    with gr.Column():
                        examples = [
                            [os.path.join("assets", "videos", "dance.mp4"), os.path.join("assets", "images", "ref.png"),
                             512, 720, 0, 300]]
                        ex_step1 = gr.Examples(examples=examples,
                                               inputs=[vid_dance_input, img_input, nb_detect_resolution,
                                                       nb_image_resolution, nb_align_frame, nb_max_frame],
                                               outputs=[vid_dance_output, vid_dance_output_demo],
                                               fn=self.pose_alignment_infer.align_pose,
                                               cache_examples="lazy")

                btn_align_pose.click(fn=self.pose_alignment_infer.align_pose,
                                     inputs=[vid_dance_input, img_input, nb_detect_resolution, nb_image_resolution,
                                             nb_align_frame, nb_max_frame],
                                     outputs=[vid_dance_output, vid_dance_output_demo])

                with gr.TabItem('Step2: MusePose Inference'):
                    with gr.Row():
                        with gr.Column(scale=3):
                            img_input = gr.Image(label="Input Image here", type="filepath", scale=5)
                            vid_pose_input = gr.Video(label="Input Aligned Pose Video here", scale=5)
                        with gr.Column(scale=3):
                            vid_output = gr.Video(label="Output Video will be displayed here", scale=5)
                            vid_output_demo = gr.Video(label="Output demo video will be displayed here", scale=5)

                        with gr.Column(scale=3):
                            with gr.Column():
                                weight_dtype = gr.Dropdown(label="Compute Type", choices=["fp16", "fp32"],
                                                           value="fp16")
                                nb_width = gr.Number(label="Width.", value=512, precision=0)
                                nb_height = gr.Number(label="Height.", value=512, precision=0)
                                nb_video_frame_length = gr.Number(label="Video Frame Length", value=300, precision=0)
                                nb_video_slice_frame_length = gr.Number(label="Video Slice Frame Number ", value=48,
                                                                        precision=0)
                                nb_video_slice_overlap_frame_number = gr.Number(
                                    label="Video Slice Overlap Frame Number", value=4, precision=0)
                                nb_cfg = gr.Number(label="CFG (Classifier Free Guidance)", value=3.5, precision=0)
                                nb_seed = gr.Number(label="Seed", value=99, precision=0)
                                nb_steps = gr.Number(label="DDIM Sampling Steps", value=20, precision=0)
                                nb_fps = gr.Number(label="FPS (Frames Per Second) ", value=-1, precision=0,
                                                   info="Set to '-1' to use same FPS with pose's")
                                nb_skip = gr.Number(label="SKIP (Frame Sample Rate = SKIP+1)", value=1, precision=0)

                            with gr.Row():
                                btn_generate = gr.Button("GENERATE", variant="primary")

                    with gr.Column():
                        examples = [
                            [os.path.join("assets", "images", "ref.png"), os.path.join("assets", "videos", "pose.mp4"),
                             "fp16", 512, 512, 300, 48, 4, 3.5, 99, 20, -1, 1]]
                        ex_step2 = gr.Examples(examples=examples,
                                               inputs=[img_input, vid_pose_input, weight_dtype, nb_width, nb_height,
                                                       nb_video_frame_length, nb_video_slice_frame_length,
                                                       nb_video_slice_overlap_frame_number, nb_cfg, nb_seed, nb_steps,
                                                       nb_fps, nb_skip],
                                               outputs=[vid_output, vid_output_demo],
                                               fn=self.musepose_infer.infer_musepose,
                                               cache_examples="lazy")

                btn_generate.click(fn=self.musepose_infer.infer_musepose,
                                   inputs=[img_input, vid_pose_input, weight_dtype, nb_width, nb_height,
                                           nb_video_frame_length, nb_video_slice_frame_length,
                                           nb_video_slice_overlap_frame_number, nb_cfg, nb_seed, nb_steps, nb_fps,
                                           nb_skip],
                                   outputs=[vid_output, vid_output_demo])
        return demo

    @staticmethod
    def header():
        header = gr.HTML(
            """
            <style>
            p, li {
                font-size: 16px;
            }
            </style>

            <h2>Gradio demo for <a href="https://github.com/TMElyralab/MusePose">MusePose</a></h2>

            <p>Demo list you can try in other environment:</p>
            <ul>
                <li><a href="https://github.com/jhj0517/MusePose-WebUI"><strong>MusePose WebUI</strong></a> (This repository, you can try in local)</li>
                <li><a href="https://github.com/jhj0517/stable-diffusion-webui-MusePose.git"><strong>stable-diffusion-webui-MusePose</strong></a> (SD WebUI extension)</li>
                <li><a href="https://github.com/TMElyralab/Comfyui-MusePose"><strong>Comfyui-MusePose</strong></a> (ComfyUI custom node)</li>
            </ul>
            """
        )
        return header

    def launch(self):
        demo = self.musepose_demo()
        demo.queue().launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.path.join("pretrained_weights"), help='Pretrained models directory for MusePose')
    parser.add_argument('--output_dir', type=str, default=os.path.join("outputs"), help='Output directory for the result')
    parser.add_argument('--disable_model_download_at_start', type=bool, default=False, nargs='?', const=True, help='Disable model download at start or not')
    args = parser.parse_args()

    app = App(args=args)
    app.launch()