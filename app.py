import gradio as gr
import argparse
import os

from musepose_inference import MusePoseInference
from pose_align import PoseAlignmentInference
from downloading_weights import download_models


class App:
    def __init__(self, args):
        self.args = args
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

    @staticmethod
    def on_step1_complete(input_img: str, input_pose_vid: str):

        return [gr.Image(label="Input Image", value=input_img, type="filepath", scale=5),
                gr.Video(label="Input Aligned Pose Video", value=input_pose_vid, scale=5)]

    def musepose_demo(self):
        with gr.Blocks() as demo:
            md_header = self.header()
            with gr.Tabs():
                with gr.TabItem('Step1: Pose Alignment'):
                    with gr.Row():
                        with gr.Column(scale=3):
                            img_pose_input = gr.Image(label="Input Image", type="filepath", scale=5)
                            vid_dance_input = gr.Video(label="Input Dance Video", scale=5)
                        with gr.Column(scale=3):
                            vid_dance_output = gr.Video(label="Aligned Pose Output", scale=5, interactive=False)
                            vid_dance_output_demo = gr.Video(label="Aligned Pose Output Demo", scale=5)
                        with gr.Column(scale=3):
                            with gr.Column():
                                nb_detect_resolution = gr.Number(label="Detect Resolution", value=512, precision=0)
                                nb_image_resolution = gr.Number(label="Image Resolution.", value=720, precision=0)
                                nb_align_frame = gr.Number(label="Align Frame", value=0, precision=0)
                                nb_max_frame = gr.Number(label="Max Frame", value=300, precision=0)

                            with gr.Row():
                                btn_align_pose = gr.Button("ALIGN POSE", variant="primary")

                btn_align_pose.click(fn=self.pose_alignment_infer.align_pose,
                                     inputs=[vid_dance_input, img_pose_input, nb_detect_resolution, nb_image_resolution,
                                             nb_align_frame, nb_max_frame],
                                     outputs=[vid_dance_output, vid_dance_output_demo])

                with gr.TabItem('Step2: MusePose Inference'):
                    with gr.Row():
                        with gr.Column(scale=3):
                            img_musepose_input = gr.Image(label="Input Image", type="filepath", scale=5)
                            vid_pose_input = gr.Video(label="Input Aligned Pose Video", scale=5)
                        with gr.Column(scale=3):
                            vid_output = gr.Video(label="MusePose Output", scale=5)
                            vid_output_demo = gr.Video(label="MusePose Output Demo", scale=5)

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

                btn_generate.click(fn=self.musepose_infer.infer_musepose,
                                   inputs=[img_musepose_input, vid_pose_input, weight_dtype, nb_width, nb_height,
                                           nb_video_frame_length, nb_video_slice_frame_length,
                                           nb_video_slice_overlap_frame_number, nb_cfg, nb_seed, nb_steps, nb_fps,
                                           nb_skip],
                                   outputs=[vid_output, vid_output_demo])
                vid_dance_output.change(fn=self.on_step1_complete,
                                        inputs=[img_pose_input, vid_dance_output],
                                        outputs=[img_musepose_input, vid_pose_input])

        return demo

    @staticmethod
    def header():
        header = gr.HTML(
            """
            <h2><a href="https://github.com/jhj0517/MusePose-WebUI">MusePose WebUI</a></h2>
            """
        )
        return header

    def launch(self):
        demo = self.musepose_demo()
        demo.queue().launch(
            share=self.args.share
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.path.join("pretrained_weights"), help='Pretrained models directory for MusePose')
    parser.add_argument('--output_dir', type=str, default=os.path.join("outputs"), help='Output directory for the result')
    parser.add_argument('--disable_model_download_at_start', type=bool, default=False, nargs='?', const=True, help='Disable model download at start or not')
    parser.add_argument('--share', type=bool, default=False, nargs='?', const=True, help='Gradio makes sharable link if it is true')
    args = parser.parse_args()

    app = App(args=args)
    app.launch()