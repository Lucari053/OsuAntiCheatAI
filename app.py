import gradio as gr
import os

from infer import CheatDetectorInfer, LoadReplayOutput

def inverser_texte(texte):
    return texte[::-1]


class GradioModel:
    def __init__(self):
        self.detector = None

    def load_model(self, checkpoint_path: str):
        
        if not os.path.exists(checkpoint_path): return "Can't find path"
        self.detector = CheatDetectorInfer(checkpoint_path)

        return "Loaded"
    
    def infer(self, replay: str, beatmap: str):
        if not self.detector:
            return "", "Please load model before"
        
        try:
            success = self.detector.load_replay(replay)
        except Exception as e:
            print(e)
            success = LoadReplayOutput.FAIL
        
        if success == LoadReplayOutput.FAIL:
            return "", f"Replay: {success.value}"
        
        if success == LoadReplayOutput.NEED_BEATMAP:
            if not beatmap:
                return "", f"Replay: {success.value}"
            try:
                self.detector.load_beatmap(beatmap)
            except Exception as e:
                print(e)
                return "", "Can't Load beatmap"
        
        bcheat, prob = self.detector.infer()

        return f"Cheat: {bcheat}, Prob: {prob}", "Success"

model = GradioModel()

with gr.Blocks(theme=gr.themes.Default()) as app:
    gr.Markdown("# Osu Cheat Detector")
    
    text_checkpoint_path = gr.Textbox(value="out/checkpoints/best.pt", label="Checkpoint path")

    with gr.Row():
        with gr.Column():
            btn_load_model = gr.Button("Load Model", variant="primary")
        with gr.Column():
            txt_load_status = gr.Textbox(value="None", label="Status")


    with gr.Row():
        with gr.Column():
            txt_replay  = gr.Textbox(label=".osr path OR replay_link OR score_id")
            txt_beatmap = gr.Textbox(label=".osu beatmap, optional if valid beatmap by replay")
            btn_predict = gr.Button("Prediction", variant="primary")
        
        with gr.Column():
            txt_pred = gr.Textbox(label="Result")
            txt_log  = gr.Textbox(label="log")

    btn_load_model.click(fn=model.load_model, inputs=text_checkpoint_path, outputs=txt_load_status)
    btn_predict.click(fn=model.infer, inputs=[txt_replay, txt_beatmap], outputs=[txt_pred, txt_log])

app.queue().launch()