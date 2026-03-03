import os
import json
import shutil
import mimetypes
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "/opt/ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Workflow file
api_json_file = "workflow_api.json"

class Predictor(BasePredictor):
    def setup(self):
        # Setup custom node configurations
        self._setup_custom_node_configs()

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Handle weights based on the workflow
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "GFPGANv1.4.pth",
                "models/buffalo_l",
                "parsing_parsenet.pth",
                "vit-base-nsfw-detector",
                "detection_Resnet50_Final.pth",
            ],
        )

    def _setup_custom_node_configs(self):
        config_files = {
            "custom_node_configs/was_suite_config.json": "/opt/ComfyUI/custom_nodes/was-node-suite-comfyui/was_suite_config.json",
            "custom_node_configs/rgthree_config.json": "/opt/ComfyUI/custom_nodes/rgthree-comfy/rgthree_config.json",
            "custom_node_configs/comfy.settings.json": "/opt/ComfyUI/user/default/comfy.settings.json",
        }
        for src, dest in config_files.items():
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                print(f"[Cropface] Copying {src} to {dest}")
                shutil.copy(src, dest)

    def handle_input_file(self, input_file: Path, filename: str = "image.png"):
        target_path = os.path.join(INPUT_DIR, filename)
        # Always save as PNG
        img = Image.open(input_file)
        img.save(target_path, "PNG")
        print(f"[Cropface] Image saved as PNG: {filename}")

    def update_workflow(self, workflow, **kwargs):
        # Map input image to LoadImage node (2)
        if kwargs.get("image_filename"):
            workflow["2"]["inputs"]["image"] = kwargs["image_filename"]

    def predict(
        self,
        image: Path = Input(description="Imagen a procesar (recorte de rostro y restauración)", default=None),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> List[Path]:
        """Ejecuta el workflow de ComfyUI para recorte y restauración facial"""
        print("[Cropface] Starting prediction...")
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        image_filename = None
        if image:
            print(f"[Cropface] Processing image: {image}")
            # Always use image.png as filename
            image_filename = "image.png"
            self.handle_input_file(image, image_filename)
            print(f"[Cropface] Image saved as: {image_filename}")

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        print(f"[Cropface] Workflow loaded, updating with image: {image_filename}")

        self.update_workflow(
            workflow,
            image_filename=image_filename,
        )

        print(f"[Cropface] Full workflow: {json.dumps(workflow, indent=2)}")
        
        wf = self.comfyUI.load_workflow(workflow)
        print("[Cropface] Workflow loaded, connecting to ComfyUI...")
        self.comfyUI.connect()
        print("[Cropface] Running workflow...")
        self.comfyUI.run_workflow(wf)
        print("[Cropface] Workflow completed, getting output files...")
        
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        print(f"[Cropface] Output files: {output_files}")

        return [Path(p) for p in optimise_images.optimise_image_files(
            output_format, output_quality, output_files
        )]
