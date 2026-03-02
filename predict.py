import os
import json
import shutil
import mimetypes
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Workflow file
api_json_file = "workflow_api.json"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Handle weights based on the workflow
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=["GFPGANv1.4.pth"],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(self, input_file: Path, filename: str = "image.png"):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

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
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "input_img")
            self.handle_input_file(image, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            image_filename=image_filename,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return [Path(p) for p in optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )]
