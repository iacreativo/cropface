from custom_node_helper import CustomNodeHelper


class ComfyUI_Reactor(CustomNodeHelper):
    facedetection_weights = {
        "retinaface_resnet50": "detection_Resnet50_Final.pth",
        "retinaface_mobile0.25": "detection_mobilenet0.25_Final.pth",
        "YOLOv5l": "yolov5l-face.pth",
        "YOLOv5n": "yolov5n-face.pth",
    }

    @staticmethod
    def add_weights(weights_to_download, node):
        if node.is_type_in(
            [
                "ReActorFaceSwap",
                "ReActorLoadFaceModel",
                "ReActorSaveFaceModel",
                "ReActorFaceSwapOpt",
                "ReActorRestoreFace",
            ]
        ):
            weights_to_download.append("models/buffalo_l")
            weights_to_download.append("parsing_parsenet.pth")
            weights_to_download.append("vit-base-nsfw-detector")

            if node.has_input("facedetection"):
                facedetection_model = node.input("facedetection")
                if facedetection_model in ComfyUI_Reactor.facedetection_weights:
                    weights_to_download.append(
                        ComfyUI_Reactor.facedetection_weights[facedetection_model]
                    )

    @staticmethod
    def prepare(**kwargs):
        print("[Reactor Helper] Running diagnostics...")
        try:
            import insightface
            print(f"[Reactor Helper] insightface version: {insightface.__version__}")
        except ImportError:
            print("[Reactor Helper] ❌ insightface NOT FOUND")
        
        try:
            import onnxruntime
            print(f"[Reactor Helper] onnxruntime version: {onnxruntime.__version__}")
            print(f"[Reactor Helper] onnxruntime providers: {onnxruntime.get_available_providers()}")
        except ImportError:
            print("[Reactor Helper] ❌ onnxruntime NOT FOUND")
        
        # Check if models directory exists
        models_path = "/opt/ComfyUI/models/insightface"
        if os.path.exists(models_path):
            print(f"[Reactor Helper] Models directory exists: {models_path}")
            print(f"[Reactor Helper] Contents: {os.listdir(models_path)}")
        else:
            print(f"[Reactor Helper] ❌ Models directory NOT FOUND: {models_path}")
