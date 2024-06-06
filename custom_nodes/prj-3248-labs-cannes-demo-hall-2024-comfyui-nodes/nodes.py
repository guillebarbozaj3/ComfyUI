import hashlib
import json
import os
import openai
import random
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence
import torch
import numpy as np
import pandas as pd
import folder_paths
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args


class ReportRow:
    def __init__(
        self,
        face_image,
        style_sample,
        control_net_sample,
        interest_prompt,
        vision_prompt,
        base_portrait,
        final_portrait,
    ):
        self.face_image = face_image
        self.style_sample = style_sample
        self.control_net_sample = control_net_sample
        self.interest_prompt = interest_prompt
        self.vision_prompt = vision_prompt
        self.base_portrait = base_portrait
        self.final_portrait = final_portrait

    def to_dict(self):
        """Convert the attributes to a dictionary."""
        return {
            "Face Image": f"<img src='../input/{self.face_image}' width='200'>",
            "Style Sample": f"<img src='../input/{self.style_sample}' width='200'>",
            "Control Net Sample": f"<img src='../input/{self.control_net_sample}' width='200'>",
            "Interest Prompt": self.interest_prompt,
            "Vision Prompt": self.vision_prompt,
            "Base Portrait": f"<img src='{self.base_portrait}' width='200'>",
            "Final Portrait": f"<img src='{self.final_portrait}' width='200'>",
        }


class HTMLReportGenerator:
    def __init__(self):
        self.data = []

    def initialize_report(self, file_path):
        """
        Initialize the report file. Create it if it doesn't exist.
        """
        if not os.path.exists(file_path):
            df = pd.DataFrame(self.data)
            df.to_html(file_path, index=False, escape=False)
            print(f"Report file '{file_path}' created successfully.")
        else:
            print(f"Report file '{file_path}' already exists.")

    def apply_styles(self, df):
        """
        Apply styles to the DataFrame.
        """
        styler = df.style

        # Apply custom styles here
        styler.set_table_styles(
            [
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#f2f2f2"),
                        ("color", "black"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ],
                },
                {"selector": "tbody td", "props": [("text-align", "center")]},
                {
                    "selector": "table",
                    "props": [("border-collapse", "collapse"), ("width", "100%")],
                },
                {
                    "selector": "th, td",
                    "props": [("border", "1px solid black"), ("padding", "8px")],
                },
            ]
        )

        # Other styling stuff
        styler.set_properties(**{"font-size": "12pt", "font-family": "Arial"})

        return styler

    def add_to_report(self, file_path, row):
        """
        Add a row to the report.

        Args:
        - file_path: Path to the report file.
        - row: ReportRow object containing the row data.
        """
        self.data.append(row.to_dict())
        df = pd.DataFrame(self.data)
        df.to_html(file_path, index=False, escape=False)
        print(f"Added row to the report.")


with open(
    "./custom_nodes/prj-3248-labs-cannes-demo-hall-2024-comfyui-nodes/credentials.json"
) as f:
    credentials = json.load(f)
    os.environ["OPENAI_API_KEY"] = credentials["openai_api_key"]

openai_client = openai.Client()


# Single text based inference using OpenAI API
class GPT4TextInference:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-4o"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "instructions": ("STRING", {"multiline": True}),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "inference"
    CATEGORY = "media.monks"

    def inference(self, model_name, temperature, instructions, prompt):
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        text = str(response.choices[0].message.content)

        return (text,)

    @classmethod
    def IS_CHANGED(s, model_name, temperature, instructions, prompt):
        return random.randint(1, 1024)


# Single image based inference using OpenAI API
class GPT4ImageInference:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_image": ("STRING", {"forceInput": True}),
                "model_name": ("STRING", {"default": "gpt-4o"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "media.monks"

    def inference(self, base64_image, model_name, temperature, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        text = str(response.choices[0].message.content)

        return (text,)

    @classmethod
    def IS_CHANGED(s, model_name, temperature, instructions, prompt):
        return random.randint(1, 1024)


# Same as default ComfyUI LoadImage, but with the option to return base64 string of the image
class LoadImageWithBase64:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "media.monks"
    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "STRING")
    RETURN_NAMES = ("image", "image_base64", "mask", "image_name")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        # get base64 string from img
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, img_str, output_mask, image_path.split("/")[-1])

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


# Same as default ComfyUI CLIP conditioning, but with the option to append text from another node
class CLIPCombineText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_to_append_1": ("STRING", {"forceInput": True}),
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP",),
            },
            "optional": {
                "text_to_append_2": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    FUNCTION = "encode"
    CATEGORY = "media.monks"

    def encode(self, clip, text, text_to_append_1, text_to_append_2=""):
        # add comma to the end of text if it doesn't end with one
        if text and text[-1] not in [",", " ,"]:
            text += ", "

        full_text = text + text_to_append_1

        if text_to_append_2 != "":
            full_text += ", " + text_to_append_2

        tokens = clip.tokenize(full_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], full_text)


# Used to display generated text in the UI
class PreviewText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    FUNCTION = "notify"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "media.monks"

    def notify(self, text):
        return {"ui": {"text": text}, "result": (text,)}


# HTML Report Generator Node
class HTMLReportGeneratorNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.report_dir = self.output_dir + "/report.html"
        self.report_generator = HTMLReportGenerator()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_image": ("STRING", {"forceInput": True}),
                "style_sample": ("STRING", {"forceInput": True}),
                "control_net_sample": ("STRING", {"forceInput": True}),
                "interest_prompt": ("STRING", {"forceInput": True}),
                "vision_prompt": ("STRING", {"forceInput": True}),
                "base_portrait": ("STRING", {"forceInput": True}),
                "final_portrait": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()

    FUNCTION = "generate_report"
    CATEGORY = "media.monks"
    OUTPUT_NODE = True

    def generate_report(
        self,
        face_image,
        style_sample,
        control_net_sample,
        interest_prompt,
        vision_prompt,
        base_portrait,
        final_portrait,
    ):

        print("Generating report...")

        # Initialize the report if needed
        self.report_generator.initialize_report(self.report_dir)
        # Create a new row with provided data
        new_row = ReportRow(
            face_image=face_image,
            style_sample=style_sample,
            control_net_sample=control_net_sample,
            interest_prompt=interest_prompt,
            vision_prompt=vision_prompt,
            base_portrait=base_portrait,
            final_portrait=final_portrait,
        )

        # Add the new row to the report
        self.report_generator.add_to_report(self.report_dir, new_row)

        return ()


class SaveImageAndReturnName:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(
        self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        final_file_name = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            final_file_name.append(file)
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1
        return {"ui": {"images": results}, "result": (final_file_name)}


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "GPT4TextInference": GPT4TextInference,
    "GPT4ImageInference": GPT4ImageInference,
    "LoadImageWithBase64": LoadImageWithBase64,
    "PreviewText": PreviewText,
    "CLIPCombineText": CLIPCombineText,
    "HTMLReportGeneratorNode": HTMLReportGeneratorNode,
    "SaveImageAndReturnName": SaveImageAndReturnName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT4TextInference": "GPT4 Text Inference",
    "GPT4ImageInference": "GPT4 Image Inference",
    "LoadImageWithBase64": "Load Image with Base64",
    "PreviewText": "Preview Text",
    "CLIPCombineText": "CLIP Combine Text",
    "HTMLReportGeneratorNode": "HTML Report Generator",
    "SaveImageAndReturnName": "Save Image and Return Name",
}
