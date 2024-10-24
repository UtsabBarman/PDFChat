import os
import json
from pdf2image import convert_from_path
import google.generativeai as genai

with open("./config.json", "r") as f:
    config = json.load(f)

GOOGLE_API_KEY = config.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


def convert_pdf_to_jpg(pdf_file_path: str, pdf_image_output_dir: str):
    converted_image_paths = []
    try:
        # pdf_file_path = os.path.abspath(pdf_file_path)
        images = convert_from_path(pdf_file_path)
        for i, img in enumerate(images):
            img_path = os.path.join(
                pdf_image_output_dir,
                f"{os.path.basename(pdf_file_path)}-page-{(i+1)}.jpeg",
            )
            img.save(img_path, "JPEG")
            converted_image_paths.append(img_path)
    except Exception as e:
        print(f"Error converting {pdf_file_path}: {e}")
    else:
        print(f"Successfully converted {pdf_file_path} to {pdf_image_output_dir}")
    return converted_image_paths


def get_image_description(image_file_path: str, prompt: str):
    img_file = genai.upload_file(image_file_path)
    return model.generate_content([img_file, "\n\n", prompt]).text


def convert_pdf_to_text_description(pdf_file_path):
    text_extraction_prompt = (
        "extract all the text from the image in nice markdown format."
        "if the page contain an technical architecture, generate a vivid description of the architecture in detail. "
        "Each component should be described in detail. "
        "Describe how the solution is working. "
        "Describe the data flow. "
        "Also mention the robustness and advantages of this architecture."
        "always present in a nice markdown format."
    )
    doc = []
    if os.path.exists(pdf_file_path):
        base_dir = os.path.dirname(pdf_file_path)
        out_pdf_dir = os.path.join(base_dir, os.path.basename(pdf_file_path))
        out_pdf_dir = out_pdf_dir.replace(".pdf", "")
        if not os.path.exists(out_pdf_dir):
            os.mkdir(out_pdf_dir)
        page_img_paths = convert_pdf_to_jpg(
            pdf_file_path=pdf_file_path, pdf_image_output_dir=out_pdf_dir
        )
        for pi, img_file_path in enumerate(page_img_paths):
            text = get_image_description(
                image_file_path=img_file_path, prompt=text_extraction_prompt
            )
            fid = f"{os.path.basename(pdf_file_path)}-page-{(pi+1)}"
            doc.append(((fid), text))
    return doc
