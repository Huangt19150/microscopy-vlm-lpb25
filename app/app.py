# import base64
import os
import sys
from PIL import Image
import gradio as gr
from openai import OpenAI

# NOTE: Deploy ONLY: Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from mwm_vlm.utils.common import encode_image, hash_image
from mwm_vlm.utils.gradio import load_cache, save_cache


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLIENT = OpenAI(api_key=OPENAI_API_KEY)
CACHE = load_cache()    


def clear_cache():
    """
    NOTE: Saved cache file `cache.json` will not be pushed to the File Repo when deployed on Hugging Face Spaces.
    Thus the global variable `CACHE` is the only control of cache when using the app on Hugging Face Spaces.
    This also means that the cache will be lost when the app is restarted (e.g. after `sleeping`).
    """
    global CACHE
    CACHE = {}
    save_cache(CACHE)
    print("Cache cleared.")
    return ""

# TODO: Refactor into src
def get_prpmpt(image_path):
    
    # Base64 encode each example image and your test image
    base64_example1 = encode_image("examples/prompt_images/6488.jpeg")
    base64_example2 = encode_image("examples/prompt_images/220.jpeg")
    base64_example3 = encode_image("examples/prompt_images/200.jpeg")
    base64_example4 = encode_image("examples/prompt_images/9123.jpeg")
    base64_test = encode_image(image_path)

    # Prepare the common part of the prompt
    instructions = (
        "You are helping with tagging Protein Crystallization images. "
        "You will be given an image and a list of labels with definitions. "
        "Choose the most appropriate single label from the list according to the definition. "
        "Add a short explanation of your choice at the end in free text.\n\n"
        "**List:**\n"
        "[clear, crystals, precipitate, other]\n\n"
        "**Definitions:**\n"
        "clear: Transparent solution without any particulate matter observed within a droplet or in the field of view.\n"
        "crystals: Crystals observed within a droplet or in the field of view.\n"
        "precipitate: Precipitate observed within a droplet or in the field of view.\n"
        "other: Unexpected observations which are not clear solution, crystals, or precipitate.\n\n"
        "Below are some examples:\n"
    )

    # Build the list of input items
    input_items = [
        {"type": "input_text", "text": instructions},

        # Example 1
        {"type": "input_text", "text": "**Example 1:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example1}"},
        {"type": "input_text", "text": "Label: crystals\nExplanation: Distinct crystal structures can be seen forming in the droplet."},

        # Example 2
        {"type": "input_text", "text": "**Example 2:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example2}"},
        {"type": "input_text", "text": "Label: clear\nExplanation: The droplet is transparent and shows no visible particles or structures."},

        # Example 3
        {"type": "input_text", "text": "**Example 3:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example3}"},
        {"type": "input_text", "text": "Label: precipitate\nExplanation: Granular particles are scattered across the droplet indicating precipitation."},

        # Example 4
        {"type": "input_text", "text": "**Example 4:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_example4}"},
        {"type": "input_text", "text": "Label: other\nExplanation: The image shows unexpected structures which are not clear solution, crystals, or precipitate."},

        # Now the actual test image
        {"type": "input_text", "text": "**Now classify the following image:**"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_test}"},
    ]

    return input_items

def classify(image: Image.Image) -> str:

    # Dump image
    image_savepath = "input_image.png"
    image.save(image_savepath)

    # Use cache if available
    hash_key = hash_image(image_savepath)
    if hash_key in CACHE:
        return CACHE[hash_key]

    # Prepare the prompt
    prompts = get_prpmpt(image_savepath)

    # Call the model
    response = CLIENT.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": prompts,
            }
        ],
        temperature=0.2,
    )

    # Save the result to cache
    CACHE[hash_key] = response.output_text
    save_cache(CACHE)

    # Clean up
    os.remove(image_savepath)

    return response.output_text

# Example images for Gradio
example_images = [
    ["examples/25.jpeg"],
    ["examples/215.jpeg"],
    ["examples/1401.jpeg"],
    ["examples/57465.jpeg"]
]

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Protein Crystallization Image Classifier
        Classify protein crystallization images using GPT4.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output_text = gr.Textbox(label="Classification Result")
            clear_btn = gr.Button("Clear Result")
            clear_cache_btn = gr.Button("🧹 Clear Cache")

    submit_btn.click(fn=classify, inputs=image_input, outputs=output_text)
    clear_btn.click(fn=lambda: "", inputs=None, outputs=output_text)
    clear_cache_btn.click(fn=clear_cache, outputs=output_text)

    gr.Examples(
        examples=example_images,
        inputs=image_input,
        outputs=output_text,
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
