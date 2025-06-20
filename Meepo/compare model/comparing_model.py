import os
import replicate

def image_by_ideogramv3(prompt: str, set_up: str, aspect_ratio: str = "1:1", magic_prompt_option: str = "On"):
    input = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "magic_prompt_option": magic_prompt_option
    }

    print(f"  Calling Replicate for Ideogram v3 with prompt: '{prompt[:50]}...'")
    output = replicate.run(
        "ideogram-ai/ideogram-v3-quality",
        input=input
    )

    os.makedirs(set_up, exist_ok=True)
    with open(os.path.join(set_up, "image_ideogramv3.png"), "wb") as file:
        file.write(output.read())
    print(f"  Saved Ideogram v3 image to: {os.path.join(set_up, 'image_ideogramv3.png')}")


def image_by_openai_gpt1(prompt: str, set_up: str, image_index: int, input_images: list = []):
    input = {
        "prompt": prompt,
        "input_images": input_images,
        "openai_api_key": "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"
    }

    print(f"  Calling Replicate for OpenAI GPT-Image-1 with prompt: '{prompt[:50]}...'")
    output = replicate.run(
        "openai/gpt-image-1",
        input=input
    )

    os.makedirs(set_up, exist_ok=True)
    for index, item in enumerate(output):
        file_name = f"image_gptimage1_{image_index}_{index}.png"
        with open(os.path.join(set_up, file_name), "wb") as file:
            file.write(item.read())
        print(f"  Saved OpenAI GPT-Image-1 image to: {os.path.join(set_up, file_name)}")

prompts = [
    "a full-shot of a modern smartphone, sleek glass and aluminium design, dark blue matte color, banana logo at the back. Transparent background, no shadow at all. Lighting equal to all parts. Taken from right-side, three-quarter view, slightly high angle.",
    "a full-shot of a stylish sneakers, leather material with aggressive tread, vibrant blue and green color, featuring 'KOKA' logo at the side. Transparent background, no shadow at all. Lighting equal to all parts. Taken from right-side, three-quarter view, slightly high angle.",
    "a full-shot of a coffee mug, ceramic material, white color with a little brown stripe, filled with hot coffee and rising steam. Transparent background, no shadow at all. Lighting equal to all parts. Taken from right-side, three-quarter view, slightly high angle.",
    "a full-shot of a crispy pizza slice, consist of pineapple slice and pepperoni, italian pizza color. Transparent background, no shadow at all. Lighting equal to all parts. Taken from right-side, three-quarter view, slightly high angle.",
    "a full-shot of a mountain bicycle, steel and aluminium material, black matte color with a hint of red. Transparent background, no shadow at all. Lighting equal to all parts. Taken from right-side, three-quarter view, slightly high angle.",
]


print("--- Starting Image Generation ---")
for i in range(len(prompts)):
    folder_name = f"product{i+1}"
    print(f"\nProcessing prompt {i+1}/{len(prompts)} for folder: '{folder_name}'")

    os.makedirs(folder_name, exist_ok=True)

    print(f"  Generating picture for Ideogram v3...")
    image_by_ideogramv3(prompts[i], folder_name)
    print(f"  Done generating picture for Ideogram v3.")

    print(f"  Generating picture for OpenAI GPT-Image-1...")
    image_by_openai_gpt1(prompts[i], folder_name, i + 1)
    print(f"  Done generating picture for OpenAI GPT-Image-1.")

print("\n--- Image Generation Complete ---")