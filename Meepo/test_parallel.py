import time
import replicate

def image_by_openai_gpt1(prompt: str, set_up:str, input_images: list = []):
    input = {
        "prompt": prompt,
        "input_images": input_images,
        "openai_api_key": "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"
    }

    output = replicate.run(
        "openai/gpt-image-1",
        input=input
    )
    for index, item in enumerate(output):
        with open(f"personal_image_research/{set_up}/image_gptimage1.png", "wb") as file:
            file.write(item.read())

print("par 1 start")

start_time = time.time()
prompt = "A Naruto keychain featuring the full team from Team 7 as adults, with Naruto as Hokage. The keychain should capture the essence of each character in their grown-up form, highlighting their unique features and attire. Full product shot on a transparent background. No shadows or background elements. Even lighting. Entire product visible, no cropping. Three-quarter right-side view, slightly high angle. Centered with margin around edges."

image_by_openai_gpt1(prompt=prompt, set_up="test")

mid_time = time.time()
first_image_time = mid_time - start_time
print("par 1: first image done!, time needed :"+ str(first_image_time))

image_by_openai_gpt1(prompt=prompt, set_up="test")

end_time = time.time()
second_image_time = end_time - mid_time
print("par 1: second image done!, time needed :"+ str(second_image_time))