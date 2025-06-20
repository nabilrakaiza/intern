import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import replicate
from pydantic import BaseModel, Field

class ProductPromptStructure(BaseModel):
    product : str = Field(description="A specific and detailed description of the setting/background of the image.")
    colors : str = Field(description="A detailed description of the color of the product.")
    condition : str = Field(description="A detailed condition of the product.")
    presentation : str = Field(description="lighting and angle of the product.")
    additional_details : str = Field(description="Additional details of the product.")

class BackgroundPromptStructure(BaseModel):
    environment : str = Field(description="A specific and detailed description of the setting/background of the image.")
    mood : str = Field(description="The emotional tone/atmosphere of the image.")
    colors : str = Field(description="A specific color palette used in the image.")
    visual_qualities : str = Field(description="Quality, direction, color of light (lighting), and the perspective of the background.")
    additional_details : str = Field(description="Additional details of the background.")

def get_model_fields_and_descriptions(model: BaseModel) -> list[str]:
    """
    Input:
        - Pydantic model (background/product)
    Output:
        - list of strings about field name and their detailed description
    Purpose:
        - Extracts field names and their descriptions from a Pydantic model and formats them as a list of strings.
    """
    formatted_list = []
    for field_name, field_info in model.model_fields.items():
        description = field_info.description if field_info.description else "No description provided."
        formatted_list.append(f"- **{field_name}**: {description}")
    return formatted_list

def enhance_user_prompt(prompt: str, image_type: str, conversation_history: str = None) -> str:
    """
    Input:
        - user prompt 
        - image type (background/product)
        - conversation history (if exist)

    Output:
        - enhanced and structured user prompt

    Purpose:
        - enhanced and structured user prompt for image generation model input to hopefully create better end results
    """

    user_prompt = """
    Give an enhanced user prompt for image generation given by the user. 

    User prompt : {prompt}

    Conversation History : {conversation_history}

    Prompt Structure : {prompt_structure}

    Your task is to improve user's prompt for image generation task. You must return the enhanced prompt strictly adhering to the specified structure provided to you.
    """

    system_prompt = """
    You are a professional prompt engineer specializing in image generation. Your primary role is to enhance user messages into highly detailed, specific, and vivid prompts suitable for creating compelling images.

    If the user provides limited detail, actively enrich the prompt by adding relevant descriptive elements, always aiming for a richer visual outcome based on their initial intent.

    **Crucially, you must return the enhanced prompt strictly adhering to the specified structure provided to you.**
    """
    
    llm = ChatOpenAI(model_name = "gpt-4o", temperature="0.3", api_key=os.environ("OPEN_API_KEY"))

    if image_type == "background":
        prompt_structure = get_model_fields_and_descriptions(BackgroundPromptStructure)
    elif image_type == "product":
        prompt_structure = get_model_fields_and_descriptions(ProductPromptStructure)
    else:
        raise ValueError(f"Unknown image_type: '{image_type}'. Expected 'background' or 'product'.")

    user_prompt = user_prompt.format(prompt=prompt, conversation_history=conversation_history, prompt_structure=prompt_structure)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    result = llm.invoke(messages)
    enhanced_user_prompt = result.content

    return enhanced_user_prompt

def generate_image(prompt: str, model: str, set_up:str) -> None:
    """
    Input: 
        - enhanced and structured prompt
        - image generation model
        - set_up (where to save file)

    Output:
        - None (directly saves image locally)
    
    Purpose:
        - uses several image generation model to generate image(s) from enhanced and structured user prompt
    """

    if model == "ideogramv3":
        ImageGenerationModels.image_by_ideogramv3(prompt, set_up)
    elif model == "fluxpro11":
        ImageGenerationModels.image_by_fluxpro11(prompt, set_up)
    elif model == "google_image_gen4":
        ImageGenerationModels.image_by_google_image_gen4(prompt, set_up)
    elif model == "openai_gpt1":
        ImageGenerationModels.image_by_openai_gpt1(prompt, set_up)
    else:
        raise ValueError(f"Unknown image generation model: '{model}'.")


class ImageGenerationModels:
    @staticmethod
    def image_by_ideogramv3(prompt: str, set_up:str ,aspect_ratio: str="1:1", magic_prompt_option: str="On"):
        input = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "magic_prompt_option": magic_prompt_option
        }

        output = replicate.run(
            "ideogram-ai/ideogram-v3-quality",
            input=input
        )
        with open(f"personal_image_research/{set_up}/image_ideogramv3.png", "wb") as file:
            file.write(output.read())

    @staticmethod
    def image_by_fluxpro11(prompt: str,set_up:str, width: int=1024, height: int=1024, prompt_upsampling: bool=True):
        input = {
            "prompt": prompt,
            "prompt_upsampling": prompt_upsampling,
            "width": width,
            "height": height,
            "safety_tolerance": 2
        }

        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input
        )
        with open(f"personal_image_research/{set_up}/image_fluxpro11.png", "wb") as file:
            file.write(output.read())

    @staticmethod
    def image_by_google_image_gen4(prompt: str, set_up:str, aspect_ratio: str="1:1"):
        input = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "safety_filter_level": "block_medium_and_above"
        }

        output = replicate.run(
            "google/imagen-4",
            input=input
        )
        with open(f"personal_image_research/{set_up}/image_googleimagegen4.png", "wb") as file:
            file.write(output.read())

    @staticmethod
    def image_by_openai_gpt1(prompt: str, set_up:str, input_images: list = []):
        input = {
            "prompt": prompt,
            "input_images": input_images,
            "openai_api_key": os.environ("OPEN_API_KEY")
        }

        output = replicate.run(
            "openai/gpt-image-1",
            input=input
        )
        for index, item in enumerate(output):
            with open(f"personal_image_research/{set_up}/image_gptimage1.png", "wb") as file:
                file.write(item.read())

# ini buat orang yang gak ngerti design
# test = enhance_user_prompt("halo, saya mau design product. productnya tuh tentang bola basket dan kita emang jualan itu. buat\
#                            designnya, sebenernya saya gak terlalu masalah, as long as it visually appleaing, sama terliat gak terlalu\
#                            aneh. mungkin bisa disesuaikan sama yang ada di market", "product")

# print(test) 

# ### hasil test enhance user prompt
# ['- **product**: A modern and sleek basketball design set against a dynamic urban basketball court background, with graffiti art on the walls and a vibrant city skyline in the distance.', 
#  '- **colors**: The basketball features a bold combination of deep orange and black, with subtle metallic silver accents for a contemporary look.', 
#  '- **condition**: The basketball is brand new, with a smooth, polished surface and clearly defined seams, showcasing its premium quality.', 
#  '- **presentation**: The product is illuminated by bright, natural sunlight, casting soft shadows, and is captured from a slightly elevated angle to highlight its texture and design details.', 
#  '- **additional_details**: The basketball includes a subtle, embossed logo that blends seamlessly with the design, and the texture of the ball is visible, emphasizing its grip and durability.']


# ini buat orang yang ngerti design (anggep aja gitu soalnya saya sebenernya tidak terlalu paham)

# "A {product} with {product_description}. Emphasize {key_visual_elements}. {lighting_and_angle}. Written text (if any) {written_text}. Overall quality: high-quality, realistic, eye-catching, natural, full-face."

# test_1 = enhance_user_prompt("Halo, saya mau design product lagi nih bang. product saya tuh tentang air mineral AKUA. Jadi, karena \
#                              ini tentang air minum, saya mau latarnya tuh fresh dan bertema pengunungan dengan air sungai yang jernih dan bersih. \
#                              selain itu, tampilkan pula botol kemasan saya dengan logo AKUA di bagian tengah atas gambar. Pastikan kalau")



# start_time = time.time()
# prompt = "A Naruto keychain featuring the full team from Team 7 as adults, with Naruto as Hokage. The keychain should capture the essence of each character in their grown-up form, highlighting their unique features and attire. Full product shot on a transparent background. No shadows or background elements. Even lighting. Entire product visible, no cropping. Three-quarter right-side view, slightly high angle. Centered with margin around edges."

# generate_image(prompt=prompt, model="openai_gpt1", set_up="test")

# mid_time = time.time()
# first_image_time = mid_time - start_time
# print("first image done!, time needed :"+ first_image_time)

# generate_image(prompt=prompt, model="openai_gpt1", set_up="test")

# end_time = time.time()
# second_image_time = end_time - mid_time
# print("second image done!, time needed :"+ second_image_time)