import json
from typing import Optional, Literal
from pydantic import BaseModel, Field
from DPN.openai_helpers import call_llm

# --- Pydantic Schemas for Component Extraction ---

class ProductComponent(BaseModel):
    """
    Represents the extracted components for a product.
    """
    product: str = Field(description="The name/short description of the product.")
    material: Optional[str] = Field(None, description="The material of the product, if specified.")
    color: Optional[str] = Field(None, description="The color of the product, if specified.")
    added_details: Optional[str] = Field(None, description="Any additional details about the product, if specified.")

class PersonComponent(BaseModel):
    """
    Represents the extracted components for a person.
    """
    framing: Literal["face-only", "half-body", "full-body"] = Field("full-body", description="The framing of the person")
    gender: Optional[Literal["male", "female", "others"]] = Field(None, description="The gender of the person, if specified.")
    age: Optional[int] = Field(None, description="The age of the person, if specified.")
    appeareance: Optional[str] = Field(None, description="The appearance of the person, if specified.") 
    attire: Optional[str] = Field(None, description="The attire of the person, if specified.")
    expression: Optional[str] = Field(None, description="The expression of the person, if specified.")
    pose: Optional[str] = Field(None, description="The pose of the person, if specified.")
    mood: Optional[str] = Field(None, description="The mood of the person, if specified.")
    added_details: Optional[str] = Field(None, description="Any additional details about the person, if specified.")

class DetermineIMRBType(BaseModel):
    """
    Determine the type of the image (product or person)
    """
    imrb_type: Literal["product", "person"] = Field(None, description="The type of the image, either product or person")

class BackgroundViewComponent(BaseModel):
    """
    Represents the extracted components for a background view.
    """
    environment: str = Field(description="The environment of the background.")
    color: Optional[str] = Field(None, description="The color of the background, if specified.")
    atmosphere: Optional[str] = Field(None, description="The atmosphere of the background, if specified.")
    added_details: Optional[str] = Field(None, description="Any additional details about the background, if specified.")
    empty_image_location: Optional[str] = Field(
        None,
        description="The location within the image to be emptied, providing clear, unobstructed space for a product, if specified."
    )

class BackgroundPatternComponent(BaseModel):
    """
    Represents the extracted components for a background pattern.
    """
    pattern: str = Field(description="The description of the background pattern")
    color: Optional[str] = Field(None, description="The color of the background pattern, if specified.")
    added_details: Optional[str] = Field(None, description="Any additional details about the background pattern, if specified.")

class DetermineBackgroundType(BaseModel):
    """
    Represents the type of the background.
    """
    background_type: Literal["view", "pattern"] = Field(None, description="The type of the background, either view or pattern")

class GeneratedImagePrompt(BaseModel):
    """
    Represents the final, detailed English image prompt.
    """
    image_prompt_description: str = Field(
        description="The detailed English image prompt based on the extracted components."
    )

# --- LLM Prompt Templates ---
    
DETERMINE_PRODUCT_OR_PERSON_IMRB_PROMPT = """
You are an expert at classifying image types. Your task is to determine if a user's description
refers to a "product" image or a "person" image.

Based on the user's description, decide if the image type is "product" or "person".

---
User Description: {user_description}
---
"""

DETERMINE_VIEW_OR_PATTERN_BACKGROUND_PROMPT = """
You are an expert at classifying background types. Your task is to determine if a user's description
refers to a "view" background or a "pattern" background.

A **view** describes something that exists in the real world, like a mountain range, a city skyline,
or a room interior.
A **pattern** describes something abstract or designed, like a geometric line pattern, a floral print,
or a textured surface that isn't a real-world scene.

Based on the user's description, decide if the background type is "view" or "pattern".

---
User Description: {user_description}
---
"""

EXTRACT_COMPONENTS_LLM_TEMPLATE = """
You are an expert information extractor. Your task is to carefully parse a user's description
and extract specific components relevant for an image generation prompt. Translate everything to english. 

Return the extracted information as a **JSON object**. This JSON object must strictly
adhere to the Pydantic schema provided below.
If a field is not present or not applicable in the user's description, you must **omit that key** from the JSON object. Do not use 'None' or 'null' if the information isn't provided.

---
User Description: {user_description}
---

Pydantic Schema:
```json
{target_schema}
```
"""

CREATE_GOOD_PROMPT_LLM_TEMPLATE = """
You are an expert prompt engineer. Your task is to carefully create a good English prompt for
image generation based on the extracted components. Ensure the prompt is simple, compact, detailed and descriptive
to guide an image generation model effectively.

---
Extracted Components: {extracted_components}
---
"""

CREATE_GOOD_PROMPT_LLM_TEMPLATE_WITH_BACKGROUND = """
You are an expert prompt engineer. Your task is to create a concise, descriptive prompt that describe a product for social media design. 
Based on the background description prompt and extracted product components, create a good descriptive prompt in english. 
The prompt should **only** describe the product, mentioning and focusing on its **color and material** that seamlessly integrate with background theme, without mentioning any background elements.

---
**Extracted Product Components:** {extracted_components}
**Background Theme (for contextual understanding, not for prompt generation):** {background_prompt}
---
"""

# --- Main Orchestration Logic ---

def _process_image_prompt_generation(
    raw_user_message: str,
    target_extraction_model: BaseModel,
    field_type: str, 
    background_prompt : str = ""
) -> Optional[str]:
    """
    Helper function to orchestrate the extraction of components and generation of the final prompt.
    """
    try:
        # Step 1: Extract components using the LLM
        extracted_components = call_llm(
            model=target_extraction_model,
            prompt_template=EXTRACT_COMPONENTS_LLM_TEMPLATE,
            params={
                "user_description": raw_user_message,
                "target_schema": json.dumps(target_extraction_model.model_json_schema(), indent=2)
            }
        )
        print(f"\nExtracted Components: {extracted_components.model_dump_json(indent=2)}")

        if background_prompt:
            generated_prompt_obj = call_llm(
            model=GeneratedImagePrompt,
            prompt_template=CREATE_GOOD_PROMPT_LLM_TEMPLATE_WITH_BACKGROUND,
            params={
                "extracted_components": extracted_components.model_dump_json(indent=2),
                "background_prompt" : background_prompt
            }
        )
            
        else:
            generated_prompt_obj = call_llm(
                model=GeneratedImagePrompt,
                prompt_template=CREATE_GOOD_PROMPT_LLM_TEMPLATE,
                params={
                    "extracted_components": extracted_components.model_dump_json(indent=2)
                }
            )

        # Append common prompt suffixes
        final_prompt = generated_prompt_obj.image_prompt_description

        # if field_type == "imrb":
        #     final_prompt += ' Full-shot on a transparent background. No shadows or background elements. Even lighting. Entire object visible, no cropping. Three-quarter right-side view, slightly high angle. Centered with margin around edges.'
        # elif field_type == "bg":
        #     final_prompt += ' High Resolution 4K.'

        return final_prompt

    except Exception as e:
        print(f"An error occurred during prompt generation: {e}")
        return None

def generate_image_prompts(raw_user_message: str, field_type: str) -> Optional[str]:
    """
    Orchestrates the process of extracting components and generating an image prompt
    based on user input and a specified field type.

    Args:
        raw_user_message: The raw description provided by the user.
        field_type: The type of component to extract ('imrb' for product, 'bg' for background).

    Returns:
        The generated image prompt string, or None if an error occurred.
    """
    if field_type == 'imrb':
        try:
            imrb_type_obbj = call_llm(
                model=DetermineIMRBType,
                prompt_template=DETERMINE_PRODUCT_OR_PERSON_IMRB_PROMPT,
                params={"user_description": raw_user_message}
            )

            imrb_type = imrb_type_obbj.imrb_type

            if imrb_type == "person":
                target_extraction_model = PersonComponent
            elif imrb_type == "product":
                target_extraction_model = ProductComponent
            else:
                raise ValueError(f"Unexpected imrb type determined by LLM: {imrb_type}")
            
            final_prompt =  _process_image_prompt_generation(raw_user_message, target_extraction_model, field_type)

            if imrb_type == "product":
                final_prompt += ' Full-shot on a transparent background. No shadows or background elements. Even lighting. Entire object visible, no cropping. Three-quarter right-side view, slightly high angle. Centered with margin around edges.'
            elif imrb_type == "person":
                final_prompt += ' Zoomed out, from a distance. Transparent background with no shadow at all. Even Lighting. No cropping at all. Centered with margin around the edges.'

            return final_prompt
        
        except Exception as e:
            print(f"An error occurred during background prompt generation: {e}")
            return None

    elif field_type == 'bg':
        try:
            background_type_obj = call_llm(
                model=DetermineBackgroundType,
                prompt_template=DETERMINE_VIEW_OR_PATTERN_BACKGROUND_PROMPT,
                params={"user_description": raw_user_message}
            )
            background_type = background_type_obj.background_type

            if background_type == "view":
                target_extraction_model = BackgroundViewComponent
            elif background_type == "pattern":
                target_extraction_model = BackgroundPatternComponent
            else:
                raise ValueError(f"Unexpected background type determined by LLM: {background_type}")
        
            final_prompt =  _process_image_prompt_generation(raw_user_message, target_extraction_model, field_type)
            final_prompt += ' High Resolution 4K.'

            print(f"\nFinal Background Image Prompt: {final_prompt}\n")

            if_user_continue = input("Want to continue create the product?\n")

            if if_user_continue.lower() == "yes":
                new_product = input("Input your product!\n")
                final_product_prompt = _process_image_prompt_generation(new_product, ProductComponent, "imrb", final_prompt[:-20])

                final_product_prompt += ' Full-shot on a transparent background. No shadows or background elements. Even lighting. Entire object visible, no cropping. Three-quarter right-side view, slightly high angle. Centered with margin around edges.'

                return {"final_backround_prompt" : final_prompt, 
                        "final_product_prompt" : final_product_prompt}

            else:
                return final_prompt

        except Exception as e:
            print(f"An error occurred during background prompt generation: {e}")
            return None

    else:
        print(f"Error: Invalid field_type '{field_type}'. Must be 'imrb' or 'bg'.")
        return None
    
if __name__ == "__main__":
    print("\n" + "="*50 + "\n")
    print("Welcome to this service")

    while True:
        print("\n" + "="*50 + "\n")
        print("Choose")
        print("1. Product (imrb)")
        print("2. Background (bg)")
        print("3. Test Product based on defined background prompt")
        print("4. Exit")
        num_choice = input("Enter a number: ")
        print("\n" + "="*50 + "\n")

        if num_choice == "1":
            print("--- Testing Product (IMRB) ---")
            product_user_message = input("Input your message below: \n")
            final_product_prompt = generate_image_prompts(product_user_message, 'imrb')
            if final_product_prompt:
                print(f"\nFinal Product Image Prompt: '{final_product_prompt}'")
            print("\n" + "="*50 + "\n")

        elif num_choice == "2":
            print("--- Testing Background (BG) ---")
            background_user_message = input("Input your message below: \n")
            final_background_prompt = generate_image_prompts(background_user_message, 'bg')
            if final_background_prompt:
                print(f"\nFinal Background Image Prompt: {final_background_prompt}")
            print("\n" + "="*50 + "\n")

        elif num_choice == "3":
            background_prompt = "A serene and peaceful depiction of Mount Fuji surrounded by its natural environment, capturing the calm atmosphere of the landscape."
            new_product = input("Input your product!\n")
            final_product_prompt = _process_image_prompt_generation(new_product, ProductComponent, "imrb", background_prompt)
            print(final_product_prompt)

        elif num_choice == "4":
            print("Thanks for using this mini service!")
            print("\n" + "="*50 + "\n")
            break

        else:
            print("choose 1 or 2 or 3 or 4")