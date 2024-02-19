from dotenv import load_dotenv

import requests 
from datetime import datetime 
from os import environ
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import autogen

import replicate

load_dotenv()

OUTPUT_FOLDER = environ["OUTPUT_FOLDER"]
REPLICATE_API_TOKEN = environ["REPLICATE_API_TOKEN"]

def format_filename_or_dir(s):
    """Take a string and return a valid filename constructed from the string.
Uses a whitelist approach: any characters not present in valid_chars are
removed. Also spaces are replaced with underscores.
 
Note: this method may produce invalid filenames such as ``, `.` or `..`
When I use this method I prepend a date string like '2009_01_15_19_46_32_'
and append a file extension like '.txt', so I avoid the potential of using
an invalid filename.
 
"""
    import string 
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename


autogen_config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    # filter_dict={
    #     # Function calling with GPT 3.5
    #     "model": ["gpt-3.5-turbo"],
    # }
)

# Create llm config for group chat manager 
# - GroupChatManager is not allowed to make function/tool calls.
autogen_llm_config = {
    "config_list": autogen_config_list
}

# Create llm config for assistants
autogen_llm_config_assistants = {
    "functions": [
        {
            "name": "create_image",
            "description": "use latest AI model to create an image based on a prompt, return the file path of image generated",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "a great text to image prompt that describe the image",
                        }
                    },
                "required": ["prompt"],
            },
        },
        {
            "name": "review_image",
            "description": "review & critique the AI generated image based on original prompt, decide how the image & prompt can be improved",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "the original prompt used to generate the image",
                        },
                        "image_file_path": {
                            "type": "string",
                            "description": "the image file path, make sure including the full file path & file extension",
                        }
                    },
                "required": ["prompt", "image_file_path"],
            },
        },
    ],
    "config_list": autogen_config_list,
}

# function to use stability-ai model to generate image
def create_image(prompt: str) -> str:
    output = replicate.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={
            "prompt": prompt
        }
    )

    if output and len(output) > 0:
        # Get the image URL from the output
        image_url = output[0]
        print(f"generated image for {prompt}: {image_url}")

        # Download the image and save it with a filename based on the prompt and current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        shortened_prompt = prompt[:50]        
        image_file_path = f"{OUTPUT_FOLDER}/{shortened_prompt}_{current_time}.png"

        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_file_path, "wb") as file:
                file.write(response.content)
            print(f"Image saved as '{image_file_path}'")
            return image_file_path
        else:
            raise Exception("Failed to download and save the image.")
    else:
        raise Exception("Failed to generate the image.")


def review_image(image_file_path: str, prompt: str):
    output = replicate.run(
        "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
        input={
            "image": open(image_file_path, "rb"),
            "prompt": f"What is happening in the image? From scale 1 to 10, decide how similar the image is to the text prompt {prompt}?",
        }
    )

    result = ""
    for item in output:
        result += item

    print("CRITIC : ", result)

    return result


# Create assistant agent
graphic_designer = AssistantAgent(
    name="graphic_designer",
    # system_message="You are a text to image AI model expert, you will use text_to_image function to generate image with prompt provided, and also improve prompt based on feedback provided until it is 10/10.",
    system_message="""Graphic Designer. You are a helpful assistant highly skilled in creating weather forecast images for weather app on android phone to reflect a text description. 
You MUST try every way to create an image to visualize the description. 
If you are not familiar with the description, you use the stocked images and modify them to represent the description. 
You should continue creating images until the graphic_critic rates the image with an average score higher than 9. 
""", 
    llm_config=autogen_llm_config_assistants,
    function_map={
        "create_image": create_image
    }
)

graphic_critic = AssistantAgent(
    name="graphic_critic",
    # system_message="You are an AI image critique, you will use img_review function to review the image generated by the text_to_img_prompt_expert against the original prompt, and provide feedback on how to improve the prompt.",
     system_message="""Critic. You are a helpful assistant highly skilled in evaluating the quality of a given image code by providing a score from 1 (bad) - 10 (good). Specifically, you can carefully evaluate the image across the following dimensions
- Content similarity (content): Does the image present the content in the prompt? If any keyword is missing, the score must less than 5. 
- Clarity (clarity): Does the image clearly communicate the prompt? Is it easy to understand for the human viewers?
- Balance and proportion (proportion): Are the elements arranged in a visually pleasing and harmonious way?
- Color and typography (color): Are the colors and fonts used effectively to enhance the message and brand identity?

YOU MUST PROVIDE A SCORE for each of the above dimensions.
{content: 0, clarity: 0, proportion: 0, color: 0}
Finally, based on the critique above, suggest a list of concreteactions that the graphic_designer should take to improve the image.
""",
    llm_config=autogen_llm_config_assistants,
    function_map={
        "review_image": review_image
    }
)

# Create user proxy agent
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=autogen_llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

# Create groupchat
groupchat = autogen.GroupChat(
    agents=[user_proxy, graphic_designer, graphic_critic], messages=[], max_round=50,)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=autogen_llm_config)


if __name__ == "__main__":
    # message = "A realistic image of a cute rabbit wearing sunglasses confidently driving a shiny red sports car on a sunny day with a scenic road in the background."
    # message = "A realistic image of a sexy girl wearning bikini lying on a beach with a hawaii mountain in the background."
    # message = "In Houston at 2pm, Sunny sky with few clouds. Current Temperature at 37 degree"
    # message = "sunny sky with few clouds, Temperature low at 37 degree and high up to 72 degree"
    # message = "overcast sky with dense clouds, some foggy, Temperaturelow at 37 degree and high up to 72 degree"
    message = "In Houston at 8pm, Heavy rain with very low visibility. Temperature in the 40s"

    # text_to_image_generation(message)

    # img_review('./output/A realistic image of a cute rabbit wearing sunglas_20240218135543.png', message)

    OUTPUT_FOLDER = f"{OUTPUT_FOLDER}/{format_filename_or_dir(message[:50])}"
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    # # Start the conversation
    user_proxy.initiate_chat(
        manager, message=message)
