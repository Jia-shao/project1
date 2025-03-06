from openai import OpenAI
import re
import json
dt_descrip = f"""In details, the image is segmented into several regions and numbered. \
And for each region, it is represented in the following form with four attributes. \n
"Object <ID>: \"semantic label\": <semantic label>, \"path of mask\": <path of mask>, \"depth\": <depth>" \n
The attribute id is just the number of each region. \
The attribute "semantic label" tells you what the object of the corresponding mask area is. \
The attribute "path of mask" tells you the save path for the corresponding mask. It is saved as an ".png" file. \
The attribute "depth" tells you the relative depth of the corresponding mask area in the image. \
"""

def extract_bracket_content(text):
    pattern = r'\[(.*?)\]'
    match = re.search(pattern, text)
    
    if match:
        content = match.group(1)
        print(content)
        items = [item.strip() for item in content.split(",")]
        return items
    else:
        return []

def check_first_word_is_no(sentence):
    sentence = sentence.strip()
    
    match = re.match(r"[a-zA-Z]+", sentence)
    
    if match:
        first_word = match.group().lower()  
        return first_word == "no"
    else:
        return False

def check_first_word_is_yes(sentence):
    sentence = sentence.strip()
    
    match = re.match(r"[a-zA-Z]+", sentence)
    
    if match:
        first_word = match.group().lower()
        return first_word == "yes"
    else:
        return False

def call_llm(prompt, api_key, model_name="qwen-max"):
    api_key=""
    model_name=""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # {
                #     'role': 'system',
                #     'content': 'You are a helpful assistant.'
                # },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        
                    ]
                }
            ],
            stream=False,
            max_tokens=8191
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"{e}")
        print("https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None
    
#########################################################
def judge_instruction_one_step(instruction, api_key, model_name):
    # prompt for whether or not the instruction is a one-step editing instrution
    prompt = f"""Please determine whether the following image editing task instruction is a one-step editing instruction. \
A one-step editing instruction refers to an image editing task that can be completed in a single operation, \
such as removing an object, adding an object, or replacing one object with another.\n
Instruction: {instruction}\n
Please give you answer as a single word, either \"yes\" or \"no\".\n
Please note that if the instruction can be described using 'replace', it should not be described using 'remove' and 'add'.
"""

    # answer for whether or not the instruction is a one-step editing instruction
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    if check_first_word_is_yes(answer):
        return True
    elif check_first_word_is_no(answer):
        return False
    else:
        print(f"\nError: the answer for judge whether or not the instruction is a one-step instruction is not yes or no\n")
        return False

#########################################################
def split_instruction(instruction, api_key, model_name):
    prompt = f"""A one-step editing instruction refers to an image editing task that can be completed in a single operation, \
such as removing an object, adding an object, or replacing one object with another.\n
However, the following image editing task instruction is not a one-step instruction. \n
Instruction: {instruction}\n
Please break down the above complex image editing task instruction into multiple one-step editing instructions. \
Each one-step instruction should be a single operation that can be completed independently, \
such as removing an object, adding an object, or replacing an object.\
Your answer should in the form [\'xxxx\', \'xxxx\', ...].
for example:
[
    "Step 1: Remove the billboard from the background",
    "Step 2: Replace the sky with blue",
    "Step 3: Add a flower in the bottom right corner of the image"
]
Please note that your answer should only include one list, as in the example.    
"""
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    return answer
    
########################################
def classify_instruction(instr, api_key, model_name):
    prompt = f"Instruction: {instr}\n"
    prompt += "This is an instruction about image editing. Please determine which type of editing it belongs to:\n"
    prompt += " If the instruction is about remove something, please response the number \'1\';\n"
    prompt += " If the instruction is about change or replace something, please response the number \'2\';\n"
    prompt += " If the instruction is about add something, please response the number \'3\'.\n"
    prompt += "Please note that you only need to response a number from 1 to 3."
    answer = call_llm(prompt, api_key=api_key, model_name=model_name)
    return answer

# #######################################
def judge_dt_answer(instruction, digital_twin, api_key, model_name):
    # get the answer for whether the digital twin information is enough to complete the instruction
    prompt = f"""Your task is to determine whether you can complete the instruction only based on the digital twin information and your common sense. \
The instruction is about editing an image. \
And the digital twin information tells you the information about the image that needs to be edited. \
"""
    prompt += dt_descrip
    prompt += f"The instruction is: {instruction}\n"
    prompt += f"The digital twin information is:\n"
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"
    prompt += f"""There are some guidelines for your answer:\n
1. You should use the digital twin information provided to determine if you can reasonably complete the instruction.
2. If the digital twin information does not provide all the specific details required by the instruction, \
you should use your common sense to make a reasonable assumption based on the given information.
3. You should answer \"yes\" whenever possible, as long as you can make a reasonable inference \
and reach a conclusion using the digital twin information and common sense.
4. You may answer \"no\" only if you are certain that there is missing semantic or spatial information.\n"""
    prompt += f"Please give you answer as a single word, either \"yes\" or \"no\"."
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    if check_first_word_is_yes(answer):
        return True
    elif check_first_word_is_no(answer):
        return False
    else:
        print(f"\nError: the answer for judge whether the digital twin information is enough or not is not yes or no\n")
        return False

#######################################
def mask_list(instrution, digital_twin, api_key, model_name):
    prompt = f"""Your task is to return the ID of the region that needs to be edited \
based on the given image editing instruction and digital twin information. \n
The digital twin information tells you the information about the image that needs to be edited.
"""
    prompt += dt_descrip
    prompt += f"""\nThe instruction is: {instrution}\n"""
    prompt += f"""The digital twin information is:\n"""
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"
    prompt += f"""Your answer should be in the form [number1, number2, ...]. \
If you think there is no correct answer, you can choose the closest option randomly. \
Please note that you only need to return a list of number, without any explanation.
"""
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    merge_list=[]
    for item in extract_bracket_content(answer):
        merge_list.append(int(item))
    return merge_list

#######################################
def simplified_editing_prompt(edit_type, instruction, api_key, model_name):
    prompt = f"""Your task is to convert the following complex and implicit instruction about image editing below \
into a simple and direct instruction based on the digital twin information of the image"""
    prompt += f"The original instruction is: {instruction}.\n"
    prompt += f"""The digital twin information tells you what is in an image by segmenting it into different regions"""
    prompt += dt_descrip
    prompt += f"""\nPlease give your answer in the following form:\n"""
    if edit_type == '1':
        prompt += f"\"remove the <objects that are required to remove>.\""
    elif edit_type == '2':
        prompt += f"\"draw the <objects that replace the original objects>.\""
    else:
        prompt += f"\"draw the <objects that are required to add>.\""
    prompt += f"""\n Please note that your answer needs to be consistent with the information in the original instruction, \
and also use the explicit, easy and direct description. \
You just need to answer in the format I requested, without any explanation"""
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    return answer

##############################################
def need_which_infor(instruction, digital_twin, api_key, model_name):
    # This function is to judge which information is needed to complete the instruction
    prompt = f"""I will give you an instruction about image editing, and the digital twin information of the image. \
The digital twin information tells you the information about the image that needs to be edited. \
"""
    prompt += dt_descrip
    prompt += f"""The instruction is: {instruction}\n"""
    prompt += f"""The digital twin information is:\n"""
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"
    prompt += f"""The digital twin information above is not enough to complete the instruction. \
To complete the instruction, please determine what type of information is needed. \n
1. If more semantic information is needed, that is you need to know more objects in the image, \
please answer a number "1" only.\n
2. If more spatial information or depth information is needed, that is you need to know more information about positioin, \
please answer a number "2" only.\n
3. If other kinds of information is needed, \
please answer a number "3" only.\n
4. If you find that you need semantic information and spatial information at the same time, \
please answer a number "1" only.
"""
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    return answer

##########################################
def find_semantic_label(instruction, digital_twin, api_key, model_name):
    prompt = f"""I will give you an instruction about image editing, and the digital twin information of the image. \
The digital twin information tells you the information about the image that needs to be edited. \
"""
    prompt += dt_descrip
    prompt += f"""The instruction is: {instruction}\n"""
    prompt += f"""The digital twin information is:\n"""
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"
    prompt += f"""The digital twin information above is not enough to complete the instruction. \
To complete the instruction, more semantic information about the object is needed \
to determine which object needs to be edited. \
Your task is to find what other objects (\"semantic label\")you need to obtain from the image \
besides those that already provided in the digital twin information, \
so that you can complete the instruction. \
"""
    prompt += f'''Please note that you should only answer the \"semantic label\" in the form [xxxx, xxxx, ...], \
where the \"semantic label\" is the name of the object you need to obtain from the image.'''
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    return answer

##########################################
def find_position(instruction, digital_twin, api_key, model_name):
    prompt = f"""I will give you an instruction about image editing, and the digital twin information of the image. \
The digital twin information tells you the information about the image that needs to be edited. \
"""
    prompt += dt_descrip
    prompt += f"""The instruction is: {instruction}\n"""
    prompt += f"""The digital twin information is:\n"""
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"
    prompt += f"""The digital twin information above is not enough to complete the instruction. \
To complete the instruction, more spatial information about position is needed to determine which area needs to be edited. \
Your task is to generate the python code \
to output the save path of the mask corresponding to the requested editing area of the instruction, \
by inputting the digital twin information.\n
You can follow the steps below to generate the python code.\n
"""
    prompt += f"""Firstly, You need to determine which objects' position information you need based on the instruction.
Then, you need to load binary masks you need from paths with their semantic label and depth information.
After that, you need to calculate the coordinate of geometric center of each mask region.
You can use the position coordinates of the geometric center as the position coordinates of the entire mask area.
Finally, you need to use the position coordinates gotten \
above to obtain the path corresponding to the mask of the area that needs to be edited as required by the instruction. \
If there are any existing masks that meet the requirements of the instruction for the edited area of the instruction, \
you can directly return the path of the existing mask; If not, you need to create a mask that meets the requirements and save it. \n
Please pay attention to the relationship between the world's top, bottom, left, right, and index in the array. \
Please note that you should only return the python code, without any explanation."""
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    return answer

##########################################
def find_question(instruction, digital_twin, api_key, model_name):
    prompt = f"""I will give you an instruction about image editing, and the digital twin information of the image. \
The digital twin information tells you the information about the image that needs to be edited. \
"""
    prompt += dt_descrip
    prompt += f"""The instruction is: {instruction}\n"""
    prompt += f"""The digital twin information is:\n"""
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"
    prompt += f"""The digital twin information above is not enough to complete the instruction. \
To complete the instruction, more information is needed to determine which area needs to be edited. \
Your task is to ask a question to get what other information you need so that you can complete the instruction.\n
""" 
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    return answer
