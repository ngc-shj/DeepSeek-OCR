import os
import re
from tqdm import tqdm
import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, MAX_CONCURRENCY, CROP_MODE, NUM_WORKERS
from concurrent.futures import ThreadPoolExecutor
import glob
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor


llm = LLM(
    model=MODEL_PATH,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True, 
    max_model_len=8192,
    max_num_seqs = MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    logits_processors=[NGramPerReqLogitsProcessor]
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=40,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def clean_formula(text):

    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)

        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        
        formula = formula.strip()
        
        return r'\[' + formula + r'\]'

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    # mathes_image = []
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def process_single_image(image):
    """single image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": image},
    }
    return cache_item


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Batch Evaluation')
    parser.add_argument('--input', '-i', type=str, default=INPUT_PATH, help='Input directory path (images)')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_PATH, help='Output directory path')
    parser.add_argument('--prompt', '-p', type=str, default=PROMPT, help='OCR prompt')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    input_path = args.input if args.input else INPUT_PATH
    output_path = args.output if args.output else OUTPUT_PATH
    prompt = args.prompt if args.prompt else PROMPT
    
    if not input_path:
        print(f'{Colors.RED}Error: INPUT_PATH is not specified. Use --input or set INPUT_PATH in config.py{Colors.RESET}')
        exit(1)
    if not output_path:
        print(f'{Colors.RED}Error: OUTPUT_PATH is not specified. Use --output or set OUTPUT_PATH in config.py{Colors.RESET}')
        exit(1)

    # INPUT_PATH = OmniDocBench images path

    os.makedirs(output_path, exist_ok=True)

    # print('image processing until processing prompts.....')

    print(f'{Colors.RED}glob images.....{Colors.RESET}')

    images_path = glob.glob(f'{input_path}/*')

    images = []

    for image_path in images_path:
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    # batch_inputs = []


    # for image in tqdm(images):

    #     prompt_in = prompt
    #     cache_list = [
    #         {
    #             "prompt": prompt_in,
    #             "multi_modal_data": {"image": Image.open(image).convert('RGB')},
    #         }
    #     ]
    #     batch_inputs.extend(cache_list)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))


    

    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )


    os.makedirs(output_path, exist_ok=True)

    print(f'{Colors.GREEN}Processing results...{Colors.RESET}')
    processed_count = 0
    error_count = 0

    for output, image in zip(outputs_list, images_path):
        try:
            content = output.outputs[0].text
            base_name = os.path.basename(image)
            name_without_ext = os.path.splitext(base_name)[0]

            mmd_det_path = output_path + '/' + name_without_ext + '_det.md'

            with open(mmd_det_path, 'w', encoding='utf-8') as afile:
                afile.write(content)

            content = clean_formula(content)
            matches_ref, mathes_other = re_match(content)
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other", leave=False)):
                content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
            
            mmd_path = output_path + '/' + name_without_ext + '.md'

            with open(mmd_path, 'w', encoding='utf-8') as afile:
                afile.write(content)
            
            processed_count += 1
        except Exception as e:
            print(f'{Colors.RED}Error processing {image}: {e}{Colors.RESET}')
            error_count += 1
            continue
    
    print(f'{Colors.GREEN}Done! Processed {processed_count} images, {error_count} errors{Colors.RESET}')
    
    # Clean up LLM engine to avoid "died unexpectedly" error
    try:
        del llm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
