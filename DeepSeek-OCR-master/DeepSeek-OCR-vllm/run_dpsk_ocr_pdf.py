import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
 

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"


from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor


llm = LLM(
    model=MODEL_PATH,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True, 
    max_model_len=8192,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    logits_processors=[NGramPerReqLogitsProcessor]
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
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

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images

def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return
    
    image_bytes_list = []
    
    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")



def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):


    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_path):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_path):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, output_path)
    return result_image


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
    
    parser = argparse.ArgumentParser(description='DeepSeek-OCR PDF Processing')
    parser.add_argument('--input', '-i', type=str, default=INPUT_PATH, help='Input PDF path')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_PATH, help='Output directory path')
    parser.add_argument('--prompt', '-p', type=str, default=PROMPT, help='OCR prompt')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode (show first 200 chars of each output)')
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

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)
    
    print(f'{Colors.RED}PDF loading .....{Colors.RESET}')


    images = pdf_to_images_high_quality(input_path)


    prompt = args.prompt if args.prompt else PROMPT

    # batch_inputs = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(process_single_image, images),
            total=len(images),
            desc="Pre-processed images"
        ))


    # for image in tqdm(images):

    #     prompt_in = prompt
    #     cache_list = [
    #         {
    #             "prompt": prompt_in,
    #             "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    #         }
    #     ]
    #     batch_inputs.extend(cache_list)


    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )


    print(f'{Colors.GREEN}Processing results...{Colors.RESET}')

    output_path = output_path

    os.makedirs(output_path, exist_ok=True)


    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]

    mmd_det_path = output_path + '/' + name_without_ext + '_det.md'
    mmd_path = output_path + '/' + name_without_ext + '.md'
    pdf_out_path = output_path + '/' + name_without_ext + '_layouts.pdf'
    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0
    skipped_pages = 0
    for output, img in zip(outputs_list, images):
        try:
            content = output.outputs[0].text
            
            if args.debug:
                print(f'{Colors.BLUE}[DEBUG] Page {jdx} output (first 200 chars): {content[:200]}...{Colors.RESET}')
                print(f'{Colors.BLUE}[DEBUG] Page {jdx} output length: {len(content)} chars{Colors.RESET}')
            
            # Check for incomplete generation (repetition without proper EOS)
            has_proper_eos = '<｜end▁of▁sentence｜>' in content or content.endswith('<|end|>') or content.endswith('</s>')
            
            if not has_proper_eos and SKIP_REPEAT:
                # Check if content seems incomplete (very short or ends abruptly)
                if len(content) < 50 or content.count('\n') < 2:
                    print(f'{Colors.YELLOW}Skipping page {jdx} (incomplete generation, length: {len(content)}){Colors.RESET}')
                    skipped_pages += 1
                    continue
                else:
                    # Content seems substantial, keep it even without proper EOS
                    print(f'{Colors.BLUE}Warning: page {jdx} has no proper EOS but content seems complete (length: {len(content)}), processing anyway{Colors.RESET}')

            # Clean up EOS markers
            content = content.replace('<｜end▁of▁sentence｜>', '')
            content = content.replace('<|end|>', '')
            content = content.replace('</s>', '')

            
            page_num = f'\n<--- Page Split --->'

            contents_det += content + f'\n{page_num}\n'

            image_draw = img.copy()

            matches_ref, matches_images, mathes_other = re_match(content)
            # print(matches_ref)
            result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_path)


            draw_images.append(result_image)


            for idx, a_match_image in enumerate(matches_images):
                content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

            for idx, a_match_other in enumerate(mathes_other):
                content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')


            contents += content + f'\n{page_num}\n'


            jdx += 1
        except Exception as e:
            print(f'{Colors.RED}Error processing page {jdx}: {e}{Colors.RESET}')
            import traceback
            traceback.print_exc()
            continue

    print(f'{Colors.YELLOW}Saving results... (processed {jdx} pages, skipped {skipped_pages} pages){Colors.RESET}')

    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)


    if draw_images:
        pil_to_pdf_img2pdf(draw_images, pdf_out_path)
        print(f'{Colors.GREEN}Done! Results saved to {output_path}{Colors.RESET}')
    else:
        print(f'{Colors.YELLOW}Warning: No images to save to PDF (all pages may have been skipped){Colors.RESET}')
    
    # Clean up LLM engine to avoid "died unexpectedly" error
    try:
        del llm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
