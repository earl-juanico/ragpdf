# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

# New imports for chart detection
from PIL import Image, ImageFilter
import numpy as np
import io

def _open_pil_from_entry(entry, base_dir=None):
    """
    Try to obtain a PIL.Image from a variety of possible image-list entry formats.
    Returns None if can't open.
    """
    try:
        if entry is None:
            return None
        # direct PIL Image
        if isinstance(entry, Image.Image):
            return entry
        # bytes
        if isinstance(entry, (bytes, bytearray)):
            return Image.open(io.BytesIO(entry)).convert("RGB")
        # path-like or string
        if isinstance(entry, (str, Path)):
            p = Path(entry)
            if base_dir and not p.is_absolute():
                p = Path(base_dir) / p
            if p.exists():
                return Image.open(p).convert("RGB")
            return None
        # dict-like structures that may contain image/path bytes
        if isinstance(entry, dict):
            for key in ("image", "img", "path", "file", "src", "data"):
                if key in entry:
                    return _open_pil_from_entry(entry[key], base_dir=base_dir)
        return None
    except Exception:
        return None

def is_chart_image(entry, *args, **kwargs):
    """
    Consider an image a chart only when its surrounding text matches the two
    sentinel phrases:
      - immediately follows "than similar nearby homes"
      - immediately precedes "Your annual electricity use"

    The function accepts either a dict-like entry carrying context (preferred)
    or a PIL.Image (in which case we cannot determine context and return False).
    Returns (bool, info).
    """
    try:
        # try extract surrounding text from common keys in dict-like entry
        before = after = ""
        if isinstance(entry, dict):
            before = entry.get("text_before") or entry.get("before") or entry.get("caption_before") or entry.get("context_before") or ""
            after = entry.get("text_after") or entry.get("after") or entry.get("caption_after") or entry.get("context_after") or ""
            # fallback to a single 'context' field if present
            if not before and "context" in entry:
                ctx = entry.get("context") or ""
                # if context contains both phrases, split around the image marker if present
                before = ctx
            if not after and "context" in entry:
                after = entry.get("context") or ""
        else:
            # no textual context available -> cannot treat as chart
            return False, {"reason": "no surrounding text available"}

        before_l = (before or "").lower()
        after_l = (after or "").lower()

        if "than similar nearby homes" in before_l and "your annual electricity use" in after_l:
            return True, {"reason": "matched_surrounding_phrases"}
        return False, {"reason": "phrases_not_matched", "before_preview": before_l[:120], "after_preview": after_l[:120]}
    except Exception as e:
        return False, {"error": str(e)}

def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-http-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable)

            for idx, model_list in enumerate(infer_results):
                model_json = copy.deepcopy(model_list)
                pdf_file_name = pdf_file_names[idx]
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

                images_list = all_image_lists[idx]
                pdf_doc = all_pdf_docs[idx]
                _lang = lang_list[idx]
                _ocr_enable = ocr_enabled_list[idx]

                # Filter images_list in-place: keep only images classified as charts.
                filtered_images = []
                for entry in images_list:
                    pil = _open_pil_from_entry(entry, base_dir=local_image_dir)
                    if pil is None:
                        # Could not open -> keep (conservative) but log
                        logger.debug(f"Could not open image-list entry, keeping entry: {type(entry)}")
                        filtered_images.append(entry)
                        continue
                    is_chart, info = is_chart_image(pil)
                    if is_chart:
                        filtered_images.append(entry)
                    else:
                        logger.info(f"Filtered out non-chart image for {pdf_file_name}: info={info}")

                # use filtered_images so only chart images are written and referenced downstream
                middle_json = pipeline_result_to_middle_json(model_list, filtered_images, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)

                pdf_info = middle_json["pdf_info"]

                pdf_bytes = pdf_bytes_list[idx]
                _process_output(
                    pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                    md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                    f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                    f_make_md_mode, middle_json, model_json, is_pipeline=True
                )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url)

            pdf_info = middle_json["pdf_info"]

            _process_output(
                pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                f_make_md_mode, middle_json, infer_result, is_pipeline=False
            )


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=True
):
    """处理输出文件"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to "pipeline"
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-transformers: More general.
            vlm-vllm-engine: Faster(engine).
            vlm-http-client: Faster(client).
            without method specified, pipeline will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to "pipeline".
        server_url: When the backend is `http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    # args
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(__dir__, "output")
    pdf_suffixes = ["pdf"]
    image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg"]

    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if guess_suffix_by_path(doc_path) in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    """Use pipeline mode if your environment does not support VLM"""
    parse_doc(doc_path_list, output_dir, backend="pipeline")

    """To enable VLM mode, change the backend to 'vlm-xxx'"""
    # parse_doc(doc_path_list, output_dir, backend="vlm-transformers")  # more general.
    # parse_doc(doc_path_list, output_dir, backend="vlm-mlx-engine")  # faster than transformers in macOS 13.5+.
    # parse_doc(doc_path_list, output_dir, backend="vlm-vllm-engine")  # faster(engine).
    # parse_doc(doc_path_list, output_dir, backend="vlm-http-client", server_url="http://127.0.0.1:30000")  # faster(client).