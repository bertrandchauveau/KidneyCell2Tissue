# KidneyCell2Tissue: Molecular Tissue Segmentation Through Fusion of Spatial Transcriptomics with Conventional Histology


## 🚀 About
While the Banff classification remains the gold standard for diagnosing kidney allograft rejection, it does not capture the immune cell composition within tissue architecture. Recent advancements in spatial transcriptomics, such as Xenium In Situ (10x Genomics), offer high-resolution molecular insights but face a key analytical challenge: integrating single cell annotations into broader, individualized tissue compartments (e.g., tubules or capillaries), which cannot be reliably defined from the molecular profiles and by single-cell data alone.
To address this gap, we developed a semi-automated analytical pipeline combining deep learning-based instance segmentation models (YOLO26) with advanced image registration algorithms. By integrating H&E histology images with molecularly derived cell annotations, we generated robust molecularly enriched instance segmentation of kidney tissue compartments. This approach achieved high concordance rates for major structures, including glomeruli, tubules, arteries/arterioles, and peritubular capillaries, while enabling discrimination between lymphatic and non-lymphatic capillaries, and among distinct tubular segments. This scalable framework provides a foundation for linking cellular phenotypes to tissue architecture, enabling more comprehensive investigations of kidney biology and disease using spatial transcriptomic technologies.

## Models
- Available with usage instructions as Hugging Face checkpoints [here](https://huggingface.co/collections/bertrandchauveau/xenium-kidney-tissue-segmentation)

## Creating a 6-channel image from H&E and molecular whole cell annotations

Example:

```python multispectral.py --rgb_dir /content/ --mask_dir /content/ --output_dir /content/output --rgb_name example_he.png --mask_name example_molecular.png```

## Creating cell semantic segmentation tiles from molecular-based cell annotations as geojson

Example:

```python cell_mask_from_geojson.py \
    --geojson_path /content/\
    --geojson_name example.geojson \
    --wsi_name my_slide.tiff \
    --wsi_height 15000 \
    --wsi_width 15000 \
    --output_dir ./my_tiles \
    --tile_size 1024 \
    --overlap 0.125```
