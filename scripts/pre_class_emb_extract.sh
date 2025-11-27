testsets=imagenet_a/imagenet/imagenet_sketch/imagenet_r/imagenetv2/dtd/oxford_flowers/food101/stanford_cars/sun397/fgvc_aircraft/oxford_pets/caltech101/ucf101/eurosat/imagenet_c

data_root=DATA_PATH
descriptor_root=GPT_DESCRIPTION_PATH

## [descriptors refernce] https://github.com/MCG-NJU/AWT/tree/main/AWT_zero_shot/descriptions/image_datasets)

python -u ./Pre_extract_class_emb_default.py \
        --data ${data_root} \
        --test_set ${testsets} \
        --vlm_name CLIP \
        --arch ViT-B/16 \
        --seed 0 \
        --descriptor_path ${descriptor_root} \
        --class_type Custom \
        --GPT