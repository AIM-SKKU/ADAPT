data_root=DATA_PATH
testsets=imagenet/imagenet_a/imagenetv2/imagenet_r/imagenet_sketch/fgvc_aircraft/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101/imagenet_c

python -u ./ADAPT_online.py \
        --data ${data_root} \
        --test_set ${testsets} \
        --arch ViT-B/16 \
        --bank_size 16 \
        --alpha 0.9 \
        --class_type Custom \
        --GPT