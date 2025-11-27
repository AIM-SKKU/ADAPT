data_root=DATA_PATH
testsets=imagenet/imagenet_a/imagenetv2/imagenet_r/imagenet_sketch/fgvc_aircraft/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101/imagenet_c

bt=64

python -u ./ADAPT_transductive.py \
        --data ${data_root} \
        --test_set ${testsets} \
        --arch ViT-B/16 \
        --bank_size 6 \
        --alpha 0.9 \
        --bt ${bt} \
        --class_type Custom \
        --GPT