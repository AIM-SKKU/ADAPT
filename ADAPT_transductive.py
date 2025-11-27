import argparse
import time
import os
import torch
import torch.nn.functional as F
from utils.tools import Summary, AverageMeter, accuracy, set_random_seed
from data.cls_to_names import custom_scale
from data.datautils import build_test_loader
from clip import clip
import wandb
from datetime import datetime
from tqdm import tqdm



def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


@torch.no_grad()
def param_estimation(image_features, banks, initial_mean, alpha, text_sample_prob):
    """Gaussian distribution parameter estimation with Constructed Knowledge Banks."""

    with torch.no_grad():
        sorted_classes = sorted(banks.keys())

        vecs = torch.cat([item[0].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0)  # (N, feature_dim)
        labels = torch.tensor([class_idx for class_idx in sorted_classes for _ in banks[class_idx]])  # (N)
        cache_pro = torch.cat([item[2].unsqueeze(0) for class_idx in sorted_classes for item in banks[class_idx]], dim=0)  # (N, num_classes)

        # update mean
        mus = torch.cat([(((cache_pro[labels == i][:, i].unsqueeze(1) * vecs[labels == i]).sum(dim=0) + (image_features * text_sample_prob[:,i].unsqueeze(1)).sum(dim=0)) / ((cache_pro[labels == i][:, i].sum()) + text_sample_prob[:,i].sum())).unsqueeze(0) if i in banks.keys() else initial_mean[i].unsqueeze(0) for i in range(initial_mean.shape[0])])
        mus = alpha * mus + (1 - alpha) * initial_mean

        # KS Estimator (Bayes ridge-type estimator)
        center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in banks.keys()])
        cov_inv = center_vecs.shape[1] * torch.linalg.pinv((center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(center_vecs.shape[1]).cuda())

        ps = torch.ones(initial_mean.shape[0]).cuda() * 1. / initial_mean.shape[0]
        W = torch.einsum('nd, dc -> cn', mus, cov_inv)
        b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

        cache_values = (F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=initial_mean.shape[0])).cuda().half()#
        return W, b, mus, image_features @ vecs.T, cache_pro * cache_values


@torch.no_grad()
def constructed_knowledge_banks(preds, features_loss, bank_size):
    """
    Update Knowledge Banks by selecting the top 'bank_size' samples for each class based on entropy loss.

    Args:
        cache (dict): Dictionary storing features per class.
        preds (Tensor): Predicted labels for the batch. Shape: (batch_size,)
        features_losses (tuple): (image_features, loss, prob_map), each of shape:
                                 - image_features: (batch_size, feature_dim)
                                 - loss: (batch_size,)
                                 - prob_map: (batch_size, num_classes)
        cbank_size (int): Maximum number of samples to store per class.

    Returns:
        bool: Whether the cache was updated.
    """
    cache = {}
    with torch.no_grad():
        image_features, losses, prob_maps = features_loss
        unique_preds = preds.unique(sorted=True)
        for pred in unique_preds:
            pred = pred.item()
            idxs = (preds == pred).nonzero(as_tuple=True)[0]

            if len(idxs) == 0:
                continue
            if len(idxs) <= bank_size:
                selected_items = [(image_features[i], losses[i].item(), prob_maps[i]) for i in idxs]
            else:
                top_k = losses[idxs].topk(min(len(idxs), bank_size), largest=False)[1]  # top bank_size
                selected_idxs = idxs[top_k]
                selected_items = [(image_features[i], losses[i].item(), prob_maps[i]) for i in selected_idxs]
            cache[pred] = selected_items
    return cache

def process_in_chunk(image_features, clip_weights, chunk_size =64):
    num_samples = image_features.shape[0]
    processed_features, processed_logits = [], []

    for i in range(0, num_samples, chunk_size):
        batch_feat = image_features[i: i + chunk_size]
        batch_logits = 100. * batch_feat.float() @ clip_weights.float()
        batch_entropy = calculate_batch_entropy(batch_logits)
        selected_idx = torch.topk(batch_entropy, max(int(batch_entropy.size(1) * 0.1), 1), dim=1, largest=False).indices
        batch_indices = torch.arange(batch_feat.size(0)).unsqueeze(1).expand_as(selected_idx)
        batch_selected_feat = batch_feat[batch_indices, selected_idx].mean(dim=1)
        batch_selected_logits = batch_logits[batch_indices, selected_idx].mean(dim=1)
        processed_features.append(batch_selected_feat)
        processed_logits.append(batch_selected_logits)
    image_features = torch.cat(processed_features, dim=0)  # [N, D]
    clip_logits = torch.cat(processed_logits, dim=0)  # [N, C]

    return image_features, clip_logits


@torch.no_grad()
def get_clip_logits(image_features, clip_weights):
    if len(image_features.shape)> 2:# if perform aug, do the 0.1 selection
        image_features, clip_logits = process_in_chunk(image_features, clip_weights, chunk_size=64)
        loss = calculate_batch_entropy(clip_logits)
        prob_map = clip_logits.softmax(dim=1)
        pred = clip_logits.argmax(dim=1)
    else:
        clip_logits = 100. * image_features.float() @ clip_weights.float()
        loss = calculate_batch_entropy(clip_logits)
        prob_map = clip_logits.softmax(1)
        pred = clip_logits.argmax(dim=1)
    return image_features, clip_logits, loss, prob_map, pred


@torch.no_grad()
def evaluation(clip_weights, val_loader, clip_model, dataset_name, args):

    start_time = time.time()
    #==============  data feature extraction  ==============
    dim = clip_weights.shape[0]
    image_features, clip_logits, loss, prob_map, preds, target = [], [], [], [], [], []

    for images, label in tqdm(val_loader):
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda(non_blocking=True)
        else:
            images = images.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            if dataset_name in ['imagenet_a', 'imagenet_sketch', 'imagenet_r', 'imagenetv2']:
                image_fet = clip_model.encode_image(images).view(64, -1, dim).permute(1, 0, 2)
            else:
                image_fet = clip_model.encode_image(images)
            image_fet = image_fet / image_fet.norm(dim=-1, keepdim=True)
        image_features.append(image_fet)
        target.append(label)

    #==============  Calculate CLIP logits for each image ==============
    image_features = torch.cat(image_features, dim=0)
    target = torch.cat(target, dim=0)
    image_features, clip_logits, loss, prob_map, preds = get_clip_logits(image_features,clip_weights)


    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    initial_mean = clip_weights.T
    accuracies = []
    # ==============  Constructed Knowledge Banks ==============
    banks = constructed_knowledge_banks(preds, [image_features, loss, prob_map], args.bank_size)

    # ==============  parameters estimation ==============
    W, b, mean, similarity_matrix, cache_logits = param_estimation(image_features, banks, initial_mean, args.alpha, text_sample_prob=prob_map)
    test_logits = clip_logits.clone()
    GDA_logits = image_features.float() @ W + b

    # ==============  finial prediction ==============
    test_logits = compute_final_prediction(test_logits, GDA_logits, similarity_matrix, cache_logits)

    acc = accuracy(test_logits, target, topk=(1,))
    top1.update(acc[0], 1)
    accuracies.append(acc[0].item())
    wandb.log({"Averaged test accuracy": round(sum(accuracies) / len(accuracies), 2)}, commit=True)


    # measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    wandb.log({"Elapsed time": f"{elapsed_time:.2f} seconds"}, commit=True)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return sum(accuracies) / len(accuracies)


def compute_final_prediction(clip_logits, GDA_logits, similarity_matrix, cache_logits):
    """
    param:
        clip_logits: [N, K], CLIP logits
        GDA_logits: [N, K], GDA logits
        cache_logits: [M, K], logits of Cache samples (M: Cache size)
        similarity_matrix: [N, M], similarity between test sample with Cache samples
    retuen:
        final_logits: [N, K], finial logits
    """
    GDA_logits = torch.log_softmax(GDA_logits, dim=1)#### log P
    intermediate = GDA_logits
    intermediate += (args.scale / (len(cache_logits) * 2)) * (similarity_matrix @ cache_logits)
    # For numerical stability
    intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
    final_logits = clip_logits * torch.exp(1 / args.scale * intermediate)

    return final_logits


def main_worker(args):
    print("=> Model created: visual backbone {}".format(args.arch))
    clip_model, preprocess = clip.load(args.arch)
    clip_model.eval()

    datasets = args.test_set.split('/')
    for dataset_name in datasets:
        print("Extracting features for: {}".format(dataset_name))
        args.scale = custom_scale[dataset_name]
        args_dict = vars(args)
        args_dict_param = {k: v for k, v in args_dict.items() if k != 'test_set'}
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.arch}_{dataset_name}_{date}"


        #============================clip_weights  ============================
        print("Evaluating: {}".format(dataset_name))
        if args.GPT:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = f"./pre_extracted_class_feat/{args.arch.replace('/', '')}/GPT_w_{args.class_type}_class_emb"
        else:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = f"./pre_extracted_class_feat/{args.arch.replace('/', '')}/{args.class_type}_class_emb"

        clip_weights = torch.load(os.path.join(clip_weights_dir, f"{dataset_name}.pth"))

        if dataset_name == 'imagenet_c':
            corruption_type = args.corruption.split('/')
            for corrup in corruption_type:
                run_name = f"{dataset_name}_{corrup}_Transductive"
                run = wandb.init(project="ADAPT_NeurIPS25", config=args_dict_param, group=group_name, name=run_name)
                val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=args.bt, corruption=corrup, level=args.level)

                # testing start
                acc = evaluation(clip_weights, val_loader, clip_model, dataset_name, args)
                wandb.log({f"{dataset_name}": acc})
                run.finish()
        else:
            run_name = f"{dataset_name}_Transductive"
            run = wandb.init(project="ADAPT_NeurIPS25", config=args_dict_param, group=group_name, name=run_name)
            val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=args.bt)

            # testing start
            acc = evaluation(clip_weights, val_loader, clip_model, dataset_name, args)
            wandb.log({f"{dataset_name}": acc})
            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADAPT Transductive Evaluation')
    parser.add_argument('--data', metavar='DIR', default='./datasets/TPT/', help='path to dataset root')
    parser.add_argument('--test_set', type=str, default='imagenet', help='dataset name')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help=" CLIP model backbone:'RN50' or'ViT-B/16'.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bank_size', type=int, default=6, help="The bank size L")
    parser.add_argument('--alpha', type=float, default=0.9, help="the alpha for EMA")

    ### ImageNet-c
    parser.add_argument('--level', type=str, default='5', help="Corruption Level")
    parser.add_argument('--corruption', type=str, default='gaussian_noise/shot_noise/impulse_noise/defocus_blur/glass_blur/motion_blur/zoom_blur/snow/frost/fog/brightness/contrast/elastic_transform/pixelate/jpeg_compression', help="corruption type for ImageNet-c")

    ### class embedding
    parser.add_argument('--class_type', default='Custom', type=str, help=" Type of the initialization of mean matrix: Custom, Vanilla, Img_temp, Ensemble")
    parser.add_argument('--GPT', action='store_true', default=True, help="use the description or not ")
    parser.add_argument('--bt', type=int, default=64, help="the batch size of test data loader")


    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)