import argparse
import time
import os
import torch
import torch.nn.functional as F
from utils.tools import Summary, AverageMeter, accuracy, set_random_seed
from data.cls_to_names import custom_scale
from clip import clip
from data.datautils import build_test_loader
import wandb
from datetime import datetime
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

@torch.no_grad()
def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

@torch.no_grad()
def param_estimation(added_sample, banks, initial_mean,  prev_mus, alpha):
    """Online Gaussian distribution parameter estimation with Constructed Knowledge Banks."""
    with torch.no_grad():
        image_features, pred, img_pro = added_sample
        vecs, labels, cache_pro = banks
        cache_keys = torch.unique(labels)

        mus = prev_mus.clone()
        mask = labels==pred
        selected_vecs = vecs[mask]  # (M, D)
        selected_cache_pro = cache_pro[mask, pred].unsqueeze(1)  # (M, 1)


        # update mean with added_samples in Constructed Knowledge Banks
        new_mu = ((selected_cache_pro * selected_vecs).sum(dim=0) + img_pro[0][pred] * image_features[0])/ (selected_cache_pro.sum() +img_pro[0][pred]).unsqueeze(0)
        new_mu = alpha * new_mu + (1 - alpha) * initial_mean[pred]
        mus[pred] = new_mu

        # KS Estimator (Bayes ridge-type estimator)
        center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in cache_keys])### [num_samples, dim]
        n, d = center_vecs.shape
        if n == 1:
            Sigma = torch.eye(d).cuda()
        else:
            Sigma = center_vecs.T.cov()
        trace = Sigma.trace()
        cov_inv = d * torch.linalg.pinv((n - 1) * Sigma + trace * torch.eye(d).cuda())

        ps = torch.ones(initial_mean.shape[0]).cuda() * 1. / initial_mean.shape[0]

        W = torch.einsum('nd, dc -> cn', mus, cov_inv)
        b = ps.log() - torch.einsum('nd, dc, nc -> n', mus, cov_inv, mus) / 2

        cache_values = (F.one_hot(torch.Tensor(labels).to(torch.int64), num_classes=initial_mean.shape[0])).cuda().half()
        return W, b, mus, image_features @ vecs.T, cache_pro * cache_values

### Constructed Knowledge Banks
def update_knowledge_banks(banks, features_loss, bank_size):
    """update vecs, labels, cache_pro"""
    pred, feature, loss, prob_map = features_loss
    cache_vecs, cache_labels, cache_cache_pro, cache_loss = banks

    start_idx = pred * bank_size
    end_idx = start_idx + bank_size

    existing_count = (cache_labels == pred).sum().item()
    update = False
    if existing_count < bank_size:
        insert_idx = start_idx + existing_count
        update = True
    else:
        max_loss_value, max_loss_idx = cache_loss[start_idx:end_idx].max(dim=0)

        if loss < max_loss_value:
            insert_idx = start_idx + max_loss_idx.item()
            update = True

    if update:
        cache_vecs[insert_idx] = feature
        cache_labels[insert_idx] = pred
        cache_cache_pro[insert_idx] = prob_map
        cache_loss[insert_idx] = loss

    return update, [cache_vecs, cache_labels, cache_cache_pro, cache_loss], [feature,pred,prob_map]


@torch.no_grad()
def get_clip_feature_logits(images, clip_weights,encoder):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = encoder(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        clip_logits = 100. * image_features.float() @ clip_weights.float()

        if image_features.size(0) > 1:# if perform aug, do the 0.1 selection
            batch_entropy = calculate_batch_entropy(clip_logits)
            selected_idx = torch.topk(batch_entropy, max(int(batch_entropy.size(0) * 0.1), 1), largest=False).indices
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = calculate_batch_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

    return image_features.float(), clip_logits, loss, prob_map, pred


@torch.no_grad()
def evaluation(val_loader, clip_weights, image_encoder, args):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    initial_mean = clip_weights.T
    mean = None
    banks,accuracies = {},[]

    # ==============  Initialize Knowledge Banks ==============
    cls_num, dim = clip_weights.shape[1], clip_weights.shape[0]
    cache_vecs = torch.zeros((cls_num * args.bank_size, dim)).cuda()
    cache_labels = torch.full((cls_num * args.bank_size,), -1, dtype=torch.long).cuda()
    cache_pro = torch.zeros((cls_num * args.bank_size, cls_num)).cuda()
    cache_loss = torch.full((cls_num * args.bank_size,), float('inf')).cuda()
    cache = [cache_vecs, cache_labels, cache_pro, cache_loss]

    start_time = time.time()
    # ==============  cache update & online parameter estimation & finial prediction ==============
    for i, (images, target) in enumerate(tqdm(val_loader, desc='Processed test images: ')):
        # ==============  calculate CLIP logits for each image ==============
        image_features, clip_logits, loss, prob_map, pred = get_clip_feature_logits(images, clip_weights, image_encoder)
        target = target.cuda()

        ## Constructed Knowledge Banks
        update_sign, cache, added_sample = update_knowledge_banks(cache, [pred, image_features, loss, prob_map], args.bank_size)

        ### Parameter Estimation
        if mean is None:
            mean = clip_weights.T.float()

        if update_sign:
            valid_mask = cache[1] != -1
            banks = [t[valid_mask] for t in cache[:3]]
            W, b, mean, similarity_matrix, cache_logits = param_estimation(added_sample, banks, initial_mean, prev_mus=mean, alpha=args.alpha)

        GDA_logits = image_features @ W + b

        test_logits = compute_final_prediction(clip_logits.clone(), GDA_logits, similarity_matrix, cache_logits)

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
    # Initialize CLIP model
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

        # ============================clip_weights  ============================
        if args.GPT:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = f"./pre_extracted_class_feat/{args.arch.replace('/', '')}/GPT_w_{args.class_type}_class_emb"
        else:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = f"./pre_extracted_class_feat/{args.arch.replace('/', '')}/{args.class_type}_class_emb"

        clip_weights = torch.load(os.path.join(clip_weights_dir, f"{dataset_name}.pth"))

        if dataset_name =='imagenet_c':
            corruption_type = args.corruption.split('/')
            for corrup in corruption_type:
                run_name = f"{dataset_name}_{corrup}_Online"
                run = wandb.init(project="ADAPT_NeurIPS25", config=args_dict_param, group=group_name, name=run_name)
                val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=1, corruption=corrup, level=args.level)

                # testing start
                acc = evaluation(val_loader, clip_weights, clip_model.encode_image, args)
                wandb.log({f"{dataset_name}": acc})
                run.finish()

        else:

            run_name = f"{dataset_name}_Online"
            run = wandb.init(project="ADAPT_NeurIPS25", config=args_dict_param, group=group_name, name=run_name)
            val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=1)

            # testing start
            acc = evaluation(val_loader, clip_weights, clip_model.encode_image, args)
            wandb.log({f"{dataset_name}": acc})
            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADAPT Online Evaluation')
    parser.add_argument('--data', metavar='DIR', default='./datasets/TPT/', help='path to dataset root')
    parser.add_argument('--test_set', type=str, default='imagenet', help='dataset name')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help=" CLIP model backbone:'RN50' or'ViT-B/16'.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bank_size', type=int, default=16, help="Bank Size L")
    parser.add_argument('--alpha', type=float, default=0.9, help="the alpha for EMA")

    ### ImageNet-c
    parser.add_argument('--level', type=str, default='5', help="Corruption Level")
    parser.add_argument('--corruption', type=str, default='gaussian_noise/shot_noise/impulse_noise/defocus_blur/glass_blur/motion_blur/zoom_blur/snow/frost/fog/brightness/contrast/elastic_transform/pixelate/jpeg_compression', help="corruption type for ImageNet-c")

    ### class embedding
    parser.add_argument('--class_type', default='Custom', type=str, help=" Type of the initialization of mean matrix: Custom, Vanilla, Img_temp, Ensemble")
    parser.add_argument('--GPT', action='store_true', default=True, help="use the description or not ")

    args = parser.parse_args()
    set_random_seed(args.seed)
    main_worker(args)