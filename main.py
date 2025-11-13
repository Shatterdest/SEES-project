import torch
import copy, math, pickle, json, os
import bertviz, uuid
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
import argparse
import os
import random
import urllib.parse
from dataclasses import dataclass


@dataclass
class Token:
    name: str
    probability: float
        
@dataclass
class Cluster:
    id: int
    points: list
    entropy: float
    strength: list
    average_strength: float
   

@dataclass
class Entropy:
    token: float
    general: float
        
@dataclass
class Result:
    filename: str
    tokens: [Token]
    clusters: [Cluster]
    entropy: Entropy
        
result = Result(
    filename = "",
    tokens = [],
    clusters = [],
    entropy = Entropy(
        token = 0,
        general = 0
    ),
)


from PIL import Image
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    center_crop)
from transformers.image_utils import (
    infer_channel_dimension_format,
    to_numpy_array)
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.cluster import DBSCAN
from collections import Counter
from dataclasses import dataclass
import csv
import json
from typing import List

LAYER_NUM = 32
HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_DIM = HEAD_NUM * HEAD_DIM
# Saad edited-----------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class PromptEntry:
    question  : str
    prefix    : str
    prompt    : str
    image_url : str

def load_prompts(csv_path: str = "results.csv") -> List[PromptEntry]:
    entries: List[PromptEntry] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(PromptEntry(
                question  = row["question"],
                prefix    = row["prefix"],
                prompt    = row["prompt"],
                image_url = row["image_url"],
            ))
    print(f"✅ Loaded {len(entries)} prompts from {csv_path}")
    return entries
#----------------------------------------------------------------------------------------------------------------------------------

# class ImagePrompt has been commented out
class ImagePrompt:
    image_url: str
    prompt: str
    prefix: str

#@dataclass
#class Entropy:
#    cluster: int
#    count_points: int
#    average_strength: float

def normalize(vector):
    max_value = max(vector)
    min_value = min(vector)
    vector1 = [(x-min_value)/(max_value-min_value) for x in vector]
    vector2 = [x/sum(vector1) for x in vector1]
    return vector2


def transfer_output(model_output):
    all_pos_layer_input = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []

    for layer_i in range(LAYER_NUM):
        cur_layer_input = model_output[layer_i][0]
        cur_layer_output = model_output[layer_i][4]
        cur_last_attn_subvalues = model_output[layer_i][5]

        all_pos_layer_input.append(cur_layer_input[0].tolist())
        all_pos_layer_output.append(cur_layer_output[0].tolist())
        all_last_attn_subvalues.append(cur_last_attn_subvalues[0].tolist())

    return all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues

def get_bsvalues(vector, model, final_var):
    vector = vector * torch.rsqrt(final_var + 1e-6)
    vector_rmsn = vector * model.language_model.norm.weight.data
    vector_bsvalues = model.language_model.lm_head(vector_rmsn).data
    return vector_bsvalues

def get_prob(vector):
    prob = torch.nn.Softmax(-1)(vector)
    return prob

def transfer_l(l):
    new_x, new_y = [], []
    for x in l:
        new_x.append(x[0])
        new_y.append(x[1])
    return new_x, new_y

def plt_bar(x, y, yname="log increase"):
    x_major_locator=MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a/2 for a in x]
    plt.xlim(-0.5, plt_x[-1]+0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i%2 == 0:
            x_attn.append(x[i]/2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i]/2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.show()

def plt_heatmap(data):
    xLabel = range(len(data[0]))
    yLabel = range(len(data))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    #plt.colorbar(im)
    plt.title("attn head log increase heatmap")
    plt.show()


class LlavaMechanism:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """
        Initialize the LlavaMechanism class by loading the model and processor.
        
        Args:
            model_id (str): The model ID to load
            device (str): Device to run the model on
        """
        # Setup CUDA
        torch.set_default_device('cuda')
        
        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            low_cpu_mem_usage=True, 
            revision='a272c74'
            # output_attentions=True, 
            # output_hidden_states=True
        ).to(device)
        
        print(f"Model loaded on {self.model.device}")
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')
        self.processor.patch_size = 14 # added
        print(f"Model loaded on {self.model.device}")
        self.model.eval()
        
        # Create output directory for saved visualizations
        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)    
        
    def get_attention_patches(self, image, prompt, prefix):
        """
        Get attention patches for an image with a specific prompt.
        
        Args:
            image (PIL.Image): Input image
            prompt (str): The prompt text
            prefix (str): The prefix text after ASSISTANT tag
            
        Returns:
            tuple: (demo_img, increase_scores_normalize)
        """
        t = time.time()
        
        # Process input
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT: {prefix}"
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()} # added
        outputs = self.model(**inputs)
        # with torch.no_grad():
        #     outputs = self.model(
        #         **inputs, 
        #         return_dict_in_generate=True, 
        #         output_attentions=True,
        #         output_hidden_states=True,
        #         max_new_tokeans=1
        #     )
        print(f'Finished inference time {time.time() - t}')
        
        # Get output probabilities
        outputs_probs = get_prob(outputs["logits"][0][-1])
        outputs_probs_sort = torch.argsort(outputs_probs, descending=True)
        token_things = [self.processor.decode(x) for x in outputs_probs_sort[:10]]
        print(token_things)
        print(outputs_probs_sort[:10].tolist())
        
        # Process model outputs
        all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues = transfer_output(outputs[2])
        print(f'Finished transfer output time {time.time() - t}')
        final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)
        
        # Process image
        resample = 3
        crop_size = {"height": 336, "width": 336}
        image_convert = convert_to_rgb(image)
        image_numpy = to_numpy_array(image_convert)
        input_data_format = infer_channel_dimension_format(image_numpy)
        output_size = get_resize_output_image_size(image_numpy, size=336,
                     default_to_square=False, input_data_format=input_data_format)
        image_resize = resize(image_numpy, output_size, resample=resample, input_data_format=input_data_format)
        image_center_crop = center_crop(image_resize, size=(crop_size["height"], crop_size["width"]), input_data_format=input_data_format)
        
        demo_img = image_center_crop
        
        predict_index = outputs_probs_sort[0].item()
        print(predict_index, self.processor.decode(predict_index))
        
        # Calculate head-level increase
        all_head_increase = []
        for test_layer in range(LAYER_NUM):
            cur_layer_input = torch.tensor(all_pos_layer_input[test_layer])
            cur_v_heads = torch.tensor(all_last_attn_subvalues[test_layer])
            cur_attn_o_split = self.model.language_model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
            cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
            cur_attn_subvalues_head_sum = torch.sum(cur_attn_subvalues_headrecompute, 0)
            cur_layer_input_last = cur_layer_input[-1]
            origin_prob = torch.log(get_prob(get_bsvalues(cur_layer_input_last, self.model, final_var))[predict_index])
            cur_attn_subvalues_head_plus = cur_attn_subvalues_head_sum + cur_layer_input_last
            cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
                    cur_attn_subvalues_head_plus, self.model, final_var))[:, predict_index])
            cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
            for i in range(len(cur_attn_plus_probs_increase)):
                all_head_increase.append([str(test_layer)+"_"+str(i), round(cur_attn_plus_probs_increase[i].item(), 4)])
        print(f'Finished head-level increase time {time.time() - t}')
        
        all_head_increase_sort = sorted(all_head_increase, key=lambda x:x[-1])[::-1]
        
        # Get the top head and calculate position increase
        test_layer, head_index = all_head_increase_sort[0][0].split("_")
        test_layer, head_index = int(test_layer), int(head_index)
        cur_layer_input = outputs[2][test_layer][0][0]
        cur_v_heads = outputs[2][test_layer][5][0]
        cur_attn_o_split = self.model.language_model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
        cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
        cur_attn_subvalues_headrecompute_curhead = cur_attn_subvalues_headrecompute[:, head_index, :]
        cur_layer_input_last = cur_layer_input[-1]
        origin_prob = torch.log(get_prob(get_bsvalues(
            cur_layer_input_last, self.model, final_var))[predict_index])
        cur_attn_subvalues_headrecompute_curhead_plus = cur_attn_subvalues_headrecompute_curhead + cur_layer_input_last
        cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
            cur_attn_subvalues_headrecompute_curhead_plus, self.model, final_var))[:, predict_index])
        cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
        head_pos_increase = cur_attn_plus_probs_increase.tolist()
        curhead_increase_scores = head_pos_increase[5:581]
        increase_scores_normalize = normalize(curhead_increase_scores)
        print(f'Finished getting patches time {time.time() - t}')
        
        return demo_img, increase_scores_normalize, outputs_probs_sort, token_things
    
    def save_vis(self, demo_img, increase_scores_normalize, output_path=None):
        """
        Save a visualization of the original image with overlayed attention patches.
        
        Args:
            demo_img (numpy.ndarray): The image to visualize
            increase_scores_normalize (list): Normalized attention scores
            output_path (str, optional): Path to save the visualization. If None, will use default.
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "attention_analysis.png")
        else:
            output_path = os.path.join(self.output_dir, f"{output_path}.png")
        demo_img_h, demo_img_w, demo_img_c = demo_img.shape
        
        # Reshape and resize the attention scores
        demo_img_inc = np.array(increase_scores_normalize).reshape((24, 24))
        demo_img_inc = cv2.resize(demo_img_inc,
                                dsize=(demo_img_w, demo_img_h),
                                interpolation=cv2.INTER_CUBIC)
        
        # Create the visualization
        plt.figure(figsize=(25, 6))
        
        # Plot target image
        plt.subplot(1, 3, 1)
        plt.imshow(demo_img)
        plt.axis("off")
        plt.title("image")
        
        # Plot image with overlay
        plt.subplot(1, 3, 2)
        plt.imshow(demo_img)
        plt.imshow(demo_img_inc, alpha=0.8, cmap="gray")
        plt.axis("off")
        plt.title("log increase")
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")

def transform_matrix_to_3d_points(array_2d: np.ndarray):
    """Transforms a 2D numpy array to an array of (x, y, value) tuples, here (x, y) is the location of the value.

    Example:
        Input: [[0.3 1.7 2.5]
                [0.1 1.2 1.9]
               ]
        Output:
            [[0, 0, 0.3]
             [0, 1, 1.7]
             [0, 2, 2.5]
             [1, 0, 0.1]
             [1, 1, 1.2]
             [1, 2, 1.9]
            ]
    Args:
        array_2d: A 2D numpy array.

    Returns:
        A new numpy array where each element is a tuple (x, y, value).
    """    
    rows, cols = array_2d.shape    
    result = np.empty([rows * cols, 3], dtype=object)

    for x in range(rows):
        for y in range(cols):
            result[x * cols + y] = [y, -x + 23, array_2d[x, y]]

    return result

def find_clusters(attentions_with_locations: np.ndarray, eps: float, min_samples: int, metric: str="euclidean") -> (DBSCAN, int, int):
    """Find clusters from a given attention 3D points.

    Args:
        attentions_with_locations: a list of attentions with location info.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        metric: the custom metric to calculate distances between instances in the provided feature array.

    Returns:
        The DBSCAN object, the number of clusters and the number of noise points.
    """
    # Get the coordinates of the patches.
    x_coords = attentions_with_locations[:, 0]
    y_coords = attentions_with_locations[:, 1]
    coords = np.stack((x_coords, y_coords), axis=-1)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    return db, n_clusters_, n_noise_

def apply_threshold(datapoints: np.ndarray, percentile: float) -> np.ndarray:
    """Remove the lowest percentile scores.
    """
    z_values = datapoints[:, 2]

    # Calculate the percentile
    p_value = np.percentile(z_values, percentile)
    print(f"{percentile}th percentile value: {p_value}")

    return datapoints[datapoints[:, 2] > p_value]

def duplicate_points(datapoints: np.ndarray, min_dup: int, max_dup: int) -> np.ndarray:
    """ Duplicate datapoints so that large valkues duplicates more times than smaller values.
    """
    # Extract value (attention score)
    values = datapoints[:, 2]
   
    # Normalize value to get number of times to duplicate
    scaled = ((values - values.min()) / (values.max() - values.min()) * (max_dup - min_dup) + min_dup + 1).astype(int)
   
    # Duplicate each point according to its scaled weight
    weighted_points = np.concatenate([np.repeat([pt], rep, axis=0) for pt, rep in zip(datapoints, scaled)], axis=0)
   
    # print(weighted_points)
    print(f"Original points: {len(datapoints)} -> After weighting: {len(weighted_points)}")
    return weighted_points

def save_attentions(weighted_attentions_with_locations: np.ndarray, db: DBSCAN, image_url: str):
    parsed_url = urllib.parse.urlparse(image_url)
    filename = os.path.basename(parsed_url.path)

    plt.scatter(weighted_attentions_with_locations[:, 0], weighted_attentions_with_locations[:, 1], c=db.labels_)
    plt.show()
    plt.savefig(os.path.join("output_images", "attention_analysis_" + filename))
    plt.close()

from dataclasses import dataclass

@dataclass
class ClusterData:
    n_points: int
    ave_strength: float
    
def calculate_metrics(db: DBSCAN, weighted_attentions_with_locations):
    labels = db.labels_
    unique_clusters = set(labels) - {-1}  # Remove noise (-1)

    cluster_strengths = {}

    # Count the number of points per cluster
    cluster_counts = Counter(labels)

    print("Points per cluster:")
    for label, count in cluster_counts.items():
        if label != -1:
            result.clusters.append(Cluster(
                id = label,
                points = count,
                entropy = [],
                strength = [],
                average_strength = 0,
            ))

    # 
    z_values = weighted_attentions_with_locations[:, 2]

    for cluster in unique_clusters:
        strengths = z_values[labels == cluster]  # Get strength values for the cluster

        for c in result.clusters:
            if c.id == cluster:
                c.strength = strengths
                c.average_strength = np.mean(strengths)

        if cluster in cluster_strengths:
            cluster_strengths[cluster].ave_strength = np.mean(strengths)
        else:
            cluster_strengths[cluster] = ClusterData(0, np.mean(strengths))

    # Print average strength of each cluster
    print("Average Strength and Number of Points per Cluster:")
    for cluster, data in cluster_strengths.items():
        print(f"Cluster {cluster}: {data}")
        
    return cluster_strengths

def calculate_entropy(datapoints: list):
    flat_list = [item for sublist in datapoints for item in sublist]
    total_count = len(flat_list)
    counts = Counter(flat_list)
    probabilities = [count / total_count for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    max_entropy = np.log2(len(counts))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def calculate_token_entropy(token_list: list):
    total_count = len(token_list)
    counts = Counter(token_list)
    probabilities = [count / total_count for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    max_entropy = np.log2(len(counts))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def cluster_entropy(db:DBSCAN):
    labels = db.labels_
    unique_clusters = set(labels) - {-1}  # Remove noise (-1)

    cluster_strengths = {}

    # Count the number of points per cluster
    cluster_counts = Counter(labels)

    print("Points per cluster:")
    for label, count in cluster_counts.items():
        if label != -1:
            cluster_strengths[label] = ClusterData(count, 0)

    # 
    z_values = weighted_attentions_with_locations[:, 2]

    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]  # Get strength values for the cluster
        
        if cluster in cluster_strengths:
            total_count = len(cluster_points)
            counts = Counter(cluster_points)
            probabilities = [count / total_count for count in counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(len(counts))
            normalized_entropy = entropy / max_entropy
            cluster_strengths[cluster].ave_strength = normalized_entropy

    # Print Entropy of each cluster
    print("Entropy per Cluster:")
    for cluster, data in cluster_strengths.items():
        print(f"Cluster {cluster}: {data}")
        
    return cluster_strengths

class ResultEntry:
    def __init__(self, question_type, question, ground_truth, top_4_tokens, model_answer):
        self.question_type = question_type
        self.question      = question
        self.ground_truth  = ground_truth
        self.top_4_tokens  = top_4_tokens
        self.model_answer  = model_answer

    def __repr__(self):
        return (f"ResultEntry(type={self.question_type!r}, "
                f"gt={self.ground_truth!r}, "
                f"top4={self.top_4_tokens!r}, "
                f"pred={self.model_answer!r})")
# -------------------------------------------------------------------



def main():
    prompts   = load_prompts("results.csv")
    mechanism = LlavaMechanism()

    for i, entry in enumerate(prompts, start=1):
        print(f"\n[{i}/{len(prompts)}]")
        print(f"  Q     : {entry.question}")
        print(f"  Prefix: {entry.prefix}")
        print(f"  Prompt: {entry.prompt}")
        print(f"  URL   : {entry.image_url}")

        # 1) Fetch image
        image = Image.open(requests.get(entry.image_url, stream=True).raw)

        # Get attention patches
        demo_img, increase_scores_normalize, token_probability, outputs = mechanism.get_attention_patches(image, imagePrompt.prompt, imagePrompt.prefix)
        
        print(outputs)

        result.tokens = [
            Token(
                name = outputs[0],
                probability = token_probability[0],
            ),
            Token(
                name = outputs[1],
                probability = token_probability[1],
            ),
            Token(
                name = outputs[2],
                probability = token_probability[2],
            ),
            Token(
                name = outputs[3],
                probability = token_probability[3],
            ),
            Token(
                name = outputs[4],
                probability = token_probability[4],
            ),
            Token(
                name = outputs[5],
                probability = token_probability[5],
            ),
            Token(
                name = outputs[6],
                probability = token_probability[6],
            ),
            Token(
                name = outputs[7],
                probability = token_probability[7],
            ),
            Token(
                name = outputs[8],
                probability = token_probability[8],
            ),
            Token(
                name = outputs[9],
                probability = token_probability[9],
            ),
        ]
        
        # Save visualization
        mechanism.save_vis(demo_img, increase_scores_normalize, imagePrompt.prompt)
        
        # increase_scores_normalize - min: 0.0, max: 0.1541638498880645
        # For each attention, prefix the patch row and column indices.
        increase_scores_normalize = np.array(increase_scores_normalize)
        increase_scores_normalize = increase_scores_normalize.reshape(24, 24)
    
        attentions_with_locations = transform_matrix_to_3d_points(increase_scores_normalize)
        print(f"Attentions with locations: {attentions_with_locations.shape}")

        # Threshold
        threshold_percentile = 80
        filtered = apply_threshold(attentions_with_locations, threshold_percentile)
        print(f"After threshold: {filtered.shape}")

        # Duplicate
        weighted = duplicate_points(filtered, min_dup=1, max_dup=9)

        # Cluster
        db, _, _ = find_clusters(weighted, eps=1.3, min_samples=15)

        # Entropy
        entropy = calculate_entropy(increase_scores_normalize)
        token_entropy = calculate_token_entropy(token_probability)
        result.entropy= Entropy(
                token = token_entropy,
                general = entropy
            )
        
        print(f"Attention Entropy: {entropy:.4f}")
        print(f"Token Entropy: {token_entropy:.4f}")
        cluster_entropy(db, weighted_attentions_with_locations)
        print(result)

if __name__ == "__main__":
    main()
