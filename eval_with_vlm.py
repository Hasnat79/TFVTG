import os
from tqdm import tqdm
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from vlm_localizer import localize
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import random
from functools import lru_cache
import time 
device = 'cuda:1'
@lru_cache(maxsize=None)
def load_model_and_processors():
  model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
  vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
  ])
  return model, vis_processors, text_processors

model, vis_processors, text_processors = load_model_and_processors()
from decord import VideoReader, cpu
from functools import lru_cache

def loadvideo(fname, fps=3, stride=None, max_duration=None):
    print("=============inside load video ===============")
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    print(f"Video loaded: {fname}")
    duration = len(vr) / vr.get_avg_fps()
    print(f"durations: {duration}")
    if fps is not None:
        print(f"fps: {fps}")
        num_sampled_frames = round(duration * fps)
        print(f"num_sampled_frames: {num_sampled_frames}")
        all_index = np.linspace(0, len(vr)-1, num=num_sampled_frames).round().astype(np.int32)
        print(f"length of all_index: {len(all_index)}")
        if max_duration is not None:
            all_index = all_index[:round(max_duration * fps)]
    else:
        assert stride is not None
        all_index = np.arange(0, len(vr), stride, dtype=np.int32)
        if max_duration is not None:
            all_index = all_index[:round(max_duration * all_index.shape[0] / duration)]
    vr.seek(0)
    print(f"vr.get_batch(all_index).shape: {vr.get_batch(all_index).shape}")
    buffer = vr.get_batch(all_index).permute(0, 3, 1, 2) / 255.
    print(f"buffer shape: {buffer.shape}")
    print("=============inside load video ===============")
    # [ sampled frames, channel, h,w] : [11, 3, 720, 1280]
    return buffer


@torch.no_grad()
def get_visual_features(video_path, fps=None, stride=None, max_duration=None, batch_size=128):
    print("=============inside get_visual_features ===============")
    video = loadvideo(video_path, fps, stride, max_duration)
    print(f"video/image list shape before vis_processor: {video.shape}") #[11, 3, 720, 1280]
    img = vis_processors(video)
    print(f"image shape after going through vis_processors: {img.shape}") #[11, 3, 364, 364]
    features = []
    print(f"img.size(0): {img.size(0)}")
    for bid in range(0, img.size(0), batch_size):
        batch_img = img[bid:bid+batch_size].to(device)
        print(f"\tbatch_img shape: {batch_img.shape}")#[11, 3, 364, 364]
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(batch_img)) 
        image_embeds = image_embeds.float()
        print(f"\timage_embeds shape after model.lnvision: {image_embeds.shape}")#[11, 677, 1408]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        print(f"\timage_atts shape: {image_atts.shape}")#[11, 677]
        print(f"\tshapes of model.query_tokens: {model.query_tokens.shape}") #[1, 32, 768]
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        print(f"\tquery_tokens shape: {query_tokens.shape}") #[11, 32, 768]
        query_output = model.Qformer.bert(
            query_embeds=query_tokens, #[11, 32, 768]
            encoder_hidden_states=image_embeds, #[11, 677, 1408]
            encoder_attention_mask=image_atts, #[11, 677]
            return_dict=True,
        )

        print(f"\tquery_output.last_hidden_state shape: {query_output.last_hidden_state.shape}")#[11, 32, 768]
        image_feats = model.vision_proj(query_output.last_hidden_state)
        print(f"\timage_feats shape after vision projection: {image_feats.shape}") #[11, 32, 256]
        features.append(image_feats.cpu().half())
    features = torch.cat(features, dim=0)

    return features.numpy()

def eval(data, feature, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        query_json = []
        for i in range(len(ann['sentences'])):
            query_json.append({'descriptions': [ann['sentences'][i]]})

        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        # video_feature_path = os.path.join(f, vid+'.npy')
        # video_feature = np.load(video_feature_path)
        video_feature = feature
        if pad_sec > 0:
            pad_noise = np.random.randn(round(video_feature.shape[0] / duration * pad_sec), video_feature.shape[1], video_feature.shape[2])
            video_feature = np.concatenate([pad_noise, video_feature], axis=0)
            duration += pad_sec

        ans = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
        for i in range(len(ans)):
            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = ans[i]['response'][0]['start'], ans[i]['response'][0]['end']
            iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))



if __name__=='__main__':


  
  data = {
      "vid": {
        "duration": 3.64,
        "timestamps": [[2.714329, 3.75]],
        "sentences": ["A guy jumps onto a bed where his son is. When the guy jumps, the son flies up and hits the wall."]
      }
  }    
  video_path = "//home/grads/h/hasnat.md.abdullah/TFVTG/sample_videos/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"

  save_path = "/home/grads/h/hasnat.md.abdullah/TFVTG/sample_videos/vid.npy"

  fps = 3
  stride = None
  batch_size = 256
  if not os.path.exists(save_path):
    start_time = time.time()
    feature  = get_visual_features(video_path, fps=fps, stride= stride,batch_size=batch_size)
    print(f"Time taken to extract features: {time.time()-start_time}")
    np.save(save_path, feature)
  
  feature = np.load(save_path)

  eval (data,feature,stride =20, max_stride_factor= 0.5, pad_sec=0.0)
  print(feature.shape)
