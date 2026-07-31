from PIL import Image, ImageOps
from torchvision import transforms
from ..utils.utils import brand_converter, resolution_alignment, l2_norm
from .models2 import KNOWN_MODELS
from ..ocr_lib.models.model_builder import ModelBuilder
from ..ocr_lib.utils.labelmaps import get_vocabulary
import torch
from torch.backends import cudnn
import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from tldextract import tldextract
import pickle

COUNTRY_TLDs = [
    ".af",
    ".ax",
    ".al",
    ".dz",
    ".as",
    ".ad",
    ".ao",
    ".ai",
    ".aq",
    ".ag",
    ".ar",
    ".am",
    ".aw",
    ".ac",
    ".au",
    ".at",
    ".az",
    ".bs",
    ".bh",
    ".bd",
    ".bb",
    ".eus",
    ".by",
    ".be",
    ".bz",
    ".bj",
    ".bm",
    ".bt",
    ".bo",
    ".bq",".an",".nl",
    ".ba",
    ".bw",
    ".bv",
    ".br",
    ".io",
    ".vg",
    ".bn",
    ".bg",
    ".bf",
    ".mm",
    ".bi",
    ".kh",
    ".cm",
    ".ca",
    ".cv",
    ".cat",
    ".ky",
    ".cf",
    ".td",
    ".cl",
    ".cn",
    ".cx",
    ".cc",
    ".co",
    ".km",
    ".cd",
    ".cg",
    ".ck",
    ".cr",
    ".ci",
    ".hr",
    ".cu",
    ".cw",
    ".cy",
    ".cz",
    ".dk",
    ".dj",
    ".dm",
    ".do",
    ".tl",".tp",
    ".ec",
    ".eg",
    ".sv",
    ".gq",
    ".er",
    ".ee",
    ".et",
    ".eu",
    ".fk",
    ".fo",
    ".fm",
    ".fj",
    ".fi",
    ".fr",
    ".gf",
    ".pf",
    ".tf",
    ".ga",
    ".gal",
    ".gm",
    ".ps",
    ".ge",
    ".de",
    ".gh",
    ".gi",
    ".gr",
    ".gl",
    ".gd",
    ".gp",
    ".gu",
    ".gt",
    ".gg",
    ".gn",
    ".gw",
    ".gy",
    ".ht",
    ".hm",
    ".hn",
    ".hk",
    ".hu",
    ".is",
    ".in",
    ".id",
    ".ir",
    ".iq",
    ".ie",
    ".im",
    ".il",
    ".it",
    ".jm",
    ".jp",
    ".je",
    ".jo",
    ".kz",
    ".ke",
    ".ki",
    ".kw",
    ".kg",
    ".la",
    ".lv",
    ".lb",
    ".ls",
    ".lr",
    ".ly",
    ".li",
    ".lt",
    ".lu",
    ".mo",
    ".mk",
    ".mg",
    ".mw",
    ".my",
    ".mv",
    ".ml",
    ".mt",
    ".mh",
    ".mq",
    ".mr",
    ".mu",
    ".yt",
    ".mx",
    ".md",
    ".mc",
    ".mn",
    ".me",
    ".ms",
    ".ma",
    ".mz",
    ".mm",
    ".na",
    ".nr",
    ".np",
    ".nl",
    ".nc",
    ".nz",
    ".ni",
    ".ne",
    ".ng",
    ".nu",
    ".nf",
    ".nc",".tr",
    ".kp",
    ".mp",
    ".no",
    ".om",
    ".pk",
    ".pw",
    ".ps",
    ".pa",
    ".pg",
    ".py",
    ".pe",
    ".ph",
    ".pn",
    ".pl",
    ".pt",
    ".pr",
    ".qa",
    ".ro",
    ".ru",
    ".rw",
    ".re",
    ".bq",".an",
    ".bl",".gp",".fr",
    ".sh",
    ".kn",
    ".lc",
    ".mf",".gp",".fr",
    ".pm",
    ".vc",
    ".ws",
    ".sm",
    ".st",
    ".sa",
    ".sn",
    ".rs",
    ".sc",
    ".sl",
    ".sg",
    ".bq",".an",".nl",
    ".sx",".an",
    ".sk",
    ".si",
    ".sb",
    ".so",
    ".so",
    ".za",
    ".gs",
    ".kr",
    ".ss",
    ".es",
    ".lk",
    ".sd",
    ".sr",
    ".sj",
    ".sz",
    ".se",
    ".ch",
    ".sy",
    ".tw",
    ".tj",
    ".tz",
    ".th",
    ".tg",
    ".tk",
    ".to",
    ".tt",
    ".tn",
    ".tr",
    ".tm",
    ".tc",
    ".tv",
    ".ug",
    ".ua",
    ".ae",
    ".uk",
    ".us",
    ".vi",
    ".uy",
    ".uz",
    ".vu",
    ".va",
    ".ve",
    ".vn",
    ".wf",
    ".eh",
    ".ma",
    ".ye",
    ".zm",
    ".zw"
]

class DataInfo(object):
    """
    Save the info about the dataset.
    This a code snippet from dataset.py
    """
    def __init__(self, voc_type):
        super(DataInfo, self).__init__()
        self.voc_type = voc_type

        assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)

def ocr_model_config(weights_path, height=None, width=None):
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import logging
    logger = logging.getLogger(__name__)
    if device == 'cuda':
        logger.info('Using CUDA for computation')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        logger.info('Using CPU for computation')
        torch.set_default_tensor_type('torch.FloatTensor')

    # Create data loaders
    if height is None or width is None:
        height, width = (32, 100)

    dataset_info = DataInfo('ALLCASES_SYMBOLS')

    # Create model
    model = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=dataset_info.rec_num_classes,
                         sDim=512, attDim=512, max_len_labels=100,
                         eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)

    # Load from checkpoint
    weights_path = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(weights_path['state_dict'])

    if device == 'cuda':
        model = model.to(device)

    return model

def siamese_model_config(num_classes: int, weights_path: str):
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=num_classes, zero_head=True)

    # Load weights
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['model'] if 'model' in weights.keys() else weights
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        if k.startswith('module'):
            name = k.split('module.')[1]
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


def image_process(image_path, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    img = Image.open(image_path).convert('RGB') if isinstance(image_path, str) else image_path.convert('RGB')

    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img


def ocr_main(image_path, model, height=None, width=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Evaluation
    model.eval()

    img = image_process(image_path)
    with torch.no_grad():
        img = img.to(device)
    input_dict = {}
    input_dict['images'] = img.unsqueeze(0)

    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    rec_targets = torch.IntTensor(1, 100).fill_(1)
    rec_targets[:, 100 - 1] = dataset_info.char2id[dataset_info.EOS]
    input_dict['rec_targets'] = rec_targets.to(device)
    input_dict['rec_lengths'] = [100]

    with torch.no_grad():
        features, decoder_feat = model.features(input_dict)
    features = features.detach().cpu()
    decoder_feat = decoder_feat.detach().cpu()
    features = torch.mean(features, dim=1)

    return features

@torch.no_grad()
def get_ocr_aided_siamese_embedding(img, model, ocr_model, grayscale=False):
    '''
    Inference for a single image
    :param img: image path in str or image in PIL.Image
    :param model: Siamese model to make inference
    :param ocr_model: OCR model
    :param imshow: enable display of image or not
    :param title: title of displayed image
    :param grayscale: convert image to grayscale or not
    :return feature embedding of shape (2048,)
    '''
    img_size = 224
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
     ])

    img = Image.open(img) if isinstance(img, str) else img
    img = img.convert("RGBA").convert("L").convert("RGB") if grayscale else img.convert("RGBA").convert("RGB")

    ## Resize the image while keeping the original aspect ratio
    pad_color = 255 if grayscale else (255, 255, 255)
    img = ImageOps.expand(img, (
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=pad_color)

    img = img.resize((img_size, img_size))

    # Predict the embedding
    # get ocr embedding from pretrained paddleOCR
    with torch.no_grad():
        ocr_emb = ocr_main(image_path=img, model=ocr_model, height=None, width=None)
        ocr_emb = ocr_emb[0]
        ocr_emb = ocr_emb[None, ...].to(device)  # remove batch dimension

    # Predict the embedding
    with torch.no_grad():
        img = img_transforms(img)
        img = img[None, ...].to(device)
        logo_feat = model.features(img, ocr_emb)
        logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy()  # L2-normalization final shape is (2560,)

    return logo_feat

def chunked_dot(logo_feat_list, img_feat, chunk_size=128):
    sim_list = []

    for start in range(0, logo_feat_list.shape[0], chunk_size):
        end = start + chunk_size
        chunk = logo_feat_list[start:end]
        sim_chunk = np.dot(chunk, img_feat.T)  # shape: (chunk_size, M)
        sim_list.extend(sim_chunk)

    return sim_list

def pred_brand(
    model,
    ocr_model,
    domain_map,
    logo_feat_list,
    file_name_list,
    shot_path: str,
    pred_bbox,
    t_s,
    grayscale=False,
    do_resolution_alignment=True,
    do_aspect_ratio_check=True,
    ref_pooling: str = "max",
    ref_topm: int = 3,
    return_top_brands: int = 0,
):
    '''
    Return predicted brand for one cropped image
    :param model: model to use
    :param domain_map: brand-domain dictionary
    :param logo_feat_list: reference logo feature embeddings
    :param file_name_list: reference logo paths
    :param shot_path: path to the screenshot
    :param pred_bbox: 1x4 np.ndarray/list/tensor bounding box coords
    :param t_s: similarity threshold for siamese
    :param grayscale: convert image(cropped) to grayscale or not
    :return: predicted target, predicted target's domain
    '''

    try:
        img = Image.open(shot_path)
    except OSError:  # if the image cannot be identified, return nothing
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f'Screenshot cannot be opened: {shot_path}')
        if int(return_top_brands) > 0:
            return None, None, None, []
        return None, None, None

    ## get predicted box --> crop from screenshot
    cropped = img.crop((pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]))
    img_feat = get_ocr_aided_siamese_embedding(cropped, model, ocr_model, grayscale=grayscale)

    sim_list = np.asarray(chunked_dot(logo_feat_list, img_feat), dtype=np.float32)
    pred_brand_list = np.asarray(file_name_list)

    if sim_list.shape[0] != pred_brand_list.shape[0] or sim_list.shape[0] == 0:
        if int(return_top_brands) > 0:
            return None, None, None, []
        return None, None, None

    p = str(ref_pooling or "max").strip().lower()
    topm = int(ref_topm)
    if topm <= 0:
        topm = 1

    brand_to_idx = {}
    for i, fp in enumerate(pred_brand_list.tolist()):
        try:
            parent = os.path.basename(os.path.dirname(str(fp)))
            if str(parent).strip().lower() == "logos":
                parent = os.path.basename(os.path.dirname(os.path.dirname(str(fp))))
            b = brand_converter(str(parent))
        except Exception:
            b = ""
        if not b:
            continue
        brand_to_idx.setdefault(b, []).append(int(i))

    if not brand_to_idx:
        top1 = float(sim_list.max()) if sim_list.size > 0 else 0.0
        if int(return_top_brands) > 0:
            return None, None, top1, []
        return None, None, top1

    best_brand = None
    best_score = None
    best_indices = None
    brand_scores: OrderedDict[str, float] = OrderedDict()
    for b, idxs in brand_to_idx.items():
        v = sim_list[np.asarray(idxs, dtype=np.int64)]
        if v.size == 0:
            continue
        if p == "mean":
            sc = float(v.mean())
        elif p == "topm_mean":
            mm = min(int(topm), int(v.size))
            sc = float(np.sort(v)[::-1][:mm].mean())
        else:
            sc = float(v.max())
        brand_scores[str(b)] = float(sc)
        if best_score is None or float(sc) > float(best_score):
            best_score = float(sc)
            best_brand = str(b)
            best_indices = list(idxs)

    if best_brand is None or best_score is None or not best_indices:
        top1 = float(sim_list.max()) if sim_list.size > 0 else 0.0
        if int(return_top_brands) > 0:
            return None, None, top1, []
        return None, None, top1

    best_indices_sorted = sorted(best_indices, key=lambda ii: float(sim_list[int(ii)]), reverse=True)[:3]
    pooled_sim = float(best_score)

    predicted_domain = domain_map[best_brand]
    enable_alignment = bool(do_resolution_alignment) and (p == "max")

    topn = int(return_top_brands)
    top_brands: List[str] = []
    if topn > 0 and brand_scores:
        top_brands = [
            str(k)
            for k, _ in sorted(brand_scores.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[:topn]
        ]

    if pooled_sim < float(t_s):
        if enable_alignment and best_indices_sorted:
            ii = int(best_indices_sorted[0])
            try:
                cand_path = str(pred_brand_list[ii])
                cand_logo = Image.open(cand_path)
                cropped_aligned, candidate_logo = resolution_alignment(cropped, cand_logo)
                img_feat2 = get_ocr_aided_siamese_embedding(cropped_aligned, model, ocr_model, grayscale=grayscale)
                logo_feat2 = get_ocr_aided_siamese_embedding(candidate_logo, model, ocr_model, grayscale=grayscale)
                final_sim = float(logo_feat2.dot(img_feat2))
            except Exception:
                if topn > 0:
                    return None, None, pooled_sim, top_brands
                return None, None, pooled_sim

            if float(final_sim) < float(t_s):
                if topn > 0:
                    return None, None, pooled_sim, top_brands
                return None, None, pooled_sim

            if bool(do_aspect_ratio_check):
                try:
                    ratio_crop = cropped.size[0] / cropped.size[1]
                    ratio_logo = cand_logo.size[0] / cand_logo.size[1]
                    if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                        return None, None, pooled_sim
                except Exception:
                    pass

            if topn > 0:
                return best_brand, predicted_domain, float(final_sim), top_brands
            return best_brand, predicted_domain, float(final_sim)

        if topn > 0:
            return None, None, pooled_sim, top_brands
        return None, None, pooled_sim

    for ii in best_indices_sorted:
        try:
            cand_path = str(pred_brand_list[int(ii)])
            cand_logo = Image.open(cand_path)
        except Exception:
            continue

        final_sim = pooled_sim

        if bool(do_aspect_ratio_check):
            try:
                ratio_crop = cropped.size[0] / cropped.size[1]
                ratio_logo = cand_logo.size[0] / cand_logo.size[1]
                if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                    continue
            except Exception:
                pass

        if topn > 0:
            return best_brand, predicted_domain, float(final_sim), top_brands
        return best_brand, predicted_domain, float(final_sim)

    if topn > 0:
        return None, None, pooled_sim, top_brands
    return None, None, pooled_sim

def cache_reference_list(model, ocr_model, targetlist_path: str, grayscale=False, reload_targetlist=False):
    '''
    cache the embeddings of the reference list
    '''

    #  Prediction for targetlists
    logo_feat_list = []
    file_name_list = []

    get_target_embedding = reload_targetlist or (not os.path.exists(os.path.join(os.getcwd(), 'LOGO_FEATS.npy')))
    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'):  # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith('.jpg') or logo_path.endswith('.PNG') \
                    or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'):  # skip homepage/loginpage
                    continue
                if get_target_embedding:
                    logo_feat_list.append(get_ocr_aided_siamese_embedding(img=os.path.join(targetlist_path, target, logo_path),
                                                                        model=model,
                                                                        ocr_model=ocr_model,
                                                                        grayscale=grayscale))
                file_name_list.append(str(os.path.join(targetlist_path, target, logo_path)))
                
    if get_target_embedding:
        np.save(os.path.join(os.getcwd(),'LOGO_FEATS.npy'), np.asarray(logo_feat_list))
        import logging
        logger = logging.getLogger(__name__)
        logger.info('Saved logo features to LOGO_FEATS.npy')
    else:
        logo_feat_list = np.load(os.path.join(os.getcwd(),'LOGO_FEATS.npy'))

    return np.asarray(logo_feat_list), np.asarray(file_name_list)

def check_domain_brand_inconsistency(logo_boxes,
                                     domain_map_path: str,
                                     model,
                                     ocr_model,
                                     logo_feat_list,
                                     file_name_list,
                                     shot_path: str,
                                     url: str,
                                     ts: float):

    # targetlist domain list
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    # look at boxes for logo class only
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f'Number of logo boxes detected: {len(logo_boxes)}')
    suffix_part = '.'+ tldextract.extract(url).suffix
    domain_part = tldextract.extract(url).domain
    extracted_domain = domain_part + suffix_part
                                         
    matched_target, matched_domain, matched_coord, this_conf = None, None, None, None


    # run logo matcher
    if len(logo_boxes) > 0:
        # siamese prediction for logo box
        for i, coord in enumerate(logo_boxes):
            min_x, min_y, max_x, max_y = coord
            bbox = [float(min_x), float(min_y), float(max_x), float(max_y)]
            matched_target, matched_domain, this_conf = pred_brand(model, ocr_model, domain_map,
                                                                  logo_feat_list, file_name_list,
                                                                  shot_path, bbox, t_s=ts, grayscale=False)

            # domain matcher to avoid FP
            # if matched_target is not None:
            #     matched_coord = coord
            #     # if tldextract.extract(url).domain+ '.'+tldextract.extract(url).suffix not in matched_domain:
            #     if tldextract.extract(url).domain not in matched_domain:
            #         # avoid fp due to godaddy domain parking, ignore webmail provider (ambiguous)
            #         if matched_target == 'GoDaddy' or matched_target == "Webmail Provider" or matched_target == "Government of the United Kingdom":
            #             matched_target = None  # ignore the prediction
            #             matched_domain = None  # ignore the prediction
            #     else:  # benign, real target
            #         matched_target = None  # ignore the prediction
            #         matched_domain = None  # ignore the prediction
            #     break  # break if target is matched
            # break  # only look at 1st logo
            if (matched_target is not None) and (matched_domain is not None):
                matched_coord = coord
                matched_domain_parts = [tldextract.extract(x).domain for x in matched_domain]
                
                # If the webpage domain exactly aligns with the target website's domain => Benign
                if extracted_domain in matched_domain:
                    matched_target, matched_domain = None, None  # Clear if domains are consistent
                elif domain_part in matched_domain_parts: # # elIf only the 2nd-level-domains align, and the tld is regional  => Benign
                    if "." + suffix_part.split('.')[-1] in COUNTRY_TLDs:
                        matched_target, matched_domain = None, None
                    else:
                        break # Inconsistent domain found, break the loop
                else:
                    break  # Inconsistent domain found, break the loop
            break  # only look at 1st logo

    return brand_converter(matched_target), matched_domain, matched_coord, this_conf



