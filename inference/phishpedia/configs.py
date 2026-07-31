# Global configuration
import yaml
from .logo_matching import cache_reference_list, load_model_weights
from .logo_recog import config_rcnn
import os
import numpy as np

# --- VLBrandPhish repo integration: anchor model weights at <repo>/models/phishpedia/ ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # inference/phishpedia -> inference -> repo root
_MODELS_DIR = os.path.join(_REPO_ROOT, "models", "phishpedia")


def get_absolute_path(relative_path):
    """Resolve a config path: models/... -> repo models dir; configs/... -> package dir."""
    if os.path.isabs(relative_path):
        return relative_path
    if relative_path.startswith("models/"):
        return os.path.join(_MODELS_DIR, os.path.basename(relative_path))
    return os.path.abspath(os.path.join(_THIS_DIR, relative_path))


def load_config(reload_targetlist=False):
    with open(os.path.join(os.path.dirname(__file__), 'configs/configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    # Iterate through the configuration and update paths
    for section, settings in configs.items():
        for key, value in settings.items():
            if 'PATH' in key and isinstance(value, str):  # Check if the key indicates a path
                absolute_path = get_absolute_path(value)
                configs[section][key] = absolute_path

    ELE_CFG_PATH = configs['ELE_MODEL']['CFG_PATH']
    ELE_WEIGHTS_PATH = configs['ELE_MODEL']['WEIGHTS_PATH']
    ELE_CONFIG_THRE = configs['ELE_MODEL']['DETECT_THRE']
    ELE_MODEL = config_rcnn(ELE_CFG_PATH,
                            ELE_WEIGHTS_PATH,
                            conf_threshold=ELE_CONFIG_THRE)

    # siamese model
    SIAMESE_THRE = configs['SIAMESE_MODEL']['MATCH_THRE']

    print('Load protected logo list')
    targetlist_zip_path = configs['SIAMESE_MODEL']['TARGETLIST_PATH']
    targetlist_dir = os.path.dirname(targetlist_zip_path)
    zip_file_name = os.path.basename(targetlist_zip_path)
    targetlist_folder = zip_file_name.split('.zip')[0]
    full_targetlist_folder_dir = os.path.join(targetlist_dir, targetlist_folder)

    # if reload_targetlist or targetlist_zip_path.endswith('.zip') and not os.path.isdir(full_targetlist_folder_dir):
    #     os.makedirs(full_targetlist_folder_dir, exist_ok=True)
    #     subprocess.run(f'unzip -o "{targetlist_zip_path}" -d "{full_targetlist_folder_dir}"', shell=True)

    SIAMESE_MODEL = load_model_weights(num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
                                       weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'])

    # Logo feature cache lives next to the weights (regenerable, gitignored).
    LOGO_FEATS_NAME = 'LOGO_FEATS.npy'
    LOGO_FILES_NAME = 'LOGO_FILES.npy'
    _feats_path = os.path.join(_MODELS_DIR, LOGO_FEATS_NAME)
    _files_path = os.path.join(_MODELS_DIR, LOGO_FILES_NAME)

    if reload_targetlist or (not os.path.exists(_feats_path)):
        LOGO_FEATS, LOGO_FILES = cache_reference_list(model=SIAMESE_MODEL,
                                                      targetlist_path=full_targetlist_folder_dir)
        print('Finish loading protected logo list')
        os.makedirs(_MODELS_DIR, exist_ok=True)
        np.save(_feats_path, LOGO_FEATS)
        np.save(_files_path, LOGO_FILES)

    else:
        LOGO_FEATS, LOGO_FILES = np.load(_feats_path), np.load(_files_path)

    DOMAIN_MAP_PATH = configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH']

    return ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH
