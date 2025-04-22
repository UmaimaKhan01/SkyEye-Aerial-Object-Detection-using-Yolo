"""
Download utilities for SkyEye models and datasets
"""

import os
import subprocess
import time
import urllib.parse
from pathlib import Path

import requests
import torch

from .general import LOGGER, check_online, is_ascii


def safe_download(file, url, url2=None, min_bytes=1e5, error_msg=''):
    """
    Safe downloading function with multiple attempts
    
    Args:
        file (str): File name/path to save
        url (str): Primary URL to download
        url2 (str, optional): Backup URL
        min_bytes (float): Minimum size of file in bytes
        error_msg (str): Error message if download fails
        
    Returns:
        Path: Path to downloaded file
    """
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or is too small"
    
    try:  # First attempt
        LOGGER.info(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file), progress=True)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:
        # Try backup URL or same URL again
        file.unlink(missing_ok=True)  # remove partial downloads
        LOGGER.info(f'ERROR: {e}\nRetrying {url2 or url}...')
        
        # Second attempt
        try:
            # Download using curl or wget
            if is_ascii(str(url2 or url)):
                subprocess.run(['curl', '-L', url2 or url, '-o', str(file), '--retry', '3', '-C', '-'], check=True)
            else:
                # Use requests for non-ASCII URLs
                with requests.get(url2 or url, stream=True) as r:
                    with open(file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
        except Exception as e:
            LOGGER.info(f'ERROR: {e}')
    
    # Check file
    if not file.exists() or file.stat().st_size < min_bytes:
        file.unlink(missing_ok=True)  # remove partial downloads
        LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
    else:
        LOGGER.info(f"Download successful: {file}")
    
    return file


def attempt_download(file, repo='skyeye-ai/skyeye'):
    """
    Attempt to download a file and handle file paths and URLs
    
    Args:
        file (str): File name or URL
        repo (str): GitHub repository for GitHub release assets
        
    Returns:
        str: Path to downloaded file
    """
    file = Path(str(file).strip().replace("'", ''))
    
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # filename
        
        # Direct URL
        if str(file).startswith(('http:/', 'https:/')):
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            name = name.split('?')[0]  # Remove auth
            file = name
            
            if Path(name).exists():
                LOGGER.info(f"Found existing {name}")
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file
        
        # GitHub release assets
        file.parent.mkdir(parents=True, exist_ok=True)  # Create parent directory
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()
            assets = [x['name'] for x in response['assets']]  # release assets
            tag = response['tag_name']  # e.g. 'v1.0'
        except Exception:
            # Fallback to hardcoded assets if GitHub API fails
            assets = [f'skyeye_{x}.pt' for x in ['s', 'm', 'l']]
            tag = 'v0.1'
        
        if name in assets:
            file_url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
            safe_download(file=file, url=file_url, min_bytes=1e5,
                         error_msg=f'Download failed: {file_url}')
    
    return str(file)


def download_weights(model_type='s', dest_path=None):
    """
    Download model weights based on model type
    
    Args:
        model_type (str): Model type ('s', 'm', 'l')
        dest_path (str, optional): Destination path
        
    Returns:
        str: Path to downloaded weights
    """
    if dest_path is None:
        dest_path = Path('weights')
    
    dest_path = Path(dest_path)
    dest_path.mkdir(exist_ok=True, parents=True)
    
    available_models = {
        's': 'skyeye_s.pt',
        'm': 'skyeye_m.pt',
        'l': 'skyeye_l.pt'
    }
    
    if model_type not in available_models:
        LOGGER.error(f"Invalid model type '{model_type}', available options: {list(available_models.keys())}")
        return None
    
    model_name = available_models[model_type]
    model_path = dest_path / model_name
    
    if not model_path.exists():
        LOGGER.info(f"Downloading {model_name} to {model_path}")
        
        # Ensure internet connectivity
        if not check_online():
            LOGGER.error("No internet connection available for downloading weights")
            return None
        
        # Attempt to download model weights
        file_url = f'https://github.com/skyeye-ai/skyeye/releases/download/v0.1/{model_name}'
        safe_download(file=model_path, url=file_url, min_bytes=1e6,
                     error_msg=f"Failed to download {model_name}")
    else:
        LOGGER.info(f"Using existing {model_name} from {model_path}")
    
    return str(model_path)
