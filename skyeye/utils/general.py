"""
General utility functions for SkyEye object detection framework
"""

import os
import re
import glob
import logging
import platform
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pkg_resources as pkg
import yaml
import torch


def set_logging(name=None, verbose=True):
    """
    Set up logging for SkyEye modules
    
    Args:
        name (str): Name for the logger
        verbose (bool): Enable verbose output
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Configure logging format and level
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARNING
    )
    return logging.getLogger(name)


# Initialize logging
LOGGER = set_logging(__name__)


def colorstr(*inputs):
    """
    Colorize strings for terminal output
    
    Args:
        *inputs: String inputs with optional color arguments
        
    Returns:
        str: Colorized string
    """
    # Colors a string using ANSI escape sequences
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def check_online():
    """
    Check if internet connection is available
    
    Returns:
        bool: True if internet is accessible, False otherwise
    """
    try:
        # Try connecting to a reliable server
        urllib.request.urlopen('https://github.com', timeout=5)
        return True
    except (urllib.request.URLError, ValueError):
        return False


def check_file(file_path):
    """
    Check if a file exists and return its path, download if it's a URL
    
    Args:
        file_path (str): File path or URL
        
    Returns:
        Path: Path object to the file
    """
    file_path = str(file_path)
    file = Path(file_path)
    
    if file.exists():
        return file
    elif file_path.startswith(('http:/', 'https:/')):
        # Fix URL format and extract filename
        url = str(file_path).replace(':/', '://')
        file = Path(urllib.parse.unquote(url).split('?')[0]).name
        
        if Path(file).exists():
            LOGGER.info(f'Found {url} locally at {file}')
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            try:
                torch.hub.download_url_to_file(url, file)
                assert Path(file).exists() and Path(file).stat().st_size > 0
            except Exception as e:
                LOGGER.error(f'Download failed: {e}')
                return None
        return file
    else:
        # Look in common directories
        for d in ['data', 'models', 'utils']:
            files = glob.glob(str(Path(d) / '**' / file_path), recursive=True)
            if files:
                assert len(files) == 1, f"Multiple files match '{file_path}': {files}"
                return Path(files[0])
        
        LOGGER.error(f"File not found: {file_path}")
        return None


def check_yaml(file):
    """
    Check and return a YAML file
    
    Args:
        file (str): YAML file path or URL
        
    Returns:
        Path: Path object to the YAML file
    """
    return check_file(file) if file.endswith(('.yaml', '.yml')) else file


def check_version(current='0.0.0', minimum='0.0.0', name='version ', strict=False):
    """
    Check version compatibility
    
    Args:
        current (str): Current version
        minimum (str): Minimum required version
        name (str): Name of the package
        strict (bool): If True, raises AssertionError when version check fails
        
    Returns:
        bool: True if version requirements are met
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current >= minimum
    
    if strict and not result:
        error_msg = f'{name}{minimum} required, but {name}{current} is installed'
        assert result, error_msg
        
    return result


def check_requirements(requirements=None, exclude=(), install=True):
    """
    Check if required packages are installed and attempt to install missing ones
    
    Args:
        requirements (str or list): Requirements file path or list of packages
        exclude (tuple): Packages to exclude from checking
        install (bool): Attempt to install missing packages
        
    Returns:
        bool: True if all requirements are met
    """
    if requirements is None:
        return True
        
    prefix = colorstr('red', 'bold', 'requirements:')
    
    # Convert file path to package list
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        if not file.exists():
            LOGGER.error(f"{prefix} {file} not found")
            return False
            
        requirements = [
            f'{x.name}{x.specifier}' 
            for x in pkg.parse_requirements(file.open()) 
            if x.name not in exclude
        ]
    else:
        requirements = [x for x in requirements if x not in exclude]
    
    # Check each requirement
    n = 0  # Number of packages updated
    for requirement in requirements:
        try:
            pkg.require(requirement)
        except (pkg.VersionConflict, pkg.DistributionNotFound):
            if install and check_online():
                LOGGER.info(f"{prefix} {requirement} not found, attempting installation...")
                try:
                    subprocess.check_call(
                        ['pip', 'install', requirement],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT
                    )
                    n += 1
                except Exception as e:
                    LOGGER.error(f"{prefix} {e}")
            else:
                LOGGER.warning(f"{prefix} {requirement} not found")
    
    if n > 0:
        source = file if 'file' in locals() else requirements
        LOGGER.info(f"{prefix} {n} package(s) updated per {source}")
    
    return True


def make_divisible(x, divisor=8):
    """
    Ensure x is divisible by divisor
    
    Args:
        x (int or float): Input value
        divisor (int): Divisor
        
    Returns:
        int: Value rounded up to be divisible by divisor
    """
    return int(np.ceil(x / divisor) * divisor)


def check_img_size(img_size, stride=32):
    """
    Make sure image size is a multiple of stride
    
    Args:
        img_size (int or list): Image size
        stride (int): Stride
        
    Returns:
        list: New image size
    """
    # Verify image size is a multiple of stride s
    if isinstance(img_size, int):
        new_size = make_divisible(img_size, int(stride))
    else:
        new_size = [make_divisible(x, int(stride)) for x in img_size]
        
    if new_size != img_size:
        LOGGER.warning(f'Image size {img_size} adjusted to {new_size} for model stride compatibility')
        
    return new_size


def is_ascii(s=''):
    """
    Check if string is ASCII
    
    Args:
        s (str): String to check
        
    Returns:
        bool: True if string is ASCII
    """
    # Test if string is composed of ASCII characters only
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    """
    Check if string contains Chinese characters
    
    Args:
        s (str): String to check
        
    Returns:
        bool: True if string contains Chinese characters
    """
    return bool(re.search('[\u4e00-\u9fff]', str(s)))


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increment file or directory path, i.e., runs/exp --> runs/exp{sep}0, runs/exp{sep}1, etc.
    
    Args:
        path (str or Path): Path to increment
        exist_ok (bool): If True, don't increment for existing paths
        sep (str): Separator to use before increment number
        mkdir (bool): Create directory if it doesn't exist
        
    Returns:
        Path: Incremented path
    """
    path = Path(path)
    
    if path.exists() and not exist_ok:
        # Look for existing paths with same prefix but higher suffix
        suffix = path.suffix
        path = path.with_suffix('') if suffix else path
        
        # Get matching directories with numeric suffixes
        matches = list(path.parent.glob(f"{path.name}{sep}*"))
        matches = [re.search(rf"%s{sep}(\d+)" % path.name, str(m)) for m in matches]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        
        path = Path(f"{path}{sep}{n}{suffix}")
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
        
    return path
