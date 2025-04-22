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
        if not
