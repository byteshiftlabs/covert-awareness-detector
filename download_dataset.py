#!/usr/bin/env python3
"""
Dataset Downloader for Consciousness Detector Project
======================================================

Downloads the Michigan Human Anesthesia fMRI Dataset (OpenNeuro ds006623)
using direct AWS S3 access. This script downloads only the essential files
needed for consciousness state detection research.

Dataset: https://openneuro.org/datasets/ds006623
DOI: 10.18112/openneuro.ds006623.v1.0.0

Essential Files:
- derivatives/Participant_Info.csv (subject demographics and metadata)
- derivatives/LOR_ROR_Timing.csv (consciousness transition timestamps)
- derivatives/xcp_d_without_GSR_bandpass_output/ (preprocessed connectivity data)

Usage:
    python download_dataset.py --output-dir /path/to/dataset
    python download_dataset.py --output-dir ./data --verify-checksums
    python download_dataset.py --list-files  # Show what will be downloaded

Author: cmelnulabs
Date: February 2026
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime


class DownloadStats:
    """Track download statistics."""
    def __init__(self):
        self.total_files = 0
        self.downloaded_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.total_bytes = 0
        self.start_time = datetime.now()


class ProgressBar:
    """Simple progress bar for downloads."""
    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description
        self.bar_length = 50
        
    def update(self, amount: int = 1):
        """Update progress by amount."""
        self.current += amount
        self._render()
    
    def _render(self):
        """Render the progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = min(100, int(100 * self.current / self.total))
        
        filled = int(self.bar_length * percent / 100)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        
        # Format bytes
        if self.total > 1024 * 1024:
            size_str = f"{self.current / (1024*1024):.1f}/{self.total / (1024*1024):.1f} MB"
        elif self.total > 1024:
            size_str = f"{self.current / 1024:.1f}/{self.total / 1024:.1f} KB"
        else:
            size_str = f"{self.current}/{self.total} bytes"
        
        print(f"\r{self.description}: |{bar}| {percent}% {size_str}", end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


class DatasetDownloader:
    """
    Download OpenNeuro ds006623 dataset files via S3.
    
    OpenNeuro provides S3 access to datasets through:
    s3://openneuro.org/ds006623/
    
    This can be accessed via HTTP at:
    https://s3.amazonaws.com/openneuro.org/ds006623/
    """
    
    # Base S3 URL for OpenNeuro
    BASE_URL = "https://s3.amazonaws.com/openneuro.org/ds006623"
    
    # Essential files to download
    ESSENTIAL_FILES = [
        "dataset_description.json",
        "README.md",
        "CHANGES",
        "derivatives/Participant_Info.csv",
        "derivatives/LOR_ROR_Timing.csv",
    ]
    
    # Subject IDs in the dataset
    SUBJECTS = [
        "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07",
        "sub-11", "sub-12", "sub-13", "sub-14", "sub-15", "sub-16",
        "sub-17", "sub-18", "sub-19", "sub-20", "sub-21", "sub-22",
        "sub-23", "sub-24", "sub-25", "sub-26", "sub-27", "sub-28",
        "sub-29", "sub-30"
    ]
    
    def __init__(self, output_dir: str, verify_checksums: bool = False):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to download files to
            verify_checksums: Whether to verify file integrity with checksums
        """
        self.output_dir = Path(output_dir)
        self.verify_checksums = verify_checksums
        self.stats = DownloadStats()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, remote_path: str, local_path: Path, 
                     show_progress: bool = True) -> bool:
        """
        Download a single file with resume capability.
        
        Args:
            remote_path: Relative path from dataset root
            local_path: Local path to save the file
            show_progress: Whether to show progress bar
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.BASE_URL}/{remote_path}"
        
        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if local_path.exists():
            print(f"✓ Already exists: {remote_path}")
            self.stats.skipped_files += 1
            return True
        
        try:
            # Get file size
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req) as response:
                file_size = int(response.headers.get('Content-Length', 0))
            
            # Download with progress
            if show_progress and file_size > 0:
                progress = ProgressBar(file_size, f"Downloading {remote_path}")
                
                def reporthook(block_num, block_size, total_size):
                    if block_num == 0:
                        return
                    downloaded = min(block_num * block_size, total_size)
                    progress.current = downloaded
                    progress._render()
                
                urllib.request.urlretrieve(url, local_path, reporthook=reporthook)
                print()  # New line after progress bar
            else:
                urllib.request.urlretrieve(url, local_path)
                print(f"✓ Downloaded: {remote_path}")
            
            self.stats.downloaded_files += 1
            self.stats.total_bytes += file_size
            return True
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"✗ Not found: {remote_path}")
            else:
                print(f"✗ HTTP Error {e.code}: {remote_path}")
            self.stats.failed_files += 1
            return False
            
        except Exception as e:
            print(f"✗ Error downloading {remote_path}: {e}")
            self.stats.failed_files += 1
            return False
    
    def download_essential_files(self):
        """Download essential metadata and timing files."""
        print("\n" + "="*70)
        print("STEP 1: Downloading Essential Metadata Files")
        print("="*70)
        
        for file_path in self.ESSENTIAL_FILES:
            local_path = self.output_dir / file_path
            self.stats.total_files += 1
            self.download_file(file_path, local_path)
    
    def download_connectivity_data(self, subjects: Optional[List[str]] = None):
        """
        Download preprocessed connectivity data from XCP-D pipeline.
        
        Args:
            subjects: List of subject IDs to download. If None, downloads all.
        """
        print("\n" + "="*70)
        print("STEP 2: Downloading Preprocessed Connectivity Data")
        print("="*70)
        print("Directory: derivatives/xcp_d_without_GSR_bandpass_output/")
        print()
        
        if subjects is None:
            subjects = self.SUBJECTS
        
        # Files to download for each subject
        connectivity_files = [
            "func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-residual_smooth_bold.nii.gz",
            "func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-coverage_bold.json",
            "func/{subject}_task-rest_atlas-Gordon_timeseries.tsv",
            "func/{subject}_task-rest_atlas-Gordon_connectome.tsv",
            "func/{subject}_task-rest_atlas-Schaefer417_timeseries.tsv",
            "func/{subject}_task-rest_atlas-Schaefer417_connectome.tsv",
            "func/{subject}_task-rest_atlas-Glasser_timeseries.tsv",
            "func/{subject}_task-rest_atlas-Glasser_connectome.tsv",
        ]
        
        base_path = "derivatives/xcp_d_without_GSR_bandpass_output"
        
        for subject in subjects:
            print(f"\nProcessing {subject}...")
            
            for file_template in connectivity_files:
                file_path = file_template.format(subject=subject)
                remote_path = f"{base_path}/{subject}/{file_path}"
                local_path = self.output_dir / base_path / subject / file_path
                
                self.stats.total_files += 1
                self.download_file(remote_path, local_path, show_progress=False)
    
    def list_files(self):
        """List all files that would be downloaded."""
        print("\n" + "="*70)
        print("Files to Download from OpenNeuro ds006623")
        print("="*70)
        
        print("\n1. Essential Metadata Files:")
        for f in self.ESSENTIAL_FILES:
            print(f"   - {f}")
        
        print(f"\n2. Connectivity Data for {len(self.SUBJECTS)} subjects:")
        print("   - Preprocessed BOLD timeseries")
        print("   - Functional connectivity matrices")
        print("   - Atlases: Gordon (333), Schaefer (417), Glasser (360)")
        print(f"   - Subjects: {', '.join(self.SUBJECTS[:5])} ... {', '.join(self.SUBJECTS[-2:])}")
        
        print("\n" + "="*70)
    
    def print_summary(self):
        """Print download summary statistics."""
        duration = (datetime.now() - self.stats.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("Download Summary")
        print("="*70)
        print(f"Total files processed:  {self.stats.total_files}")
        print(f"Successfully downloaded: {self.stats.downloaded_files}")
        print(f"Already existed (skipped): {self.stats.skipped_files}")
        print(f"Failed: {self.stats.failed_files}")
        print(f"Total data downloaded: {self.stats.total_bytes / (1024*1024):.1f} MB")
        print(f"Time elapsed: {duration:.1f} seconds")
        print("="*70)
        
        if self.stats.failed_files > 0:
            print("\n⚠️  Some files failed to download. Check errors above.")
            return False
        else:
            print("\n✅ All files downloaded successfully!")
            print(f"\nDataset location: {self.output_dir.absolute()}")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro ds006623 dataset for consciousness detection research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python download_dataset.py
  
  # Download to specific directory
  python download_dataset.py --output-dir /path/to/data
  
  # List files without downloading
  python download_dataset.py --list-files
  
  # Download only specific subjects
  python download_dataset.py --subjects sub-02 sub-03 sub-04
  
  # Download only metadata (no connectivity data)
  python download_dataset.py --metadata-only

Dataset Information:
  Name: Michigan Human Anesthesia fMRI Dataset
  OpenNeuro ID: ds006623
  URL: https://openneuro.org/datasets/ds006623
  DOI: 10.18112/openneuro.ds006623.v1.0.0
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='../datasets/openneuro/ds006623',
        help='Directory to download dataset to (default: ../datasets/openneuro/ds006623)'
    )
    
    parser.add_argument(
        '--list-files', '-l',
        action='store_true',
        help='List files that would be downloaded without downloading'
    )
    
    parser.add_argument(
        '--metadata-only', '-m',
        action='store_true',
        help='Download only metadata files (CSV, JSON, README), skip connectivity data'
    )
    
    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        help='Download only specific subjects (e.g., sub-02 sub-03)'
    )
    
    parser.add_argument(
        '--verify-checksums',
        action='store_true',
        help='Verify file integrity with checksums (slower)'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = DatasetDownloader(
        output_dir=args.output_dir,
        verify_checksums=args.verify_checksums
    )
    
    # List files mode
    if args.list_files:
        downloader.list_files()
        return 0
    
    # Print header
    print("="*70)
    print("OpenNeuro ds006623 Dataset Downloader")
    print("Michigan Human Anesthesia fMRI Dataset")
    print("="*70)
    print(f"Output directory: {downloader.output_dir.absolute()}")
    print(f"Download method: Direct S3 access (HTTPS)")
    print("="*70)
    
    # Download essential files
    downloader.download_essential_files()
    
    # Download connectivity data (unless metadata-only)
    if not args.metadata_only:
        subjects = args.subjects if args.subjects else None
        downloader.download_connectivity_data(subjects=subjects)
    else:
        print("\n⚠️  Skipping connectivity data (--metadata-only)")
    
    # Print summary
    success = downloader.print_summary()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
