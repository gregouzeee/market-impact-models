"""
Utility functions for S3 storage (compatible with SSP Cloud MinIO).

This module provides functions to upload/download data to/from S3-compatible
storage, making it easy to share data across different environments.

Usage:
- On SSP Cloud: S3 is used automatically (credentials from environment)
- Locally: Falls back to local file storage (no S3 required)
"""

import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables (for local .env file)
load_dotenv()


def is_ssp_cloud():
    """
    Detect if running on SSP Cloud.

    SSP Cloud sets specific environment variables like:
    - AWS_S3_ENDPOINT
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_SESSION_TOKEN (optional)
    """
    # Check for SSP Cloud specific variables
    return bool(os.getenv('AWS_S3_ENDPOINT') or os.getenv('S3_ENDPOINT_URL'))


class S3Manager:
    """Manager for S3 operations compatible with SSP Cloud MinIO."""

    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        # SSP Cloud uses AWS_S3_ENDPOINT, fallback to S3_ENDPOINT_URL for compatibility
        self.endpoint_url = os.getenv('AWS_S3_ENDPOINT') or os.getenv('S3_ENDPOINT_URL')
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.session_token = os.getenv('AWS_SESSION_TOKEN')  # SSP Cloud may use session tokens
        self.bucket_name = os.getenv('S3_BUCKET_NAME') or os.getenv('AWS_BUCKET_NAME')
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            raise ValueError(
                "Missing S3 credentials. Please set the following environment variables:\n"
                "  - S3_ENDPOINT_URL\n"
                "  - AWS_ACCESS_KEY_ID\n"
                "  - AWS_SECRET_ACCESS_KEY\n"
                "  - S3_BUCKET_NAME\n"
                "See .env.example for details."
            )

        # Initialize S3 client with optional session token
        client_kwargs = {
            'endpoint_url': self.endpoint_url,
            'aws_access_key_id': self.access_key,
            'aws_secret_access_key': self.secret_key,
            'region_name': self.region
        }

        if self.session_token:
            client_kwargs['aws_session_token'] = self.session_token

        self.s3 = boto3.client('s3', **client_kwargs)

        print(f"✅ S3 client initialized (bucket: {self.bucket_name})")

    def upload_file(self, local_path, s3_key=None):
        """
        Upload a file to S3.

        Parameters:
        -----------
        local_path : str or Path
            Path to local file
        s3_key : str, optional
            S3 object key (path in bucket). If None, uses local filename

        Returns:
        --------
        str : S3 key of uploaded file
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if s3_key is None:
            s3_key = local_path.name

        try:
            self.s3.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key
            )
            print(f"✅ Uploaded {local_path} → s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise

    def download_file(self, s3_key, local_path=None):
        """
        Download a file from S3.

        Parameters:
        -----------
        s3_key : str
            S3 object key (path in bucket)
        local_path : str or Path, optional
            Path to save file locally. If None, uses s3_key as filename

        Returns:
        --------
        Path : Path to downloaded file
        """
        if local_path is None:
            local_path = Path(s3_key)
        else:
            local_path = Path(local_path)

        # Create parent directory if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            print(f"✅ Downloaded s3://{self.bucket_name}/{s3_key} → {local_path}")
            return local_path

        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

    def file_exists(self, s3_key):
        """
        Check if a file exists in S3.

        Parameters:
        -----------
        s3_key : str
            S3 object key to check

        Returns:
        --------
        bool : True if file exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False

    def list_files(self, prefix=''):
        """
        List files in S3 bucket with optional prefix.

        Parameters:
        -----------
        prefix : str, optional
            Filter files by prefix (folder path)

        Returns:
        --------
        list of str : S3 keys matching the prefix
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                return []

            files = [obj['Key'] for obj in response['Contents']]
            return files

        except Exception as e:
            print(f"❌ List failed: {e}")
            raise

    def upload_json(self, data, s3_key):
        """
        Upload JSON data directly to S3.

        Parameters:
        -----------
        data : dict or list
            Data to serialize as JSON
        s3_key : str
            S3 object key

        Returns:
        --------
        str : S3 key of uploaded file
        """
        try:
            json_str = json.dumps(data, indent=2, default=str)
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✅ Uploaded JSON → s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except Exception as e:
            print(f"❌ JSON upload failed: {e}")
            raise

    def download_json(self, s3_key):
        """
        Download and parse JSON data from S3.

        Parameters:
        -----------
        s3_key : str
            S3 object key

        Returns:
        --------
        dict or list : Parsed JSON data
        """
        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            print(f"✅ Downloaded JSON from s3://{self.bucket_name}/{s3_key}")
            return data

        except Exception as e:
            print(f"❌ JSON download failed: {e}")
            raise


def get_s3_manager():
    """
    Get S3Manager instance with error handling.

    Returns:
    --------
    S3Manager or None : Manager if credentials available, None otherwise
    """
    try:
        return S3Manager()
    except ValueError as e:
        print(f"⚠️ S3 not configured: {e}")
        return None


# Convenience functions
def upload_to_s3(local_path, s3_key=None):
    """Upload a file to S3 (convenience wrapper)."""
    manager = get_s3_manager()
    if manager:
        return manager.upload_file(local_path, s3_key)
    return None


def download_from_s3(s3_key, local_path=None):
    """Download a file from S3 (convenience wrapper)."""
    manager = get_s3_manager()
    if manager:
        return manager.download_file(s3_key, local_path)
    return None


def load_calibrated_parameters(local_path='results/calibrated_parameters.json',
                               s3_key='market-impact-data/results/calibrated_parameters.json'):
    """
    Load calibrated parameters with automatic S3/local fallback.

    On SSP Cloud: tries S3 first, then falls back to local
    Locally: uses local file only

    Parameters:
    -----------
    local_path : str
        Path to local calibrated parameters file
    s3_key : str
        S3 key for calibrated parameters

    Returns:
    --------
    dict or None : Calibrated parameters if found, None otherwise
    """
    import json
    from pathlib import Path

    # Try S3 first if on SSP Cloud
    if is_ssp_cloud():
        print("☁️  Running on SSP Cloud - checking S3 for calibrated parameters...")
        s3 = get_s3_manager()
        if s3 and s3.file_exists(s3_key):
            try:
                # Download to local path
                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(s3_key, local_path)
                with open(local_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load from S3: {e}")

    # Fallback to local file
    if Path(local_path).exists():
        print(f"📂 Loading calibrated parameters from local file: {local_path}")
        with open(local_path, 'r') as f:
            return json.load(f)

    print(f"⚠️  No calibrated parameters found (checked: {local_path})")
    return None
