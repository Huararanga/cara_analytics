import io
import boto3

_client_cache = {}

def initS3(S3_ENDPOINT,S3_ACCESS_KEY, S3_SECRET_KEY):
    if "S3" not in _client_cache:
        print("Connecting to S3...")
        _client_cache["S3"] = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY
        )
    
def read_from_minio(bucket, key):
    print(bucket, key)
    if "S3" not in _client_cache:
        raise Exception("initS3 must be called first")
    pass

    obj = _client_cache["S3"].get_object(Bucket=bucket, Key=key)
    print(f"✅ Loaded existing {key} from S3 ({bucket})")
    return io.BytesIO(obj["Body"].read());

def read_from_samba(path):
    # Connect using smbprotocol or smbclient
    pass
