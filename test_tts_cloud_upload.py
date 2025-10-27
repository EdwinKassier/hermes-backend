#!/usr/bin/env python3
"""
Local test script to verify TTS cloud upload functionality.
Tests the parameter fix for CloudStorageService.upload_file()
"""

import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from app.shared.services.tts.providers.elevenlabs_provider import ElevenLabsTTSProvider


def test_cloud_upload_parameters():
    """Test that upload_file is called with correct parameters"""
    
    print("=" * 80)
    print("Testing TTS Cloud Upload Parameter Fix")
    print("=" * 80)
    
    # Create mock cloud storage service
    mock_cloud_service = Mock()
    mock_cloud_service.upload_file = Mock(return_value="https://storage.googleapis.com/test-bucket/test-audio.mp3")
    
    # Create mock ElevenLabs API
    mock_elevenlabs = MagicMock()
    mock_elevenlabs.generate = Mock(return_value=b"fake_audio_data")
    
    # Create provider with mocked dependencies
    with patch('app.shared.services.tts.providers.elevenlabs_provider.ElevenLabs', return_value=mock_elevenlabs):
        provider = ElevenLabsTTSProvider(
            api_key="test_key",
            cloud_storage_service=mock_cloud_service
        )
    
    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(b"fake_audio_data")
    
    try:
        print(f"\n✓ Created test provider")
        print(f"✓ Mock cloud service created")
        print(f"✓ Temporary file: {tmp_path}")
        
        # Call _handle_cloud_upload directly to test parameter passing
        print("\n" + "-" * 80)
        print("Testing _handle_cloud_upload with cloud upload enabled...")
        print("-" * 80)
        
        result = provider._handle_cloud_upload(
            local_path=tmp_path,
            sample_rate=44100,
            upload_to_cloud=True,
            cloud_destination_path="tts_generated/test_audio.mp3",
            cloud_storage_service_override=None,
            file_extension=".mp3",
            audio_format="mp3_44100_128"
        )
        
        # Verify upload_file was called
        assert mock_cloud_service.upload_file.called, "❌ upload_file was not called"
        print("✓ upload_file was called")
        
        # Get the call arguments
        call_args = mock_cloud_service.upload_file.call_args
        print(f"\n📋 upload_file called with:")
        print(f"   - Args: {call_args.args}")
        print(f"   - Kwargs: {call_args.kwargs}")
        
        # Verify correct parameter names
        assert 'local_file_path' in call_args.kwargs, "❌ Missing 'local_file_path' parameter"
        assert 'destination_blob_name' in call_args.kwargs, "❌ Missing 'destination_blob_name' parameter"
        
        # Verify correct values
        assert call_args.kwargs['local_file_path'] == tmp_path, "❌ Incorrect local_file_path value"
        assert call_args.kwargs['destination_blob_name'] == "tts_generated/test_audio.mp3", "❌ Incorrect destination_blob_name value"
        
        print("\n✅ CORRECT PARAMETERS:")
        print(f"   ✓ local_file_path={call_args.kwargs['local_file_path']}")
        print(f"   ✓ destination_blob_name={call_args.kwargs['destination_blob_name']}")
        
        # Verify result contains cloud_url
        assert 'cloud_url' in result, "❌ Result missing 'cloud_url'"
        assert result['cloud_url'] == "https://storage.googleapis.com/test-bucket/test-audio.mp3", "❌ Incorrect cloud_url"
        print(f"\n✅ RESULT CORRECT:")
        print(f"   ✓ cloud_url={result['cloud_url']}")
        print(f"   ✓ local_path={result['local_path']}")
        print(f"   ✓ sample_rate={result['sample_rate']}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\n✨ Summary:")
        print("   • CloudStorageService.upload_file() called with correct parameters")
        print("   • local_file_path (not local_path)")
        print("   • destination_blob_name (not destination_path)")
        print("   • cloud_url returned successfully")
        print("\n🎉 Fix verified! Safe to deploy.")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"\n✓ Cleaned up temporary file")


def test_without_cloud_upload():
    """Test that skipping cloud upload works"""
    
    print("\n" + "=" * 80)
    print("Testing TTS Without Cloud Upload")
    print("=" * 80)
    
    mock_cloud_service = Mock()
    mock_elevenlabs = MagicMock()
    mock_elevenlabs.generate = Mock(return_value=b"fake_audio_data")
    
    with patch('app.shared.services.tts.providers.elevenlabs_provider.ElevenLabs', return_value=mock_elevenlabs):
        provider = ElevenLabsTTSProvider(
            api_key="test_key",
            cloud_storage_service=mock_cloud_service
        )
    
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(b"fake_audio_data")
    
    try:
        result = provider._handle_cloud_upload(
            local_path=tmp_path,
            sample_rate=44100,
            upload_to_cloud=False,  # Disabled
            cloud_destination_path=None,
            cloud_storage_service_override=None,
            file_extension=".mp3",
            audio_format="mp3_44100_128"
        )
        
        # Verify upload_file was NOT called
        assert not mock_cloud_service.upload_file.called, "❌ upload_file should not be called when upload_to_cloud=False"
        print("✓ upload_file correctly NOT called when upload_to_cloud=False")
        
        # Verify result does NOT contain cloud_url
        assert 'cloud_url' not in result, "❌ Result should not contain 'cloud_url' when upload disabled"
        print("✓ Result correctly omits cloud_url when upload disabled")
        
        print("\n✅ SKIP UPLOAD TEST PASSED!")
        return True
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("\n🧪 Starting Local TTS Cloud Upload Tests\n")
    
    success = True
    
    # Test 1: With cloud upload
    if not test_cloud_upload_parameters():
        success = False
    
    # Test 2: Without cloud upload
    if not test_without_cloud_upload():
        success = False
    
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL LOCAL TESTS PASSED!")
        print("=" * 80)
        print("\n🚀 Ready to deploy to production")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 80)
        print("\n⚠️  Do NOT deploy until tests pass")
        sys.exit(1)

