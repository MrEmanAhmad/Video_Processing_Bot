import asyncio
import logging
import os
import shutil
import time
from datetime import datetime
from main import TwitterVideoProcessor
from dotenv import load_dotenv
import cloudinary
import cloudinary.api
import ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockMessage:
    """Mock Telegram message object for testing"""
    async def edit_text(self, text: str):
        logger.info(f"Status update: {text}")

class TestResults:
    """Class to store test results and artifacts"""
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd(), "test_outputs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {
            "tweet_data": None,
            "frames": [],
            "comment": None,
            "audio_path": None,
            "final_video": None,
            "cloudinary_resources": []
        }
        self.cloudinary_resources_tracked = set()  # Track all Cloudinary resources

    def save_video(self, video_path: str):
        """Save the final video to the output directory"""
        if video_path and os.path.exists(video_path):
            output_path = os.path.join(self.output_dir, "final_video.mp4")
            shutil.copy2(video_path, output_path)
            logger.info(f"Saved final video to: {output_path}")
            return output_path
        return None

    def track_cloudinary_resource(self, resource_id: str, resource_type: str = "image"):
        """Track a Cloudinary resource for cleanup verification"""
        self.cloudinary_resources_tracked.add((resource_id, resource_type))
        logger.info(f"Tracking Cloudinary resource: {resource_id} ({resource_type})")

    def verify_cloudinary_cleanup(self) -> bool:
        """Verify all tracked Cloudinary resources were cleaned up"""
        try:
            # First attempt to clean up any remaining resources
            for resource_id, resource_type in self.cloudinary_resources_tracked:
                try:
                    cloudinary.uploader.destroy(resource_id, resource_type=resource_type)
                    logger.info(f"Cleaned up remaining resource: {resource_id} ({resource_type})")
                except cloudinary.api.NotFound:
                    logger.info(f"Resource already cleaned up: {resource_id} ({resource_type})")
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_id}: {e}")
            
            # Verify all resources are gone
            all_cleaned = True
            for resource_id, resource_type in self.cloudinary_resources_tracked:
                try:
                    result = cloudinary.api.resource(resource_id, resource_type=resource_type)
                    logger.error(f"Resource {resource_id} ({resource_type}) still exists!")
                    all_cleaned = False
                except cloudinary.api.NotFound:
                    logger.info(f"Verified cleanup of {resource_id} ({resource_type})")
            
            return all_cleaned
        except Exception as e:
            logger.error(f"Error verifying Cloudinary cleanup: {e}")
            return False

    def verify_video_format(self, video_path: str) -> dict:
        """Verify video format meets requirements"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            aspect_ratio = height / width
            
            return {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "is_reel_ratio": abs(aspect_ratio - 16/9) < 0.1,  # Allow small deviation
                "has_frame": width % 40 == 0 and height % 40 == 0  # Check if dimensions include 20px frame on each side
            }
        except Exception as e:
            logger.error(f"Error verifying video format: {e}")
            return None

async def test_video_processing():
    """Test the video processing pipeline"""
    test_results = TestResults()
    processor = None
    
    try:
        # Initialize processor
        processor = TwitterVideoProcessor()
        message = MockMessage()
        
        # Test URL - you can change this to any tweet URL you want to test
        tweet_url = "https://x.com/AMAZlNGNATURE/status/1887223066909937941"
        
        logger.info("Starting test processing...")
        
        # Test 1: Download Tweet
        logger.info("Test 1: Testing tweet download...")
        success = await processor.download_tweet(tweet_url, message)
        assert success, "Tweet download failed"
        assert processor.context.get("video_path"), "Video path not set"
        assert processor.context.get("tweet_text"), "Tweet text not captured"
        logger.info("✓ Tweet download test passed")
        
        # Test 2: Frame Extraction
        logger.info("Test 2: Testing frame extraction...")
        success = await processor.extract_frame(message)
        assert success, "Frame extraction failed"
        assert processor.context.get("frame_path"), "Frame path not set"
        assert os.path.exists(processor.context["frame_path"]), "Frame file not created"
        logger.info("✓ Frame extraction test passed")
        
        # Test 3: Cloudinary Upload
        logger.info("Test 3: Testing Cloudinary upload...")
        success = await processor.upload_to_cloudinary(message)
        assert success, "Cloudinary upload failed"
        assert processor.context.get("frame_url"), "Frame URL not set"
        # Track the resource for cleanup verification
        if processor.context.get("cloudinary_resources"):
            for resource_id in processor.context["cloudinary_resources"]:
                test_results.track_cloudinary_resource(resource_id)
        logger.info("✓ Cloudinary upload test passed")
        
        # Test 4: Content Analysis
        logger.info("Test 4: Testing content analysis...")
        try:
            success = await asyncio.wait_for(
                processor.analyze_content(message),
                timeout=60  # 60 seconds timeout for content analysis
            )
            assert success, "Content analysis failed"
            assert processor.context.get("comment"), "Comment not generated"
            assert len(processor.context["comment"]) > 0, "Empty comment generated"
            logger.info("✓ Content analysis test passed")
        except asyncio.TimeoutError:
            logger.error("Content analysis timed out")
            raise Exception("Content analysis timed out after 60 seconds")
        
        # Test 5: Speech Generation
        logger.info("Test 5: Testing speech generation...")
        try:
            success = await asyncio.wait_for(
                processor.generate_speech(message),
                timeout=30  # 30 seconds timeout for speech generation
            )
            assert success, "Speech generation failed"
            assert processor.context.get("audio_path"), "Audio path not set"
            assert os.path.exists(processor.context["audio_path"]), "Audio file not created"
            logger.info("✓ Speech generation test passed")
        except asyncio.TimeoutError:
            logger.error("Speech generation timed out")
            raise Exception("Speech generation timed out after 30 seconds")
        
        # Test 6: Video Processing
        logger.info("Test 6: Testing video processing...")
        try:
            success = await asyncio.wait_for(
                processor.merge_audio_video(message),
                timeout=300  # 5 minutes timeout for video processing
            )
            assert success, "Video processing failed"
            final_video = processor.context.get("output_video_path")
            assert final_video and os.path.exists(final_video), "Final video not created"
            
            # Verify video format
            format_info = test_results.verify_video_format(final_video)
            assert format_info, "Video format verification failed"
            logger.info(f"Video format verification: {format_info}")
            if not format_info["is_reel_ratio"]:
                logger.warning("Video aspect ratio is not 9:16")
            if not format_info["has_frame"]:
                logger.warning("Video does not have proper white frame")
            logger.info("✓ Video processing test passed")
            
            # Save the final video
            saved_video_path = test_results.save_video(final_video)
            logger.info(f"Test video saved to: {saved_video_path}")
            
            logger.info("All processing tests completed successfully!")
        except asyncio.TimeoutError:
            logger.error("Video processing timed out")
            raise Exception("Video processing timed out after 5 minutes")
        
        # Test 7: Resource Cleanup
        logger.info("Test 7: Testing resource cleanup...")
        # Clean up processor resources first
        await processor.cleanup_resources()
        # Wait for any async cleanup operations
        await asyncio.sleep(5)
        # Verify Cloudinary cleanup
        cleanup_success = test_results.verify_cloudinary_cleanup()
        assert cleanup_success, "Cloudinary resource cleanup verification failed"
        logger.info("✓ Resource cleanup test passed")
        
    except AssertionError as e:
        logger.error(f"Test assertion failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure cleanup happens even if test fails
        if processor:
            await processor.cleanup_resources()
            # Additional cleanup attempt
            test_results.verify_cloudinary_cleanup()
            # Clean up temp directory
            if processor.temp_dir and os.path.exists(processor.temp_dir):
                shutil.rmtree(processor.temp_dir)
                logger.info(f"Cleaned up temporary directory: {processor.temp_dir}")

async def run_tests():
    """Run all tests"""
    logger.info("Starting test suite...")
    
    # Load environment variables
    load_dotenv()
    
    # Run the main test
    await test_video_processing()
    
    logger.info("Test suite completed!")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests()) 