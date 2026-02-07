"""
Speaker Identification System - Main Entry Point

This is the main entry point for the speaker identification system.
It initializes all components and launches the Gradio interface.

Usage:
    python main.py

The application will start a web server accessible at:
    http://localhost:7860
"""

import sys
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from src.ui.gradio_app import create_app


def main():
    """
    Main function to launch the application.
    """
    print("=" * 60)
    print("Speaker Identification System")
    print("=" * 60)
    print()
    print("System Requirements:")
    print("  - Python 3.8+")
    print("  - Microphone access")
    print("  - GPU (optional, will use CPU otherwise)")
    print()
    print("Starting application...")
    print()
    
    try:
        # Create Gradio app
        demo = create_app()
        
        # Launch app
        print("=" * 60)
        print("Application ready!")
        print("=" * 60)
        print()
        print("Open your web browser and go to:")
        print("  → http://localhost:7860")
        print()
        print("Press Ctrl+C to stop the server.")
        print()
        
        # Try different queue configurations for Gradio compatibility
        try:
            demo.queue(concurrency_count=5).launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                quiet=False
            )
        except TypeError:
            # Fallback for newer Gradio without concurrency_count
            try:
                demo.queue(max_size=20).launch(
                    server_name="127.0.0.1",
                    server_port=7860,
                    share=False,
                    quiet=False
                )
            except TypeError:
                # Last resort - no queue arguments
                demo.queue().launch(
                    server_name="127.0.0.1",
                    server_port=7860,
                    share=False,
                    quiet=False
                )
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
