from setuptools import setup, find_packages

setup(
    name="video-processing-bot",
    version="1.0.0",
    packages=find_packages(include=['pipeline', 'pipeline.*']),
    package_data={
        'pipeline': ['*.py'],
    },
    include_package_data=True,
    install_requires=[
        'openai>=1.0.0',
        'python-dotenv>=0.19.0',
        'yt-dlp>=2023.11.16',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'google-cloud-vision>=3.4.4',
        'google-cloud-texttospeech>=2.14.1',
        'cloudinary>=1.36.0',
        'requests>=2.31.0',
        'selenium>=4.15.2',
        'webdriver-manager>=4.0.1',
        'undetected-chromedriver>=3.5.3',
        'python-telegram-bot>=20.7',
        'psutil>=5.9.0'
    ],
    python_requires='>=3.10',
) 