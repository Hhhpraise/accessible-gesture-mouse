# 🤚 Accessible Gesture Mouse Control

**A computer vision-based mouse control system designed for accessibility, using simple hand gestures to control cursor movement, clicking, and scrolling.**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-red.svg)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### Simple & Intuitive Gestures
- **✋ Open Hand** → Move cursor smoothly
- **✊ Closed Fist** → Left click
- **👆 Point (1 finger)** → Right click  
- **✌️ Two Fingers + Swipe** → Scroll up/down

### Accessibility-First Design
- 🎯 **Single hand operation** - Works with just one hand
- 🔧 **Customizable thresholds** - Adjustable for different motor abilities
- 🖼️ **Visual feedback** - Clear on-screen indicators for all gestures
- ⚡ **Low latency** - Optimized for real-time responsiveness
- 🛡️ **Error prevention** - Gesture stabilization prevents accidental clicks

### Smart Features
- 📏 **Control zone** - Define specific area for cursor control
- 🎛️ **Velocity-based scrolling** - Faster swipes = faster scrolling
- 🔄 **Gesture smoothing** - Reduces jitter and false positives
- ⏸️ **Toggle control** - Press SPACE to disable/enable
- 📊 **Real-time feedback** - See gesture recognition in action

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- Webcam/Camera
- Good lighting conditions

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/hhhpraise/accessible-gesture-mouse.git
cd accessible-gesture-mouse
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python gesture_mouse.py
```

## 📋 Usage Instructions

### Getting Started
1. **Position yourself** in front of your camera with good lighting
2. **Keep your hand within the green control zone** displayed on screen
3. **Use the gesture controls** as shown in the interface

### Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| ✋ **Open Hand** | Move Cursor | Spread all fingers to control cursor movement |
| ✊ **Closed Fist** | Left Click | Make a fist and hold briefly to click |
| 👆 **Point** | Right Click | Point with index finger only |
| ✌️ **Two Fingers** | Scroll | Hold up index + middle finger, then swipe up/down |

### Keyboard Shortcuts
- **SPACE** - Toggle mouse control on/off
- **Q** - Quit application

### Tips for Best Results
- 🔆 **Ensure good lighting** - Natural light or bright room lighting works best
- 📏 **Stay within control zone** - The green rectangle shows the active area
- ⏱️ **Hold gestures briefly** - Wait for gesture recognition before releasing
- 🤲 **Use one hand** - System is optimized for single-hand operation
- 💡 **Practice gestures** - Start with simple movements and build muscle memory

## 🛠️ Customization

### Adjusting Sensitivity
Edit the `GestureConfig` class in `gesture_mouse.py`:

```python
class GestureConfig:
    # Gesture thresholds (0.0 to 1.0)
    FIST_THRESHOLD = 0.6        # How closed for fist detection
    OPEN_HAND_THRESHOLD = 0.8   # How open for cursor movement
    
    # Timing controls (seconds)
    GESTURE_HOLD_TIME = 0.3     # Time to hold gesture before activation
    CLICK_COOLDOWN = 0.5        # Minimum time between clicks
    
    # Control zone (screen percentage)
    CONTROL_ZONE = {
        'x_min': 0.2, 'x_max': 0.8,  # Left to right boundaries
        'y_min': 0.1, 'y_max': 0.7   # Top to bottom boundaries
    }
```

### Camera Settings
```python
# Adjust for your camera's capabilities
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

## 🏥 Accessibility Features

This application was designed with accessibility in mind:

### For Users with Limited Mobility
- **Reduced precision requirements** - Forgiving gesture recognition
- **Single hand operation** - No need for complex two-handed gestures
- **Customizable control zone** - Adjust active area to comfortable range
- **Hold-time activation** - Prevents accidental clicks from tremors

### For Users with Motor Impairments
- **Gesture smoothing** - Reduces impact of hand tremors
- **Velocity-based controls** - Adapts to user's natural movement speed
- **Visual feedback** - Clear indication of recognized gestures
- **Emergency disable** - Quick way to stop mouse control

### For Users with Different Abilities
- **Configurable thresholds** - Adjust sensitivity for different hand sizes/mobility
- **Multiple gesture options** - Choose the most comfortable gestures
- **Progressive difficulty** - Start with simple gestures, add complexity as needed

## 🔧 Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check available cameras
python -c "import cv2; print('Camera 0:', cv2.VideoCapture(0).isOpened())"
```

**Poor gesture recognition:**
- Ensure good lighting (avoid backlighting)
- Keep hand within the green control zone
- Check camera focus and cleanliness
- Adjust gesture thresholds in config

**High CPU usage:**
- Reduce camera resolution in config
- Close other camera applications
- Use lighter MediaPipe model complexity

**Gestures not responding:**
- Hold gestures longer (increase `GESTURE_HOLD_TIME`)
- Check if mouse control is enabled (press SPACE)
- Verify hand is clearly visible to camera

## 📖 Technical Details

### Dependencies
- **OpenCV** (cv2) - Computer vision and camera handling
- **MediaPipe** - Hand landmark detection and tracking
- **PyAutoGUI** - System mouse control
- **NumPy** - Mathematical operations and smoothing

### How It Works
1. **Camera Capture** - Captures video feed from webcam
2. **Hand Detection** - MediaPipe detects hand landmarks in real-time
3. **Gesture Recognition** - Analyzes finger positions to classify gestures
4. **Smoothing** - Applies filters to reduce noise and jitter
5. **Action Execution** - Converts gestures to mouse actions via PyAutoGUI

### Performance Optimizations
- **Lightweight MediaPipe model** - Faster processing with minimal accuracy loss
- **Efficient coordinate mapping** - Direct screen coordinate calculation
- **Smart smoothing** - Weighted averaging for natural cursor movement
- **Gesture stabilization** - Prevents false positives from brief movements

## 🤝 Contributing

Contributions are welcome! Here are ways you can help:

### Ideas for Contributions
- 🎨 **UI Improvements** - Better visual feedback and controls
- 🔧 **New Gestures** - Additional useful gesture patterns
- 📊 **Calibration System** - Auto-adjust settings for different users
- 🌐 **Multi-language Support** - Translate interface and documentation
- 🔍 **Testing** - Test with different hardware and accessibility needs
- 📚 **Documentation** - Improve setup guides and tutorials

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly with different gestures and conditions
5. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team** - For excellent hand tracking technology
- **OpenCV Community** - For computer vision tools
- **Accessibility Community** - For feedback and testing
- **Contributors** - Everyone who helps improve this project

## 📞 Support

- 🐛 **Found a bug?** Open an [issue](https://github.com/yourusername/accessible-gesture-mouse/issues)
- 💡 **Have an idea?** Start a [discussion](https://github.com/yourusername/accessible-gesture-mouse/discussions)
- ❓ **Need help?** Check the [troubleshooting section](#-troubleshooting) or ask in discussions

---

**Made with ❤️ for accessibility and inclusion**

*If this project helps you or someone you know, please consider giving it a ⭐ star!*