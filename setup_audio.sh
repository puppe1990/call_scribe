#!/bin/bash

echo "=========================================="
echo "🎙️  Call Scribe - Audio Setup"
echo "=========================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

echo "📋 This script will help you set up system audio recording."
echo ""

# Check if BlackHole is installed
echo "🔍 Checking for BlackHole..."
if system_profiler SPAudioDataType | grep -q "BlackHole"; then
    echo "✅ BlackHole is installed!"
else
    echo "❌ BlackHole is NOT installed"
    echo ""
    echo "📥 Installing BlackHole..."
    echo ""
    echo "Please follow these steps:"
    echo "1. Download BlackHole from: https://github.com/ExistentialAudio/BlackHole/releases"
    echo "2. Open the downloaded .dmg file"
    echo "3. Install BlackHole 2ch (for mono) or 16ch (for multi-channel)"
    echo "4. Run this script again after installation"
    echo ""
    read -p "Press Enter after installing BlackHole, or 'q' to quit: " response
    if [[ "$response" == "q" ]]; then
        exit 0
    fi
    
    # Check again
    if system_profiler SPAudioDataType | grep -q "BlackHole"; then
        echo "✅ BlackHole detected!"
    else
        echo "⚠️  BlackHole still not detected. Please restart your Mac after installation."
        exit 1
    fi
fi

echo ""
echo "🔧 Configuring audio devices..."
echo ""

# List available audio devices
echo "📱 Available audio output devices:"
system_profiler SPAudioDataType | grep -A 2 "Output Device" | grep ":" | head -5

echo ""
echo "⚠️  IMPORTANT: To record system audio, you need to:"
echo ""
echo "1. Open System Settings > Sound"
echo "2. Set 'Output' to 'BlackHole 2ch' (or BlackHole 16ch)"
echo "3. Keep your speakers/headphones connected for listening"
echo ""
echo "💡 Alternative: Use Multi-Output Device"
echo "   This allows you to hear audio AND record it simultaneously:"
echo ""
echo "   a) Open Audio MIDI Setup (Applications > Utilities > Audio MIDI Setup)"
echo "   b) Click '+' and select 'Create Multi-Output Device'"
echo "   c) Check both 'BlackHole 2ch' and your speakers/headphones"
echo "   d) Set this Multi-Output Device as your system output"
echo ""

read -p "Press Enter to open System Settings, or 'q' to skip: " response
if [[ "$response" != "q" ]]; then
    open "x-apple.systempreferences:com.apple.preference.sound"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Notes:"
echo "   - When recording, select option 3 (Both mic and system audio)"
echo "   - Make sure BlackHole is set as output device before recording"
echo "   - After recording, you can switch back to your normal speakers"
echo ""

