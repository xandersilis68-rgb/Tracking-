# AI Volleyball Scoreboard

An AI-powered volleyball scoreboard that uses a webcam to detect and track a volleyball and automatically scores points based on where the ball hits the ground. Built with Python, OpenCV, and NumPy.

## Features
- Detects a volleyball using color-based tracking (HSV).
- Tracks ball movement and displays a trail.
- Scores points for Team A (right side) or Team B (left side) when the ball hits the ground.
- Displays scores and a court midline (simulated net) on the video feed.

## Prerequisites
- Python 3.6+
- A webcam
- A yellow/orange volleyball (adjust HSV ranges if using a different color)
- A side view of the volleyball court with the net in the middle

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-volleyball-scoreboard.git
   cd ai-volleyball-scoreboard
